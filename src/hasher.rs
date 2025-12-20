use crate::{cpu, error::HasherError, gpu};
use std::io::{Read, Seek, Write};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

pub enum HasherResult {
    Found(u32, u32),
    Continue,
    End,
}

pub struct Hasher {
    cpu: cpu::CPUHasher,
    gpus: Vec<gpu::GPUHasher>,
    target_checksum: u64,
    y_bits: Vec<u32>,
    y_base: u32,
}

enum ThreadOutcome {
    Found {
        y: u32,
        x: u32,
        duration: Duration,
    },
    Exhausted {
        y: u32,
        duration: Duration,
    },
}

impl Hasher {
    pub fn new(
        path: std::path::PathBuf,
        gpu_adapters: Option<Vec<usize>>,
        workgroups: (u32, u32, u32),
        shader: gpu::GPUHasherShader,
        seed: u8,
        target_checksum: u64,
        y_bits: Vec<u32>,
        y_init: u32,
    ) -> Result<Self, HasherError> {
        let ipl3 = Self::load_ipl3(path)?;

        let cpu = cpu::CPUHasher::new(&ipl3, seed);

        let adapters = gpu::GPUHasher::list_gpu_adapters();
        let selected_adapters: Vec<usize> = gpu_adapters
            .unwrap_or_else(|| (0..adapters.len()).collect());

        if selected_adapters.is_empty() {
            return Err(HasherError::GPUAdapterOutOfBounds);
        }

        let required_features =
            wgpu::Features::PUSH_CONSTANTS | wgpu::Features::SHADER_INT64;

        let mut gpus = Vec::with_capacity(selected_adapters.len());
        let mut skipped: Vec<(wgpu::AdapterInfo, &'static str)> = Vec::new();

        for adapter_idx in selected_adapters {
            let adapter = adapters
                .get(adapter_idx)
                .ok_or(HasherError::GPUAdapterOutOfBounds)?;
            let info = adapter.get_info();

            match info.device_type {
                wgpu::DeviceType::Cpu | wgpu::DeviceType::Other => {
                    skipped.push((info, "device type is CPU/Other"));
                    continue;
                }
                _ => {}
            };

            if adapter.features().contains(required_features) {
                gpus.push(gpu::GPUHasher::new(adapter.clone(), shader, workgroups)?);
            } else {
                skipped.push((info, "missing required features"));
            }
        }

        if gpus.is_empty() {
            eprintln!(
                "No compatible GPUs: require {:?}. Skipped adapters:",
                required_features
            );
            for (info, reason) in skipped {
                eprintln!("  \"{}\", backend: \"{}\" ({})", info.name, info.backend, reason);
            }
            return Err(HasherError::NoCompatibleGpu);
        }

        if !skipped.is_empty() {
            eprintln!("Warning: skipped GPUs without required features:");
            for (info, reason) in skipped {
                eprintln!("  \"{}\", backend: \"{}\" ({})", info.name, info.backend, reason);
            }
        }

        Ok(Self {
            cpu,
            gpus,
            target_checksum,
            y_bits,
            y_base: y_init,
        })
    }

    fn load_ipl3(path: std::path::PathBuf) -> Result<[u8; 4032], HasherError> {
        let mut f = std::fs::File::open(path)?;

        let mut ipl3 = [0u8; 4032];

        f.seek(std::io::SeekFrom::Start(64))?;
        f.read_exact(&mut ipl3)?;

        Ok(ipl3)
    }

    pub fn sign_rom(
        path: std::path::PathBuf,
        y_bits: Vec<u32>,
        y: u32,
        x: u32,
    ) -> Result<(), HasherError> {
        let mut f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)?;

        for (i, offset) in y_bits.iter().enumerate() {
            let index = offset / 8;
            let bit = 7 - (offset % 8);
            let shift = y_bits.len() - 1;
            let value = ((y >> (shift - i)) & (1 << 0)) as u8;

            let mut byte = [0u8];

            f.seek(std::io::SeekFrom::Start(index as u64 + 64))?;
            f.read_exact(&mut byte)?;

            byte[0] &= !(1 << bit);
            byte[0] |= value << bit;

            f.seek(std::io::SeekFrom::Current(-1))?;
            f.write_all(&mut byte)?;
        }

        f.seek(std::io::SeekFrom::Start(4092))?;
        f.write_all(&mut x.to_be_bytes().to_vec())?;

        f.flush()?;

        Ok(())
    }

    pub fn get_y(&self) -> u32 {
        self.y_base
    }

    fn max_y(&self) -> u32 {
        ((1u64 << self.y_bits.len()) - 1) as u32
    }

    fn is_y_finished(&self, y: u32) -> bool {
        (y as u64) > ((1u64 << self.y_bits.len()) - 1)
    }

    pub fn get_gpu_infos(&self) -> Vec<wgpu::AdapterInfo> {
        self.gpus
            .iter()
            .map(|gpu| gpu.get_gpu_info())
            .collect()
    }

    pub fn compute_round(&mut self) -> Result<HasherResult, HasherError> {
        if self.is_y_finished(self.y_base) {
            return Ok(HasherResult::End);
        }

        if self.gpus.is_empty() {
            return Err(HasherError::GPUAdapterOutOfBounds);
        }

        let max_y = self.max_y();
        let start_y = self.y_base;

        let (found, next_y) = self.run_and_stream(start_y, max_y)?;

        if let Some((y, x)) = found {
            return Ok(HasherResult::Found(y, x));
        }

        self.y_base = next_y;

        if self.is_y_finished(self.y_base) {
            Ok(HasherResult::End)
        } else {
            Ok(HasherResult::Continue)
        }
    }

    fn run_and_stream(
        &mut self,
        start_y: u32,
        max_y: u32,
    ) -> Result<(Option<(u32, u32)>, u32), HasherError> {
        let target_checksum = self.target_checksum;
        let cpu = &self.cpu;
        let y_bits = self.y_bits.clone();

        let next_y = AtomicU32::new(start_y);
        let found_flag = AtomicBool::new(false);

        let (tx, rx): (
            mpsc::Sender<Result<ThreadOutcome, HasherError>>,
            mpsc::Receiver<Result<ThreadOutcome, HasherError>>,
        ) = mpsc::channel();

        let mut next_print_y = start_y;
        let mut pending: HashMap<u32, ThreadOutcome> = HashMap::new();
        let mut found_result: Option<(u32, u32)> = None;
        let mut recv_error: Option<HasherError> = None;

        thread::scope(|scope| {
            for gpu in self.gpus.iter_mut() {
                let tx = tx.clone();
                let cpu = cpu;
                let y_bits = y_bits.clone();
                let gpu = gpu;
                let found_flag = &found_flag;
                let next_y = &next_y;
                scope.spawn(move || {
                    let mut work = || -> Result<(), HasherError> {
                        loop {
                            if found_flag.load(Ordering::Acquire) {
                                break;
                            }

                            let y_current = next_y.fetch_add(1, Ordering::Relaxed);

                            if y_current > max_y {
                                break;
                            }

                            let (y_offset, state) = cpu.y_round(y_bits.clone(), y_current);
                            let start = Instant::now();
                            let mut x_offset = 0u32;

                            loop {
                                let result = gpu.x_round(
                                    target_checksum,
                                    y_offset,
                                    x_offset,
                                    state,
                                )?;

                                match result {
                                    gpu::GPUHasherResult::Found(x) => {
                                        let duration = start.elapsed();
                                        found_flag.store(true, Ordering::Release);
                                        let _ = tx.send(Ok(ThreadOutcome::Found {
                                            y: y_current,
                                            x,
                                            duration,
                                        }));
                                        return Ok(());
                                    }
                                    gpu::GPUHasherResult::Continue(x_step) => {
                                        let next = x_offset as u64 + x_step as u64;
                                        if next > u32::MAX as u64 {
                                            break;
                                        }
                                        x_offset = next as u32;
                                    }
                                    gpu::GPUHasherResult::End => break,
                                }
                            }

                            let duration = start.elapsed();
                            let _ = tx.send(Ok(ThreadOutcome::Exhausted {
                                y: y_current,
                                duration,
                            }));
                        }

                        Ok(())
                    };

                    if let Err(err) = work() {
                        let _ = tx.send(Err(err));
                    }
                });
            }

            drop(tx);

            for msg in rx {
                let outcome = match msg {
                    Ok(outcome) => outcome,
                    Err(err) => {
                        recv_error = Some(err);
                        break;
                    }
                };

                let y = match outcome {
                    ThreadOutcome::Found { y, .. } => y,
                    ThreadOutcome::Exhausted { y, .. } => y,
                };

                pending.insert(y, outcome);

                while let Some(outcome) = pending.remove(&next_print_y) {
                    match outcome {
                        ThreadOutcome::Found { y, x, duration } => {
                            println!("Y={} took {:?}", y, duration);
                            if found_result.is_none() {
                                found_result = Some((y, x));
                            }
                        }
                        ThreadOutcome::Exhausted { y, duration } => {
                            println!("Y={} took {:?}", y, duration);
                        }
                    }
                    next_print_y = next_print_y.saturating_add(1);
                    if found_result.is_some() {
                        break;
                    }
                }
                if found_result.is_some() {
                    break;
                }
            }
        });

        if let Some(err) = recv_error {
            return Err(err);
        }

        // Drain any remaining pending entries after channel close.
        while found_result.is_none() {
            if let Some(outcome) = pending.remove(&next_print_y) {
                match outcome {
                    ThreadOutcome::Found { y, x, duration } => {
                        println!("Y={} took {:?}", y, duration);
                        if found_result.is_none() {
                            found_result = Some((y, x));
                        }
                    }
                    ThreadOutcome::Exhausted { y, duration } => {
                        println!("Y={} took {:?}", y, duration);
                    }
                }
                next_print_y = next_print_y.saturating_add(1);
            } else {
                break;
            }
        }

        if let Some((y, x)) = found_result {
            let verify_checksum = self.cpu.verify(self.y_bits.clone(), y, x);
            if verify_checksum != self.target_checksum {
                return Err(HasherError::ChecksumVerifyError(y, x, verify_checksum));
            }
            return Ok((Some((y, x)), next_print_y));
        }

        let next_y_value = next_y.load(Ordering::Relaxed);
        let capped_next_y = if next_y_value > max_y {
            max_y.saturating_add(1)
        } else {
            next_y_value.max(next_print_y)
        };

        Ok((None, capped_next_y))
    }
}
