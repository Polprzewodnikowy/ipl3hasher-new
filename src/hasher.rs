use crate::{cpu, gpu};
use std::io::{Read, Seek, Write};

pub enum HasherResult {
    Found(u32, u32),
    Continue,
    End,
}

#[derive(Debug)]
pub enum HasherError {
    VerifyError(u32, u32, u64),
    GPUAdapterOutOfBounds,
    GPUHasherError(gpu::GPUHasherError),
    IoError(std::io::Error),
}

impl std::fmt::Display for HasherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerifyError(y, x, verify_checksum) => f.write_fmt(format_args!(
                "GPU Hasher result is wrong: {y:08X} {x:08X} | 0x{verify_checksum:012X}"
            )),
            Self::GPUAdapterOutOfBounds => f.write_str("Selected GPU adapter doesn't exist"),
            Self::GPUHasherError(error) => f.write_str(error.to_string().as_str()),
            Self::IoError(error) => f.write_str(error.to_string().as_str()),
        }
    }
}

impl From<gpu::GPUHasherError> for HasherError {
    fn from(value: gpu::GPUHasherError) -> Self {
        Self::GPUHasherError(value)
    }
}

impl From<std::io::Error> for HasherError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

pub struct Hasher {
    cpu: cpu::CPUHasher,
    gpu: gpu::GPUHasher,
    workgroup_configuration: (u32, u32, u32),
    target_checksum: u64,
    y: u32,
}

impl Hasher {
    pub fn new(
        path: std::path::PathBuf,
        gpu_adapter_id: usize,
        workgroup_configuration: (u32, u32, u32),
        seed: u8,
        target_checksum: u64,
    ) -> Result<Self, HasherError> {
        let ipl3 = Self::load_ipl3(path)?;

        let cpu = cpu::CPUHasher::new(&ipl3, seed);

        let adapters = gpu::GPUHasher::list_gpu_adapters();
        let adapter = adapters
            .get(gpu_adapter_id)
            .ok_or(HasherError::GPUAdapterOutOfBounds)?;
        let gpu = gpu::GPUHasher::new(adapter.clone())?;

        Ok(Self {
            cpu,
            gpu,
            workgroup_configuration,
            target_checksum,
            y: 0,
        })
    }

    fn load_ipl3(path: std::path::PathBuf) -> Result<[u8; 4032], HasherError> {
        let mut f = std::fs::File::open(path)?;
        let mut ipl3 = [0u8; 4032];
        f.seek(std::io::SeekFrom::Start(64))?;
        f.read_exact(&mut ipl3)?;
        Ok(ipl3)
    }

    pub fn sign_rom(path: std::path::PathBuf, y: u32, x: u32) -> Result<(), HasherError> {
        let mut f = std::fs::OpenOptions::new().write(true).open(path)?;
        f.seek(std::io::SeekFrom::Start(4088))?;
        let mut data: Vec<u8> = vec![];
        data.append(&mut y.to_be_bytes().to_vec());
        data.append(&mut x.to_be_bytes().to_vec());
        f.write_all(&data)?;
        f.flush()?;
        Ok(())
    }

    pub fn set_y(&mut self, y: u32) {
        self.y = y;
    }

    pub fn get_y(&self) -> u32 {
        self.y
    }

    pub fn compute_round(&mut self) -> Result<HasherResult, HasherError> {
        let state = self.cpu.y_round(self.y);

        let mut x_offset = 0;

        loop {
            let result = self.gpu.x_round(
                self.target_checksum,
                state,
                self.y,
                x_offset,
                self.workgroup_configuration,
            )?;

            match result {
                gpu::GPUHasherResult::Found(y, x) => {
                    let verify_checksum = self.cpu.verify(y, x);
                    if verify_checksum != self.target_checksum {
                        return Err(HasherError::VerifyError(y, x, verify_checksum));
                    }
                    return Ok(HasherResult::Found(y, x));
                }
                gpu::GPUHasherResult::Continue(x_step) => {
                    x_offset += x_step;
                }
                gpu::GPUHasherResult::End => {
                    break;
                }
            }
        }

        if self.y == u32::MAX {
            return Ok(HasherResult::End);
        }

        self.y += 1;

        Ok(HasherResult::Continue)
    }

    pub fn get_gpu_info(&self) -> wgpu::AdapterInfo {
        self.gpu.get_gpu_info()
    }
}
