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
    y_bits: Vec<u32>,
    y: u32,
}

impl Hasher {
    pub fn new(
        path: std::path::PathBuf,
        gpu_adapter_id: usize,
        workgroup_configuration: (u32, u32, u32),
        seed: u8,
        target_checksum: u64,
        y_bits: Vec<u32>
    ) -> Result<Self, HasherError> {
        let ipl3 = Self::load_ipl3(path)?;

        let cpu = cpu::CPUHasher::new(&ipl3, seed, y_bits.clone());

        let adapters = gpu::GPUHasher::list_gpu_adapters();
        let adapter = adapters
            .get(gpu_adapter_id)
            .ok_or(HasherError::GPUAdapterOutOfBounds)?;
        let gpu = gpu::GPUHasher::new(adapter.clone())?;

        // print adapter info
        println!("GPU: {:?}", adapter.get_info());

        Ok(Self {
            cpu,
            gpu,
            workgroup_configuration,
            target_checksum,
            y_bits,
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

    pub fn sign_rom(path: std::path::PathBuf, y_bits: Vec<u32>, y: u32, x: u32) -> Result<(), HasherError> {
        let mut f = std::fs::OpenOptions::new().write(true).read(true).open(path)?;
        for i in 0..y_bits.len() {
            let byte_index = y_bits[i] / 8;
            let bit_offset = y_bits[i] % 8;
            f.seek(std::io::SeekFrom::Start(byte_index as u64))?;
            
            let mut byte = [0u8; 1];
            f.read_exact(&mut byte)?;
            let mask = 1 << bit_offset;
            if (y >> i) & 1 == 1 {
                byte[0] |= mask;
            } else {
                byte[0] &= !mask;
            }
            f.seek(std::io::SeekFrom::Start(byte_index as u64))?;
            f.write_all(&byte)?;
        }
        
        f.seek(std::io::SeekFrom::Start(4092))?;
        let mut data: Vec<u8> = vec![];
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
        if self.y as u64 > (1u64 << self.y_bits.len()) - 1 {
            return Ok(HasherResult::End);
        }

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

        if self.y as u64 == (1u64 << self.y_bits.len()) - 1 {
            return Ok(HasherResult::End);
        }

        self.y += 1;

        Ok(HasherResult::Continue)
    }
}
