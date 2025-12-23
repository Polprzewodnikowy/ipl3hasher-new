mod cli;
mod cpu;
mod error;
mod gpu;
mod hasher;

fn format_duration(d: std::time::Duration) -> String {
    let secs = d.as_secs();
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    let millis = d.subsec_millis();
    format!("{hours:02}:{minutes:02}:{seconds:02}.{millis:03}")
}

fn run_hasher() -> Result<(), error::HasherError> {
    let cli::Cli {
        rom,
        sign,
        cic,
        y_bits,
        y_init,
        gpu_adapters,
        workgroups,
        shader,
    } = cli::parse();
    let (seed, target_checksum) = cic;

    let shader = match shader {
        cli::ShaderType::Glsl => gpu::GPUHasherShader::Glsl,
        cli::ShaderType::Wgsl => gpu::GPUHasherShader::Wgsl,
    };

    let mut hasher = hasher::Hasher::new(
        rom.clone().into(),
        gpu_adapters,
        workgroups,
        shader,
        seed,
        target_checksum,
        y_bits.clone(),
        y_init,
    )?;

    let gpu_infos = hasher.get_gpu_infos();

    println!("GPUs in use:");
    for gpu_info in gpu_infos {
        println!("  \"{}\", backend: \"{}\"", gpu_info.name, gpu_info.backend);
    }

    println!("Target seed and checksum: 0x{seed:02X} 0x{target_checksum:012X}");

    let total_start = std::time::Instant::now();

    loop {
        match hasher.compute_round()? {
            hasher::HasherResult::Found(y, x) => {
                let total = total_start.elapsed();
                println!("Found collision: Y={y:08X} X={x:08X}");
                println!("Total time: {}", format_duration(total));
                if sign {
                    hasher::Hasher::sign_rom(rom.into(), y_bits, y, x)?;
                    println!("ROM has been successfully signed");
                }
                return Ok(());
            }
            hasher::HasherResult::Continue => {}
            hasher::HasherResult::End => {
                let total = total_start.elapsed();
                println!("Total time: {}", format_duration(total));
                break;
            }
        }
    }

    println!("Sorry nothing");

    Ok(())
}

fn main() {
    if let Err(error) = run_hasher() {
        println!("IPL3 hasher error: {error}");
    }
}
