mod cli;
mod cpu;
mod gpu;
mod hasher;

use hasher::*;

fn print_execution_time(y: u32, time: std::time::Instant) {
    println!("Y == {} took {:?}", y, time.elapsed());
}

fn run_hasher() -> Result<(), HasherError> {
    let cli::Cli {
        rom,
        sign,
        cic,
        y_init,
        gpu_adapter,
        workgroups,
        y_bits
    } = cli::parse();
    let (seed, target_checksum) = cic;

    let mut hasher = Hasher::new(
        rom.clone().into(),
        gpu_adapter,
        workgroups,
        seed,
        target_checksum,
        y_bits.clone(),
    )?;

    if let Some(y_init) = y_init {
        hasher.set_y(y_init);
    }

    loop {
        let time = std::time::Instant::now();

        let y_current = hasher.get_y();

        match hasher.compute_round()? {
            HasherResult::Found(y, x) => {
                print_execution_time(y_current, time);
                println!("Found collision: Y={y:08X} X={x:08X}");
                if sign {
                    Hasher::sign_rom(rom.into(), y_bits, y, x)?;
                    println!("ROM has been successfully signed");
                }
                return Ok(());
            }
            HasherResult::Continue => {
                print_execution_time(y_current, time);
            }
            HasherResult::End => {
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
