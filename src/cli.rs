use clap::Parser;

#[derive(Parser)]
pub struct Cli {
    /// Path to the source ROM file with IPL3 to be brute forced
    pub rom: std::path::PathBuf,

    /// Sign the source ROM file with found collision data
    #[arg(short = 's', long)]
    pub sign: bool,

    /// The CIC for which a checksum must be calculated
    #[arg(short = 'c', long, default_value("6102"), value_parser = cic_parser)]
    pub cic: (u8, u64),

    /// The Y coordinate to start with
    #[arg(short = 'y', long)]
    pub y_init: Option<u32>,

    /// Y bits to use: 32-bit word indices and bit ranges (eg: 40[8..16],56[12..24]).
    #[arg(short = 'b', long, default_value("1022[0..31]"), value_parser = ybits_parser)]
    pub y_bits: std::vec::Vec<u32>,

    /// The GPU to use (0 for first, 1 for second, etc.)
    #[arg(short = 'd', long, default_value("0"))]
    pub gpu_adapter: usize,

    /// The number of workgroups to use (x,y,z format, total threads = x*y*z*256)
    #[arg(short = 'w', long, default_value("256,256,256"), value_parser = workgroups_parser)]
    pub workgroups: (u32, u32, u32),
}

fn ybits_parser(str: &str) -> Result<Vec<u32>, String> {
    let slices: Vec<&str> = str.split(',').collect();

    if slices.len() == 0 {
        return Err("invalid format".to_string());
    }

    let mut values = vec![];

    for slice in slices.iter() {
        if slice.contains('[') {
            let parts: Vec<&str> = slice.split('[').collect();
            if parts.len() != 2 {
                return Err("invalid format".to_string());
            }
            let index = u32::from_str_radix(parts[0], 10).map_err(|e| e.to_string())?;
            if index <= 16 || index >= 1023 {
                return Err(format!("invalid Y word index: {}", index));
            }

            let range = parts[1].trim_end_matches(']');
            let range_parts: Vec<&str> = range.split("..").collect();
            if range_parts.len() != 2 {
                return Err("invalid format".to_string());
            }
            let start = u32::from_str_radix(range_parts[0], 10).map_err(|e| e.to_string())?;
            let end = u32::from_str_radix(range_parts[1], 10).map_err(|e| e.to_string())?;

            if start > end {
                return Err("invalid range".to_string());
            }
            if end >= 32 {
                return Err("invalid range".to_string());
            }

            for i in start..=end {
                values.push((index - 16)*32 + i);
            }
        } else {
            let index = u32::from_str_radix(slice, 10).map_err(|e| e.to_string())?;
            if index <= 16 || index >= 1023 {
                return Err(format!("invalid Y word index: {}", index));
            }

            for i in 0..=31 {
                values.push((index - 16)*32 + i);
            }
        }
    }

    if values.len() > 32 {
        return Err(format!("too many Y bits: {} (max: 32)", values.len()));
    }

    values.sort();

    Ok(values)
}

fn cic_parser(str: &str) -> Result<(u8, u64), String> {
    let (seed, target_checksum) = match str {
        "6101" => (0x3F, 0x45CC73EE317A),
        "6102" | "7101" => (0x3F, 0xA536C0F1D859),
        "6103" | "7103" => (0x78, 0x586FD4709867),
        "6105" | "7105" => (0x91, 0x8618A45BC2D3),
        "6106" | "7106" => (0x85, 0x2BBAD4E6EB74),
        "8303" => (0xDD, 0x32B294E2AB90),
        "8401" => (0xDD, 0x6EE8D9E84970),
        "5167" => (0xDD, 0x083C6C77E0B1),
        "DDUS" => (0xDE, 0x05BA2EF0A5F1),
        _ => return Err("Unknown CIC".to_string()),
    };
    Ok((seed, target_checksum))
}

fn workgroups_parser(str: &str) -> Result<(u32, u32, u32), String> {
    let slices: Vec<&str> = str.split(',').collect();

    if slices.len() == 0 || slices.len() > 3 {
        return Err("invalid format".to_string());
    }

    let mut values = [1u32; 3];

    for (i, slice) in slices.iter().enumerate() {
        values[i] = u32::from_str_radix(&slice, 10).map_err(|e| e.to_string())?;
    }

    Ok((values[0], values[1], values[2]))
}

pub fn parse() -> Cli {
    Cli::parse()
}
