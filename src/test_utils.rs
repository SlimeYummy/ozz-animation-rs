#![allow(dead_code)]

use bincode::{Decode, Encode};
use std::env::consts::{ARCH, OS};
use std::error::Error;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::sync::OnceLock;
use std::{env, mem};

const MAX_CHUNK_SIZE: usize = 1024;

// f16 -> f32
// ignore overflow, infinite, NaN
pub fn f16(f: f32) -> u16 {
    let n = unsafe { mem::transmute::<f32, u32>(f) };
    if (n & 0x7FFFFFFF) == 0 {
        return (n >> 16) as u16;
    }
    let sign = (n >> 16) & 0x8000;
    let expo = (((n & 0x7f800000) - 0x38000000) >> 13) & 0x7c00;
    let base = (n >> 13) & 0x03ff;
    return (sign | expo | base) as u16;
}

static FOLDER: OnceLock<()> = OnceLock::new();

pub fn save_to_file<T: Encode>(folder: &str, name: &str, data: &T) -> Result<(), Box<dyn Error>> {
    FOLDER.get_or_init(|| {
        fs::create_dir_all(format!("./expected/{}", folder)).unwrap();
        fs::create_dir_all(format!("./output/{}", folder)).unwrap();
    });

    let buf = bincode::encode_to_vec(data, bincode::config::standard())?;
    let path = if env::var("SAVE_TO_EXPECTED").is_ok() {
        format!("./expected/{0}/{0}_{1}", folder, name)
    } else {
        format!("./output/{0}/{0}_{1}_{2}_{3}", folder, OS, ARCH, name)
    };
    let mut file = File::create(path)?;
    file.write_all(&buf)?;
    return Ok(());
}

pub fn compare_with_file<T: Decode + PartialEq>(folder: &str, name: &str, data: &T) -> Result<(), Box<dyn Error>> {
    if env::var("SAVE_TO_EXPECTED").is_ok() {
        return Ok(());
    }
    let path = format!("./expected/{0}/{0}_{1}", folder, name);
    let mut file = File::open(&path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    let (expected, _): (T, usize) = bincode::decode_from_slice(&buf, bincode::config::standard())?;
    if data != &expected {
        return Err(format!("compare_with_file({})", path).into());
    }
    return Ok(());
}
