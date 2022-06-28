#![allow(dead_code)]

use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::Read;
use std::mem;
use std::path::Path;

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

pub fn read_chunk<P, T>(path: P) -> Result<Vec<T>>
where
    P: AsRef<Path>,
    T: Sized + Copy,
{
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;

    let chunk_size = mem::size_of::<T>();
    if chunk_size > MAX_CHUNK_SIZE {
        return Err(anyhow!("Chunk size > 1024"));
    }
    if bytes.len() % chunk_size != 0 {
        return Err(anyhow!("Chunk size miss match"));
    }
    let chunk_count = bytes.len() / chunk_size;

    let mut chunk = [0u8; MAX_CHUNK_SIZE];
    let mut vec: Vec<T> = Vec::with_capacity(chunk_count);
    for i in 0..chunk_count {
        for j in 0..chunk_size {
            chunk[j] = bytes[i * chunk_size + j];
        }
        vec.push(unsafe { *(chunk.as_ptr() as *const T) });
    }

    return Ok(vec);
}
