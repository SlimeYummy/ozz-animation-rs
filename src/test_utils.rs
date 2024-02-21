//!
//! Test utilities for integration tests in `./tests/` folder.
//!

use glam::Mat4;
use std::env::consts::{ARCH, OS};
use std::error::Error;
use std::fs::{self, File};
use std::io::prelude::*;
use std::{env, mem, slice};

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

pub fn compare_with_cpp(folder: &str, name: &str, data: &[Mat4], diff: f32) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(format!("./expected/{}", folder)).unwrap();
    fs::create_dir_all(format!("./output/{}", folder)).unwrap();

    let path = format!("./output/{0}/{1}_rust_{2}_{3}.bin", folder, name, OS, ARCH);
    let mut file = File::create(path)?;
    let data_size = data.len() * mem::size_of::<Mat4>();
    file.write_all(unsafe { slice::from_raw_parts(data.as_ptr() as *mut _, data_size) })?;

    let path = format!("./expected/{0}/{1}_cpp.bin", folder, name);
    let mut file = File::open(&path)?;
    if file.metadata()?.len() != data_size as u64 {
        return Err(format!("compare_with_cpp() size:{}", data_size).into());
    }

    let mut expected: Vec<Mat4> = vec![Mat4::default(); data.len()];
    file.read(unsafe { slice::from_raw_parts_mut(expected.as_mut_ptr() as *mut _, data_size) })?;
    for i in 0..expected.len() {
        if !Mat4::abs_diff_eq(&data[i], expected[i], diff) {
            println!("actual: {:?}", data[i]);
            println!("expected: {:?}", expected[i]);
            return Err(format!("compare_with_cpp() idx:{}", i).into());
        }
    }
    return Ok(());
}

#[cfg(feature = "rkyv")]
pub fn compare_with_rkyv<T>(folder: &str, name: &str, data: &T) -> Result<(), Box<dyn Error>>
where
    T: PartialEq + rkyv::Serialize<rkyv::ser::serializers::AllocSerializer<30720>>,
    T::Archived: rkyv::Deserialize<T, rkyv::Infallible>,
{
    use miniz_oxide::deflate::compress_to_vec;
    use miniz_oxide::inflate::decompress_to_vec;
    use rkyv::ser::Serializer;
    use rkyv::{AlignedVec, Deserialize};

    fs::create_dir_all(format!("./expected/{}", folder)).unwrap();
    fs::create_dir_all(format!("./output/{}", folder)).unwrap();

    let to_expected = env::var("SAVE_TO_EXPECTED").is_ok();

    let mut serializer = rkyv::ser::serializers::AllocSerializer::<30720>::default();
    serializer.serialize_value(data)?;
    let current_buf = serializer.into_serializer().into_inner();
    let wbuf = compress_to_vec(&current_buf, 6);
    let path = if to_expected {
        format!("./expected/{0}/{1}.rkyv", folder, name)
    } else {
        format!("./output/{0}/{1}_{2}_{3}.rkyv", folder, name, OS, ARCH)
    };
    let mut file = File::create(path)?;
    file.write_all(&wbuf)?;

    if !to_expected {
        let path = format!("./expected/{0}/{1}.rkyv", folder, name);
        let mut file = File::open(&path)?;
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut rbuf = Vec::with_capacity(size as usize);
        file.read_to_end(&mut rbuf)?;
        let unaligned_buf = decompress_to_vec(&rbuf).map_err(|e| e.to_string())?;
        let mut expected_buf = AlignedVec::new();
        expected_buf.extend_from_slice(&unaligned_buf);

        let archived = unsafe { rkyv::archived_root::<T>(&expected_buf) };
        let mut deserializer = rkyv::Infallible::default();
        let expected = archived.deserialize(&mut deserializer)?;
        if data != &expected {
            return Err(format!("compare_with_rkyv({})", path).into());
        }
    }
    return Ok(());
}

#[cfg(not(feature = "rkyv"))]
pub fn compare_with_rkyv<T>(_folder: &str, _name: &str, _data: &T) -> Result<(), Box<dyn Error>> {
    return Ok(());
}
