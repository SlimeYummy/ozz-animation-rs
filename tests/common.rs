#![allow(unused_imports)]
#![allow(dead_code)]

use glam::Mat4;
#[cfg(all(feature = "wasm", feature = "nodejs"))]
use ozz_animation_rs::nodejs;
use ozz_animation_rs::{Animation, Archive, Skeleton};
use std::env::consts::{ARCH, OS};
use std::error::Error;
use std::fs::{self, File};
use std::io::prelude::*;
use std::{env, mem, slice};

#[cfg(not(feature = "wasm"))]
pub fn compare_with_cpp(folder: &str, name: &str, data: &[Mat4], diff: f32) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(format!("./expected/{}", folder)).unwrap();
    fs::create_dir_all(format!("./output/{}", folder)).unwrap();

    let path = format!("./output/{0}/{1}_rust_{2}_{3}.bin", folder, name, OS, ARCH);
    let mut file = File::create(path)?;
    let data_size = std::mem::size_of_val(data);
    file.write_all(unsafe { slice::from_raw_parts(data.as_ptr() as *mut _, data_size) })?;

    let path = format!("./expected/{0}/{1}_cpp.bin", folder, name);
    let mut file = File::open(&path)?;
    if file.metadata()?.len() != data_size as u64 {
        return Err(format!("compare_with_cpp() size:{}", data_size).into());
    }

    let mut expected: Vec<Mat4> = vec![Mat4::default(); data.len()];
    file.read_exact(unsafe { slice::from_raw_parts_mut(expected.as_mut_ptr() as *mut _, data_size) })?;
    for i in 0..expected.len() {
        if !Mat4::abs_diff_eq(&data[i], expected[i], diff) {
            println!("actual: {:?}", data[i]);
            println!("expected: {:?}", expected[i]);
            return Err(format!("compare_with_cpp() idx:{}", i).into());
        }
    }
    Ok(())
}

#[cfg(all(feature = "wasm", feature = "nodejs"))]
pub fn compare_with_cpp(folder: &str, name: &str, data: &[Mat4], diff: f32) -> Result<(), Box<dyn Error>> {
    let path = format!("./expected/{0}/{1}_cpp.bin", folder, name);
    let buf = nodejs::read_file(&path).map_err(|e| String::from(e.to_string()))?;

    let data_size = data.len() * mem::size_of::<Mat4>();
    if buf.len() != data_size {
        return Err(format!("compare_with_cpp() size:{}", data_size).into());
    }

    let mut expected: Vec<Mat4> = vec![Mat4::default(); data.len()];
    let expected_buf = unsafe { slice::from_raw_parts_mut(expected.as_mut_ptr() as *mut _, data_size) };
    expected_buf.copy_from_slice(&buf);
    for i in 0..expected.len() {
        if !Mat4::abs_diff_eq(&data[i], expected[i], diff) {
            println!("actual: {:?}", data[i]);
            println!("expected: {:?}", expected[i]);
            return Err(format!("compare_with_cpp() idx:{}", i).into());
        }
    }
    Ok(())
}

#[cfg(feature = "rkyv")]
#[cfg(not(feature = "wasm"))]
pub fn compare_with_rkyv<T>(folder: &str, name: &str, data: &T) -> Result<(), Box<dyn Error>>
where
    T: PartialEq
        + for<'a> rkyv::Serialize<
            rkyv::api::high::HighSerializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::rancor::Error,
            >,
        >,
    T::Archived: rkyv::Deserialize<T, rkyv::api::high::HighDeserializer<rkyv::rancor::Error>>,
{
    use miniz_oxide::deflate::compress_to_vec;
    use miniz_oxide::inflate::decompress_to_vec;
    use rkyv::rancor::Error;
    use rkyv::ser::Serializer;
    use rkyv::util::AlignedVec;
    use rkyv::Deserialize;

    fs::create_dir_all(format!("./expected/{}", folder)).unwrap();
    fs::create_dir_all(format!("./output/{}", folder)).unwrap();

    let to_expected = env::var("SAVE_TO_EXPECTED").is_ok();

    let current_buf = rkyv::to_bytes(data)?;
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
        let mut expected_buf = AlignedVec::<32>::new();
        expected_buf.extend_from_slice(&unaligned_buf);

        let archived = unsafe { rkyv::access_unchecked::<T::Archived>(&expected_buf) };
        let expected = rkyv::deserialize::<_, Error>(archived)?;
        if data != &expected {
            return Err(format!("compare_with_rkyv({})", path).into());
        }
    }
    Ok(())
}

#[cfg(feature = "rkyv")]
#[cfg(all(feature = "wasm", feature = "nodejs"))]
pub fn compare_with_rkyv<T>(folder: &str, name: &str, data: &T) -> Result<(), Box<dyn Error>>
where
    T: PartialEq
        + for<'a> rkyv::Serialize<
            rkyv::api::high::HighSerializer<
                rkyv::util::AlignedVec,
                rkyv::ser::allocator::ArenaHandle<'a>,
                rkyv::rancor::Error,
            >,
        >,
    T::Archived: rkyv::Deserialize<T, rkyv::api::high::HighDeserializer<rkyv::rancor::Error>>,
{
    use miniz_oxide::deflate::compress_to_vec;
    use miniz_oxide::inflate::decompress_to_vec;
    use rkyv::rancor::Error;
    use rkyv::ser::Serializer;
    use rkyv::util::AlignedVec;
    use rkyv::Deserialize;

    let path = format!("./expected/{0}/{1}.rkyv", folder, name);
    let rbuf = nodejs::read_file(&path).map_err(|e| String::from(e.to_string()))?;
    let unaligned_buf = decompress_to_vec(&rbuf).map_err(|e| e.to_string())?;
    let mut expected_buf = AlignedVec::<32>::new();
    expected_buf.extend_from_slice(&unaligned_buf);

    let archived = unsafe { rkyv::access_unchecked::<T::Archived>(&expected_buf) };
    let expected = rkyv::deserialize::<_, Error>(archived)?;
    if data != &expected {
        return Err(format!("compare_with_rkyv({})", path).into());
    }
    Ok(())
}

#[cfg(not(feature = "rkyv"))]
pub fn compare_with_rkyv<T>(_folder: &str, _name: &str, _data: &T) -> Result<(), Box<dyn Error>> {
    Ok(())
}

#[cfg(feature = "rkyv")]
#[cfg(not(feature = "wasm"))]
pub fn save_rkyv<T>(folder: &str, name: &str, data: &T, to_expected: bool) -> Result<(), Box<dyn Error>>
where
    T: for<'a> rkyv::Serialize<
        rkyv::api::high::HighSerializer<
            rkyv::util::AlignedVec,
            rkyv::ser::allocator::ArenaHandle<'a>,
            rkyv::rancor::Error,
        >,
    >,
    T::Archived: rkyv::Deserialize<T, rkyv::api::high::HighDeserializer<rkyv::rancor::Error>>,
{
    use miniz_oxide::deflate::compress_to_vec;
    use rkyv::ser::Serializer;

    fs::create_dir_all(format!("./expected/{}", folder)).unwrap();
    fs::create_dir_all(format!("./output/{}", folder)).unwrap();

    let current_buf = rkyv::to_bytes(data)?;
    let wbuf = compress_to_vec(&current_buf, 6);
    let path = if to_expected {
        format!("./expected/{0}/{1}.rkyv", folder, name)
    } else {
        format!("./output/{0}/{1}_{2}_{3}.rkyv", folder, name, OS, ARCH)
    };
    let mut file = File::create(path)?;
    file.write_all(&wbuf)?;
    Ok(())
}

#[cfg(any(not(feature = "rkyv"), feature = "wasm"))]
pub fn save_rkyv<T>(_folder: &str, _name: &str, _data: &T, _to_expected: bool) -> Result<(), Box<dyn Error>> {
    Ok(())
}

#[cfg(feature = "rkyv")]
#[cfg(not(feature = "wasm"))]
pub fn load_rkyv<T>(folder: &str, name: &str) -> Result<T, Box<dyn Error>>
where
    T: for<'a> rkyv::Serialize<
        rkyv::api::high::HighSerializer<
            rkyv::util::AlignedVec,
            rkyv::ser::allocator::ArenaHandle<'a>,
            rkyv::rancor::Error,
        >,
    >,
    T::Archived: rkyv::Deserialize<T, rkyv::api::high::HighDeserializer<rkyv::rancor::Error>>,
{
    use miniz_oxide::inflate::decompress_to_vec;
    use rkyv::rancor::Error;
    use rkyv::util::AlignedVec;
    use rkyv::Deserialize;

    let path = format!("./expected/{0}/{1}.rkyv", folder, name);
    let mut file = File::open(&path)?;
    let size = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut rbuf = Vec::with_capacity(size as usize);
    file.read_to_end(&mut rbuf)?;
    let unaligned_buf = decompress_to_vec(&rbuf).map_err(|e| e.to_string())?;
    let mut expected_buf = AlignedVec::<32>::new();
    expected_buf.extend_from_slice(&unaligned_buf);

    let archived = unsafe { rkyv::access_unchecked::<T::Archived>(&expected_buf) };
    let data = rkyv::deserialize::<_, Error>(archived)?;
    Ok(data)
}

#[cfg(feature = "rkyv")]
#[cfg(all(feature = "wasm", feature = "nodejs"))]
pub fn load_rkyv<T>(folder: &str, name: &str) -> Result<T, Box<dyn Error>>
where
    T: for<'a> rkyv::Serialize<
        rkyv::api::high::HighSerializer<
            rkyv::util::AlignedVec,
            rkyv::ser::allocator::ArenaHandle<'a>,
            rkyv::rancor::Error,
        >,
    >,
    T::Archived: rkyv::Deserialize<T, rkyv::api::high::HighDeserializer<rkyv::rancor::Error>>,
{
    use miniz_oxide::inflate::decompress_to_vec;
    use rkyv::rancor::Error;
    use rkyv::util::AlignedVec;
    use rkyv::Deserialize;

    let path = format!("./expected/{0}/{1}.rkyv", folder, name);
    let rbuf = nodejs::read_file(&path).map_err(|e| String::from(e.to_string()))?;
    let unaligned_buf = decompress_to_vec(&rbuf).map_err(|e| e.to_string())?;
    let mut expected_buf = AlignedVec::<32>::new();
    expected_buf.extend_from_slice(&unaligned_buf);

    let archived = unsafe { rkyv::access_unchecked::<T::Archived>(&expected_buf) };
    let data = rkyv::deserialize::<_, Error>(archived)?;
    Ok(data)
}

#[cfg(not(feature = "rkyv"))]
pub fn load_rkyv<T>(_folder: &str, _name: &str) -> Result<T, Box<dyn Error>> {
    unimplemented!()
}
