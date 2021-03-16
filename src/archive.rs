use crate::endian::{Endian, SwapEndian};
use anyhow::Result;
use std::fs::File;
use std::io::Read;
use std::mem;
use std::path::Path;
use std::str;

pub trait ArchiveVersion {
    fn version() -> u32;
}

pub trait ArchiveTag {
    fn tag() -> &'static str;
}

pub trait ArchiveReader<T> {
    fn read(archive: &mut IArchive) -> Result<T>;
}

pub struct IArchive {
    file: File,
    endian_swap: bool,
}

impl IArchive {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<IArchive> {
        let mut file = File::open(path)?;

        let mut endian_tag = [0u8; 1];
        file.read_exact(&mut endian_tag)?;
        let file_endian = Endian::from_tag(endian_tag[0]);

        let native_endian = Endian::native();

        return Ok(IArchive {
            file,
            endian_swap: file_endian != native_endian,
        });
    }

    pub fn test_tag<T: ArchiveTag>(&mut self) -> Result<bool> {
        let file_tag = self.read_string(0)?;
        let type_tag = T::tag();
        return Ok(type_tag == file_tag);
    }

    pub fn read_version(&mut self) -> Result<u32> {
        return self.read::<u32>();
    }

    pub fn read<T: ArchiveReader<T>>(&mut self) -> Result<T> {
        return T::read(self);
    }

    pub fn read_vec<T: ArchiveReader<T>>(&mut self, count: usize) -> Result<Vec<T>> {
        let mut buffer = Vec::with_capacity(count);
        for _ in 0..count {
            buffer.push(self.read()?);
        }
        return Ok(buffer);
    }

    pub fn read_string(&mut self, count: usize) -> Result<String> {
        if count != 0 {
            let buffer = self.read_vec::<u8>(count)?;
            let text = String::from_utf8(buffer)?;
            return Ok(text);
        } else {
            let mut buffer = Vec::new();
            loop {
                let char = self.read::<u8>()?;
                if char != 0 {
                    buffer.push(char);
                } else {
                    break;
                }
            }
            let text = String::from_utf8(buffer)?;
            return Ok(text);
        }
    }
}

macro_rules! primitive_reader {
    ($type:ty) => {
        impl ArchiveReader<$type> for $type {
            fn read(archive: &mut IArchive) -> Result<$type> {
                let mut bytes = [0u8; mem::size_of::<$type>()];
                archive.file.read_exact(&mut bytes)?;
                let raw = unsafe { mem::transmute(bytes) };
                if !archive.endian_swap {
                    return Ok(raw);
                } else {
                    return Ok(raw.swap());
                }
            }
        }
    };
}

primitive_reader!(u8);
primitive_reader!(i8);
primitive_reader!(bool);
primitive_reader!(u16);
primitive_reader!(i16);
primitive_reader!(u32);
primitive_reader!(i32);
primitive_reader!(f32);
primitive_reader!(u64);
primitive_reader!(i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archive_new() {
        let archive = IArchive::new("./test_files/animation-simple.ozz").unwrap();
        assert_eq!(archive.endian_swap, false);
    }
}
