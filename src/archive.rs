//!
//! Animation data structure definition.
//!

use glam::{Quat, Vec2, Vec3, Vec4};
#[cfg(not(feature = "wasm"))]
use std::fs::File;
use std::io::{Cursor, Read};
#[cfg(not(feature = "wasm"))]
use std::path::Path;
use std::{mem, slice, str};

use crate::base::OzzError;
use crate::endian::{Endian, SwapEndian};

/// Implements input archive concept used to load/de-serialize data.
/// Endianness conversions are automatically performed according to the Archive
/// and the native formats.
pub struct Archive<R: Read> {
    read: R,
    endian_swap: bool,
    tag: String,
    version: u32,
}

impl<R: Read> Archive<R> {
    /// Creates an `Archive` from a file.
    pub fn new(mut read: R) -> Result<Archive<R>, OzzError> {
        let mut endian_tag = [0u8; 1];
        read.read_exact(&mut endian_tag)?;
        let file_endian = Endian::from_tag(endian_tag[0]);
        let native_endian = Endian::native();

        let mut archive = Archive {
            read,
            endian_swap: file_endian != native_endian,
            tag: String::new(),
            version: 0,
        };

        let tag = archive.read::<String>()?;
        archive.tag = tag;

        let version = archive.read::<u32>()?;
        archive.version = version;

        return Ok(archive);
    }

    /// Reads `T` from the archive.
    pub fn read<T: ArchiveRead<T>>(&mut self) -> Result<T, OzzError> {
        return T::read(self);
    }

    /// Reads `Vec<T>` from the archive.
    /// * `count` - The number of elements to read.
    pub fn read_vec<T: ArchiveRead<T>>(&mut self, count: usize) -> Result<Vec<T>, OzzError> {
        return T::read_vec(self, count);
    }

    /// Does the endian need to be swapped.
    pub fn endian_swap(&self) -> bool {
        return self.endian_swap;
    }

    /// Gets the tag of the archive.
    pub fn tag(&self) -> &str {
        return &self.tag;
    }

    /// Gets the version of the archive.
    pub fn version(&self) -> u32 {
        return self.version;
    }
}

#[cfg(not(feature = "wasm"))]
impl Archive<File> {
    /// Creates an `Archive` from a path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Archive<File>, OzzError> {
        let file = File::open(path)?;
        return Archive::new(file);
    }

    /// Creates an `Archive` from a file.
    pub fn from_file(file: File) -> Result<Archive<File>, OzzError> {
        return Archive::new(file);
    }
}

impl Archive<Cursor<Vec<u8>>> {
    /// Creates an `Archive` from a `Vec<u8>`.
    pub fn from_vec(buf: Vec<u8>) -> Result<Archive<Cursor<Vec<u8>>>, OzzError> {
        let cursor = Cursor::new(buf);
        return Archive::new(cursor);
    }

    /// Creates an `Archive` from a path.
    #[cfg(all(feature = "wasm", feature = "nodejs"))]
    pub fn from_path(path: &str) -> Result<Archive<Cursor<Vec<u8>>>, OzzError> {
        match crate::nodejs::read_file(path) {
            Ok(buf) => return Archive::from_vec(buf),
            Err(err) => return Err(OzzError::Custom(Box::new(err.as_string().unwrap_or("".into())))),
        };
    }
}

impl Archive<Cursor<&[u8]>> {
    /// Creates an `Archive` from a `&[u8]`.
    pub fn from_slice(buf: &[u8]) -> Result<Archive<Cursor<&[u8]>>, OzzError> {
        let cursor = Cursor::new(buf);
        return Archive::new(cursor);
    }
}

/// Implements `ArchiveRead` to read `T` from Archive.
pub trait ArchiveRead<T> {
    /// Reads `T` from the archive.
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<T, OzzError>;

    /// Reads `Vec<T>` from the archive.
    /// * `count` - The number of elements to read.
    fn read_vec<R: Read>(archive: &mut Archive<R>, count: usize) -> Result<Vec<T>, OzzError> {
        let mut buffer = Vec::with_capacity(count);
        for _ in 0..count {
            buffer.push(Self::read(archive)?);
        }
        return Ok(buffer);
    }
}

macro_rules! primitive_reader {
    ($type:ty) => {
        impl ArchiveRead<$type> for $type {
            fn read<R: Read>(archive: &mut Archive<R>) -> Result<$type, OzzError> {
                let val = Default::default();
                archive.read.read_exact(unsafe {
                    slice::from_raw_parts_mut(&val as *const $type as *mut u8, mem::size_of::<$type>())
                })?;
                if !archive.endian_swap {
                    return Ok(val);
                } else {
                    return Ok(val.swap_endian());
                }
            }
        }
    };
}

primitive_reader!(bool);
primitive_reader!(u8);
primitive_reader!(i8);
primitive_reader!(u16);
primitive_reader!(i16);
primitive_reader!(u32);
primitive_reader!(i32);
primitive_reader!(u64);
primitive_reader!(i64);
primitive_reader!(f32);
primitive_reader!(f64);

impl ArchiveRead<Vec2> for Vec2 {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<Vec2, OzzError> {
        let x = f32::read(archive)?;
        let y = f32::read(archive)?;
        return Ok(Vec2::new(x, y));
    }
}

impl ArchiveRead<Vec3> for Vec3 {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<Vec3, OzzError> {
        let x = f32::read(archive)?;
        let y = f32::read(archive)?;
        let z = f32::read(archive)?;
        return Ok(Vec3::new(x, y, z));
    }
}

impl ArchiveRead<Vec4> for Vec4 {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<Vec4, OzzError> {
        let x = f32::read(archive)?;
        let y = f32::read(archive)?;
        let z = f32::read(archive)?;
        let w = f32::read(archive)?;
        return Ok(Vec4::new(x, y, z, w));
    }
}

impl ArchiveRead<Quat> for Quat {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<Quat, OzzError> {
        let x = f32::read(archive)?;
        let y = f32::read(archive)?;
        let z = f32::read(archive)?;
        let w = f32::read(archive)?;
        return Ok(Quat::from_xyzw(x, y, z, w));
    }
}

impl ArchiveRead<String> for String {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<String, OzzError> {
        let mut buffer = Vec::new();
        loop {
            let char = u8::read(archive)?;
            if char != 0 {
                buffer.push(char);
            } else {
                break;
            }
        }
        let text = String::from_utf8(buffer).map_err(|e| e.utf8_error())?;
        return Ok(text);
    }
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_archive_new() {
        let archive = Archive::from_path("./resource/playback/animation.ozz").unwrap();
        assert_eq!(archive.endian_swap, false);
    }
}
