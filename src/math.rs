//!
//! Math library for SIMD operations.
//!
//! Soa is short for Structure of Arrays. Represents SSE packaged data.
//!

#![allow(dead_code)]

use glam::{Mat4, Quat, Vec3, Vec3A, Vec4};
use static_assertions::const_assert_eq;
use std::fmt::Debug;
use std::io::Read;
use std::mem;
use std::simd::prelude::*;
use std::simd::*;

use crate::archive::{Archive, ArchiveRead};
use crate::base::OzzError;

pub(crate) const ZERO: f32x4 = f32x4::from_array([0.0; 4]);
pub(crate) const ONE: f32x4 = f32x4::from_array([1.0; 4]);
pub(crate) const TWO: f32x4 = f32x4::from_array([2.0; 4]);
pub(crate) const THREE: f32x4 = f32x4::from_array([3.0; 4]);
pub(crate) const NEG_ONE: f32x4 = f32x4::from_array([-1.0; 4]);
pub(crate) const FRAC_1_2: f32x4 = f32x4::from_array([0.5; 4]);
pub(crate) const PI: f32x4 = f32x4::from_array([core::f32::consts::PI; 4]);
pub(crate) const FRAC_2_PI: f32x4 = f32x4::from_array([core::f32::consts::FRAC_2_PI; 4]);
pub(crate) const FRAC_PI_2: f32x4 = f32x4::from_array([core::f32::consts::FRAC_PI_2; 4]);

pub(crate) const X_AXIS: f32x4 = f32x4::from_array([1.0, 0.0, 0.0, 0.0]);
pub(crate) const Y_AXIS: f32x4 = f32x4::from_array([0.0, 1.0, 0.0, 0.0]);
pub(crate) const Z_AXIS: f32x4 = f32x4::from_array([0.0, 0.0, 1.0, 0.0]);

pub(crate) const QUAT_UNIT: f32x4 = f32x4::from_array([0.0, 0.0, 0.0, 1.0]);

const SIGN: i32x4 = i32x4::from_array([core::i32::MIN; 4]);
const SIGN_W: i32x4 = i32x4::from_array([0, 0, 0, core::i32::MIN]);

//
// SoaVec3
//

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SoaVec3 {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
}

impl SoaVec3 {
    #[inline]
    pub const fn new(x: [f32; 4], y: [f32; 4], z: [f32; 4]) -> SoaVec3 {
        return SoaVec3 {
            x: f32x4::from_array(x),
            y: f32x4::from_array(y),
            z: f32x4::from_array(z),
        };
    }

    #[inline]
    pub const fn splat_row(v: f32x4) -> SoaVec3 {
        return SoaVec3 { x: v, y: v, z: v };
    }

    #[inline]
    pub const fn splat_col(v: [f32; 3]) -> SoaVec3 {
        return SoaVec3 {
            x: f32x4::from_array([v[0]; 4]),
            y: f32x4::from_array([v[1]; 4]),
            z: f32x4::from_array([v[2]; 4]),
        };
    }

    #[inline]
    pub fn col(&self, idx: usize) -> Vec3 {
        return Vec3::new(self.x[idx], self.y[idx], self.z[idx]);
    }

    #[inline]
    pub fn set_col(&mut self, idx: usize, v: Vec3) {
        self.x[idx] = v.x;
        self.y[idx] = v.y;
        self.z[idx] = v.z;
    }

    #[inline]
    pub fn neg(&self) -> SoaVec3 {
        return SoaVec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        };
    }

    #[inline]
    pub fn add(&self, other: &SoaVec3) -> SoaVec3 {
        return SoaVec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        };
    }

    #[inline]
    pub fn sub(&self, other: &SoaVec3) -> SoaVec3 {
        return SoaVec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        };
    }

    #[inline]
    pub fn component_mul(&self, other: &SoaVec3) -> SoaVec3 {
        return SoaVec3 {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        };
    }

    #[inline]
    pub fn mul_num(&self, f: f32x4) -> SoaVec3 {
        return SoaVec3 {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f,
        };
    }

    #[inline]
    pub fn lerp(from: &SoaVec3, to: &SoaVec3, alpha: f32x4) -> SoaVec3 {
        return SoaVec3 {
            x: (to.x - from.x) * alpha + from.x,
            y: (to.y - from.y) * alpha + from.y,
            z: (to.z - from.z) * alpha + from.z,
        };
    }

    #[inline]
    pub fn and_num(&self, i: i32x4) -> SoaVec3 {
        return SoaVec3 {
            x: fx4_and(self.x, i),
            y: fx4_and(self.y, i),
            z: fx4_and(self.z, i),
        };
    }

    #[inline]
    pub fn or_num(&self, i: i32x4) -> SoaVec3 {
        return SoaVec3 {
            x: fx4_or(self.x, i),
            y: fx4_or(self.y, i),
            z: fx4_or(self.z, i),
        };
    }

    #[inline]
    pub fn xor_num(&self, i: i32x4) -> SoaVec3 {
        return SoaVec3 {
            x: fx4_xor(self.x, i),
            y: fx4_xor(self.y, i),
            z: fx4_xor(self.z, i),
        };
    }
}

#[cfg(feature = "rkyv")]
const _: () = {
    use bytecheck::CheckBytes;
    use rkyv::{from_archived, to_archived, Archive, Deserialize, Fallible, Serialize};
    use std::io::{Error, ErrorKind};

    impl Archive for SoaVec3 {
        type Archived = SoaVec3;
        type Resolver = ();

        #[inline]
        unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
            out.write(to_archived!(*self as Self));
        }
    }

    impl<S: Fallible + ?Sized> Serialize<S> for SoaVec3 {
        #[inline]
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            return Ok(());
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<SoaVec3, D> for SoaVec3 {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<SoaVec3, D::Error> {
            return Ok(from_archived!(*self));
        }
    }

    impl<C: ?Sized> CheckBytes<C> for SoaVec3 {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<SoaVec3>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            return Ok(&*value);
        }
    }
};

#[cfg(feature = "serde")]
const _: () = {
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Serialize, Serializer};

    impl Serialize for SoaVec3 {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut seq = serializer.serialize_seq(Some(3))?;
            seq.serialize_element(&self.x.as_array())?;
            seq.serialize_element(&self.y.as_array())?;
            seq.serialize_element(&self.z.as_array())?;
            return seq.end();
        }
    }

    impl<'de> Deserialize<'de> for SoaVec3 {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let tmp: [[f32; 4]; 3] = Deserialize::deserialize(deserializer)?;
            return Ok(SoaVec3::new(tmp[0], tmp[1], tmp[2]));
        }
    }
};

//
// SoaQuat
//

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SoaQuat {
    pub x: f32x4,
    pub y: f32x4,
    pub z: f32x4,
    pub w: f32x4,
}

impl SoaQuat {
    #[inline]
    pub const fn new(x: [f32; 4], y: [f32; 4], z: [f32; 4], w: [f32; 4]) -> SoaQuat {
        return SoaQuat {
            x: f32x4::from_array(x),
            y: f32x4::from_array(y),
            z: f32x4::from_array(z),
            w: f32x4::from_array(w),
        };
    }

    #[inline]
    pub const fn splat_row(v: f32x4) -> SoaQuat {
        return SoaQuat { x: v, y: v, z: v, w: v };
    }

    #[inline]
    pub const fn splat_col(v: [f32; 4]) -> SoaQuat {
        return SoaQuat {
            x: f32x4::from_array([v[0]; 4]),
            y: f32x4::from_array([v[1]; 4]),
            z: f32x4::from_array([v[2]; 4]),
            w: f32x4::from_array([v[3]; 4]),
        };
    }

    #[inline]
    pub fn col(&self, idx: usize) -> Quat {
        return Quat::from_xyzw(self.x[idx], self.y[idx], self.z[idx], self.w[idx]);
    }

    #[inline]
    pub fn set_col(&mut self, idx: usize, v: Quat) {
        self.x[idx] = v.x;
        self.y[idx] = v.y;
        self.z[idx] = v.z;
        self.w[idx] = v.w;
    }

    #[inline]
    pub fn conjugate(&self) -> SoaQuat {
        return SoaQuat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        };
    }

    #[inline]
    pub fn add(&self, other: &SoaQuat) -> SoaQuat {
        return SoaQuat {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        };
    }

    #[inline]
    pub fn mul(&self, other: &SoaQuat) -> SoaQuat {
        let x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y;
        let y = self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z;
        let z = self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x;
        let w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z;
        return SoaQuat { x, y, z, w };
    }

    #[inline]
    pub fn mul_num(&self, f: f32x4) -> SoaQuat {
        return SoaQuat {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f,
            w: self.w * f,
        };
    }

    #[inline]
    pub fn normalize(&self) -> SoaQuat {
        let len2 = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        let inv_len = len2.sqrt().recip();
        return SoaQuat {
            x: self.x * inv_len,
            y: self.y * inv_len,
            z: self.z * inv_len,
            w: self.w * inv_len,
        };
    }

    #[inline]
    pub fn dot(&self, other: &SoaQuat) -> f32x4 {
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
    }

    #[inline]
    pub fn nlerp(&self, other: &SoaQuat, f: f32x4) -> SoaQuat {
        let lerp_x = (other.x - self.x) * f + self.x;
        let lerp_y = (other.y - self.y) * f + self.y;
        let lerp_z = (other.z - self.z) * f + self.z;
        let lerp_w = (other.w - self.w) * f + self.w;
        let len2 = lerp_x * lerp_x + lerp_y * lerp_y + lerp_z * lerp_z + lerp_w * lerp_w;
        let inv_len = len2.sqrt().recip();
        return SoaQuat {
            x: lerp_x * inv_len,
            y: lerp_y * inv_len,
            z: lerp_z * inv_len,
            w: lerp_w * inv_len,
        };
    }

    #[inline]
    pub fn and_num(&self, i: i32x4) -> SoaQuat {
        return SoaQuat {
            x: fx4_and(self.x, i),
            y: fx4_and(self.y, i),
            z: fx4_and(self.z, i),
            w: fx4_and(self.w, i),
        };
    }

    #[inline]
    pub fn or_num(&self, i: i32x4) -> SoaQuat {
        return SoaQuat {
            x: fx4_or(self.x, i),
            y: fx4_or(self.y, i),
            z: fx4_or(self.z, i),
            w: fx4_or(self.w, i),
        };
    }

    #[inline]
    pub fn xor_num(&self, i: i32x4) -> SoaQuat {
        return SoaQuat {
            x: fx4_xor(self.x, i),
            y: fx4_xor(self.y, i),
            z: fx4_xor(self.z, i),
            w: fx4_xor(self.w, i),
        };
    }

    #[inline]
    pub fn positive_w(&self) -> SoaQuat {
        let sign = fx4_sign(self.w);
        return SoaQuat {
            x: fx4_xor(self.x, sign),
            y: fx4_xor(self.y, sign),
            z: fx4_xor(self.z, sign),
            w: fx4_xor(self.w, sign),
        };
    }
}

#[cfg(feature = "rkyv")]
const _: () = {
    use bytecheck::CheckBytes;
    use rkyv::{from_archived, to_archived, Archive, Deserialize, Fallible, Serialize};
    use std::io::{Error, ErrorKind};

    impl Archive for SoaQuat {
        type Archived = SoaQuat;
        type Resolver = ();

        #[inline]
        unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
            out.write(to_archived!(*self as Self));
        }
    }

    impl<S: Fallible + ?Sized> Serialize<S> for SoaQuat {
        #[inline]
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            return Ok(());
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<SoaQuat, D> for SoaQuat {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<SoaQuat, D::Error> {
            return Ok(from_archived!(*self));
        }
    }

    impl<C: ?Sized> CheckBytes<C> for SoaQuat {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<SoaQuat>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            return Ok(&*value);
        }
    }
};

#[cfg(feature = "serde")]
const _: () = {
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Serialize, Serializer};

    impl Serialize for SoaQuat {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut seq = serializer.serialize_seq(Some(3))?;
            seq.serialize_element(&self.x.as_array())?;
            seq.serialize_element(&self.y.as_array())?;
            seq.serialize_element(&self.z.as_array())?;
            seq.serialize_element(&self.w.as_array())?;
            return seq.end();
        }
    }

    impl<'de> Deserialize<'de> for SoaQuat {
        fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let tmp: [[f32; 4]; 4] = Deserialize::deserialize(deserializer)?;
            return Ok(SoaQuat::new(tmp[0], tmp[1], tmp[2], tmp[3]));
        }
    }
};

//
// SoaTransform
//

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SoaTransform {
    pub translation: SoaVec3,
    pub rotation: SoaQuat,
    pub scale: SoaVec3,
}

impl ArchiveRead<SoaTransform> for SoaTransform {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<SoaTransform, OzzError> {
        const COUNT: usize = mem::size_of::<SoaTransform>() / mem::size_of::<f32>();
        let mut buffer = [0f32; COUNT];
        for idx in 0..COUNT {
            buffer[idx] = archive.read()?;
        }
        return Ok(unsafe { mem::transmute(buffer) });
    }
}

impl SoaTransform {
    pub fn new(translation: SoaVec3, rotation: SoaQuat, scale: SoaVec3) -> SoaTransform {
        return SoaTransform {
            translation,
            rotation,
            scale,
        };
    }
}

#[cfg(feature = "rkyv")]
const _: () = {
    use bytecheck::CheckBytes;
    use rkyv::{from_archived, to_archived, Archive, Deserialize, Fallible, Serialize};
    use std::io::{Error, ErrorKind};

    impl Archive for SoaTransform {
        type Archived = SoaTransform;
        type Resolver = ();

        #[inline]
        unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
            out.write(to_archived!(*self as Self));
        }
    }

    impl<S: Fallible + ?Sized> Serialize<S> for SoaTransform {
        #[inline]
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            return Ok(());
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<SoaTransform, D> for SoaTransform {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<SoaTransform, D::Error> {
            return Ok(from_archived!(*self));
        }
    }

    impl<C: ?Sized> CheckBytes<C> for SoaTransform {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<SoaTransform>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            return Ok(&*value);
        }
    }
};

//
// AosMat4
//

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub(crate) struct AosMat4 {
    pub cols: [f32x4; 4],
}

const_assert_eq!(mem::size_of::<AosMat4>(), mem::size_of::<Mat4>());

impl From<Mat4> for AosMat4 {
    fn from(mat: Mat4) -> AosMat4 {
        return unsafe { mem::transmute(mat) };
    }
}

impl Into<Mat4> for AosMat4 {
    fn into(self) -> Mat4 {
        return unsafe { mem::transmute(self) };
    }
}

impl AosMat4 {
    #[rustfmt::skip]
    #[inline]
    pub(crate) fn new(
        n00: f32, n01: f32, n02: f32, n03: f32,
        n10: f32, n11: f32, n12: f32, n13: f32,
        n20: f32, n21: f32, n22: f32, n23: f32,
        n30: f32, n31: f32, n32: f32, n33: f32,
    ) -> AosMat4 {
        return AosMat4 {
            cols: [
                f32x4::from_array([n00, n01, n02, n03]),
                f32x4::from_array([n10, n11, n12, n13]),
                f32x4::from_array([n20, n21, n22, n23]),
                f32x4::from_array([n30, n31, n32, n33]),
            ],
        };
    }

    #[inline]
    pub(crate) fn new_translation(t: Vec3) -> AosMat4 {
        return AosMat4 {
            cols: [
                f32x4::from_array([1.0, 0.0, 0.0, 0.0]),
                f32x4::from_array([0.0, 1.0, 0.0, 0.0]),
                f32x4::from_array([0.0, 0.0, 1.0, 0.0]),
                f32x4::from_array([t.x, t.y, t.z, 1.0]),
            ],
        };
    }

    #[inline]
    pub(crate) fn new_scaling(s: Vec3) -> AosMat4 {
        return AosMat4 {
            cols: [
                f32x4::from_array([s.x, 0.0, 0.0, 0.0]),
                f32x4::from_array([0.0, s.y, 0.0, 0.0]),
                f32x4::from_array([0.0, 0.0, s.z, 0.0]),
                f32x4::from_array([0.0, 0.0, 0.0, 1.0]),
            ],
        };
    }

    #[inline]
    pub(crate) fn identity() -> AosMat4 {
        return AosMat4 {
            cols: [
                f32x4::from_array([1.0, 0.0, 0.0, 0.0]),
                f32x4::from_array([0.0, 1.0, 0.0, 0.0]),
                f32x4::from_array([0.0, 0.0, 1.0, 0.0]),
                f32x4::from_array([0.0, 0.0, 0.0, 1.0]),
            ],
        };
    }

    pub(crate) fn mul(&self, other: &AosMat4) -> AosMat4 {
        const X: [usize; 4] = [0, 0, 0, 0];
        const Y: [usize; 4] = [1, 1, 1, 1];
        const Z: [usize; 4] = [2, 2, 2, 2];
        const W: [usize; 4] = [3, 3, 3, 3];

        let mut result = AosMat4::default();

        let xxxx = simd_swizzle!(other.cols[0], X) * self.cols[0];
        let zzzz = simd_swizzle!(other.cols[0], Z) * self.cols[2];
        let a01 = simd_swizzle!(other.cols[0], Y) * self.cols[1] + xxxx;
        let a23 = simd_swizzle!(other.cols[0], W) * self.cols[3] + zzzz;
        result.cols[0] = a01 + a23;

        let xxxx = simd_swizzle!(other.cols[1], X) * self.cols[0];
        let zzzz = simd_swizzle!(other.cols[1], Z) * self.cols[2];
        let a01 = simd_swizzle!(other.cols[1], Y) * self.cols[1] + xxxx;
        let a23 = simd_swizzle!(other.cols[1], W) * self.cols[3] + zzzz;
        result.cols[1] = a01 + a23;

        let xxxx = simd_swizzle!(other.cols[2], X) * self.cols[0];
        let zzzz = simd_swizzle!(other.cols[2], Z) * self.cols[2];
        let a01 = simd_swizzle!(other.cols[2], Y) * self.cols[1] + xxxx;
        let a23 = simd_swizzle!(other.cols[2], W) * self.cols[3] + zzzz;
        result.cols[2] = a01 + a23;

        let xxxx = simd_swizzle!(other.cols[3], X) * self.cols[0];
        let zzzz = simd_swizzle!(other.cols[3], Z) * self.cols[2];
        let a01 = simd_swizzle!(other.cols[3], Y) * self.cols[1] + xxxx;
        let a23 = simd_swizzle!(other.cols[3], W) * self.cols[3] + zzzz;
        result.cols[3] = a01 + a23;

        return result;
    }

    pub(crate) fn invert(&self) -> AosMat4 {
        const IB1: [usize; 4] = [1, 0, 3, 2]; // 0xB1
        const I4E: [usize; 4] = [2, 3, 0, 1]; // 0x4E

        let t0 = simd_swizzle!(self.cols[0], self.cols[1], [0, 1, 4, 5]);
        let t1 = simd_swizzle!(self.cols[2], self.cols[3], [0, 1, 4, 5]);
        let t2 = simd_swizzle!(self.cols[0], self.cols[1], [2, 3, 6, 7]);
        let t3 = simd_swizzle!(self.cols[2], self.cols[3], [2, 3, 6, 7]);
        let c0 = simd_swizzle!(t0, t1, [0, 2, 4, 6]);
        let c1 = simd_swizzle!(t1, t0, [1, 3, 5, 7]);
        let c2 = simd_swizzle!(t2, t3, [0, 2, 4, 6]);
        let c3 = simd_swizzle!(t3, t2, [1, 3, 5, 7]);

        let mut minor0;
        let mut minor1;
        let mut minor2;
        let mut minor3;
        let mut tmp1;
        let tmp2;

        tmp1 = simd_swizzle!(c2 * c3, IB1);
        minor0 = c1 * tmp1;
        minor1 = c0 * tmp1;
        tmp1 = simd_swizzle!(tmp1, I4E);
        minor0 = c1 * tmp1 - minor0;
        minor1 = c0 * tmp1 - minor1;
        minor1 = simd_swizzle!(minor1, I4E);

        tmp1 = simd_swizzle!(c1 * c2, IB1);
        minor0 = c3 * tmp1 + minor0;
        minor3 = c0 * tmp1;
        tmp1 = simd_swizzle!(tmp1, I4E);
        minor0 = minor0 - c3 * tmp1;
        minor3 = c0 * tmp1 - minor3;
        minor3 = simd_swizzle!(minor3, I4E);

        tmp1 = simd_swizzle!(c1, I4E) * c3;
        tmp1 = simd_swizzle!(tmp1, IB1);
        tmp2 = simd_swizzle!(c2, I4E);
        minor0 = tmp2 * tmp1 + minor0;
        minor2 = c0 * tmp1;
        tmp1 = simd_swizzle!(tmp1, I4E);
        minor0 = minor0 - tmp2 * tmp1;
        minor2 = c0 * tmp1 - minor2;
        minor2 = simd_swizzle!(minor2, I4E);

        tmp1 = simd_swizzle!(c0 * c1, IB1);
        minor2 = c3 * tmp1 + minor2;
        minor3 = tmp2 * tmp1 - minor3;
        tmp1 = simd_swizzle!(tmp1, I4E);
        minor2 = c3 * tmp1 - minor2;
        minor3 = minor3 - tmp2 * tmp1;

        tmp1 = simd_swizzle!(c0 * c3, IB1);
        minor1 = minor1 - tmp2 * tmp1;
        minor2 = c1 * tmp1 + minor2;
        tmp1 = simd_swizzle!(tmp1, I4E);
        minor1 = tmp2 * tmp1 + minor1;
        minor2 = minor2 - c1 * tmp1;

        tmp1 = simd_swizzle!(c0 * tmp2, IB1);
        minor1 = c3 * tmp1 + minor1;
        minor3 = minor3 - c1 * tmp1;
        tmp1 = simd_swizzle!(tmp1, I4E);
        minor1 = minor1 - c3 * tmp1;
        minor3 = c1 * tmp1 + minor3;

        let mut det = c0 * minor0;
        det = simd_swizzle!(det, I4E) + det;
        det = simd_swizzle!(det, IB1) + det; // first
        let invertible: i32x4 = det.simd_ne(ZERO).to_int(); // first

        let det_recip = det.recip(); // first
                                     // det_recip = (det_recip + det_recip) - det_recip * det_recip * det; // first
        tmp1 = fx4((ix4(det_recip) & invertible) | (!invertible & ix4(ZERO))); // first
        det = (tmp1 + tmp1) - det * (tmp1 * tmp1); // first
        det = fx4_splat_x(det);
        return AosMat4 {
            cols: [det * minor0, det * minor1, det * minor2, det * minor3],
        };
    }

    #[inline]
    pub(crate) fn transform_point(&self, p: f32x4) -> f32x4 {
        let xxxx = simd_swizzle!(p, [0; 4]) * self.cols[0];
        let a23 = simd_swizzle!(p, [2; 4]) * self.cols[2] + self.cols[3];
        let a01 = simd_swizzle!(p, [1; 4]) * self.cols[1] + xxxx;
        return a01 + a23;
    }

    #[inline]
    pub(crate) fn transform_vector(&self, p: f32x4) -> f32x4 {
        let xxxx = simd_swizzle!(p, [0; 4]) * self.cols[0];
        let zzzz = simd_swizzle!(p, [1; 4]) * self.cols[1];
        let a21 = simd_swizzle!(p, [2; 4]) * self.cols[2] + xxxx;
        return zzzz + a21;
    }
}

//
// SoaMat4
//
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct SoaMat4 {
    pub cols: [f32x4; 16],
}

impl SoaMat4 {
    pub fn from_affine(translation: &SoaVec3, rotation: &SoaQuat, scale: &SoaVec3) -> SoaMat4 {
        let xx = rotation.x * rotation.x;
        let xy = rotation.x * rotation.y;
        let xz = rotation.x * rotation.z;
        let xw = rotation.x * rotation.w;
        let yy = rotation.y * rotation.y;
        let yz = rotation.y * rotation.z;
        let yw = rotation.y * rotation.w;
        let zz = rotation.z * rotation.z;
        let zw = rotation.z * rotation.w;
        return SoaMat4 {
            cols: [
                scale.x * (ONE - TWO * (yy + zz)),
                scale.x * TWO * (xy + zw),
                scale.x * TWO * (xz - yw),
                ZERO,
                scale.y * TWO * (xy - zw),
                scale.y * (ONE - TWO * (xx + zz)),
                scale.y * (TWO * (yz + xw)),
                ZERO,
                scale.z * TWO * (xz + yw),
                scale.z * TWO * (yz - xw),
                scale.z * (ONE - TWO * (xx + yy)),
                ZERO,
                translation.x,
                translation.y,
                translation.z,
                ONE,
            ],
        };
    }

    pub(crate) fn to_aos(&self) -> [AosMat4; 4] {
        const LOW: [usize; 4] = [0, 4, 1, 5];
        const HIGH: [usize; 4] = [2, 6, 3, 7];

        let mut aos = [f32x4::default(); 16];

        let tmp0 = simd_swizzle!(self.cols[0], self.cols[2], LOW);
        let tmp1 = simd_swizzle!(self.cols[1], self.cols[3], LOW);
        aos[0] = simd_swizzle!(tmp0, tmp1, LOW);
        aos[4] = simd_swizzle!(tmp0, tmp1, HIGH);
        let tmp2 = simd_swizzle!(self.cols[0], self.cols[2], HIGH);
        let tmp3 = simd_swizzle!(self.cols[1], self.cols[3], HIGH);
        aos[8] = simd_swizzle!(tmp2, tmp3, LOW);
        aos[12] = simd_swizzle!(tmp2, tmp3, HIGH);
        let tmp4 = simd_swizzle!(self.cols[4], self.cols[6], LOW);
        let tmp5 = simd_swizzle!(self.cols[5], self.cols[7], LOW);
        aos[1] = simd_swizzle!(tmp4, tmp5, LOW);
        aos[5] = simd_swizzle!(tmp4, tmp5, HIGH);
        let tmp6 = simd_swizzle!(self.cols[4], self.cols[6], HIGH);
        let tmp7 = simd_swizzle!(self.cols[5], self.cols[7], HIGH);
        aos[9] = simd_swizzle!(tmp6, tmp7, LOW);
        aos[13] = simd_swizzle!(tmp6, tmp7, HIGH);
        let tmp8 = simd_swizzle!(self.cols[8], self.cols[10], LOW);
        let tmp9 = simd_swizzle!(self.cols[9], self.cols[11], LOW);
        aos[2] = simd_swizzle!(tmp8, tmp9, LOW);
        aos[6] = simd_swizzle!(tmp8, tmp9, HIGH);
        let tmp10 = simd_swizzle!(self.cols[8], self.cols[10], HIGH);
        let tmp11 = simd_swizzle!(self.cols[9], self.cols[11], HIGH);
        aos[10] = simd_swizzle!(tmp10, tmp11, LOW);
        aos[14] = simd_swizzle!(tmp10, tmp11, HIGH);
        let tmp12 = simd_swizzle!(self.cols[12], self.cols[14], LOW);
        let tmp13 = simd_swizzle!(self.cols[13], self.cols[15], LOW);
        aos[3] = simd_swizzle!(tmp12, tmp13, LOW);
        aos[7] = simd_swizzle!(tmp12, tmp13, HIGH);
        let tmp14 = simd_swizzle!(self.cols[12], self.cols[14], HIGH);
        let tmp15 = simd_swizzle!(self.cols[13], self.cols[15], HIGH);
        aos[11] = simd_swizzle!(tmp14, tmp15, LOW);
        aos[15] = simd_swizzle!(tmp14, tmp15, HIGH);

        return unsafe { mem::transmute(aos) };
    }
}

//
// functions
//

pub(crate) fn f16_to_f32(n: u16) -> f32 {
    let sign = (n & 0x8000) as u32;
    let expo = (n & 0x7C00) as u32;
    let base = (n & 0x03FF) as u32;
    if expo == 0x7C00 {
        if base != 0 {
            return f32::NAN;
        }
        if sign == 0x8000 {
            return f32::NEG_INFINITY;
        } else {
            return f32::INFINITY;
        }
    }
    let expmant = (n & 0x7FFF) as u32;
    unsafe {
        let magic = mem::transmute::<u32, f32>((254 - 15) << 23);
        let shifted = mem::transmute::<u32, f32>(expmant << 13);
        let scaled = mem::transmute::<f32, u32>(shifted * magic);
        return mem::transmute::<u32, f32>((sign << 16) | scaled);
    };
}

pub(crate) fn simd_f16_to_f32(half4: [u16; 4]) -> f32x4 {
    const MASK_NO_SIGN: i32x4 = i32x4::from_array([0x7FFF; 4]);
    const MAGIC: f32x4 = fx4(i32x4::from_array([(254 - 15) << 23; 4]));
    const WAS_INFNAN: i32x4 = i32x4::from_array([0x7BFF; 4]);
    const EXP_INFNAN: i32x4 = i32x4::from_array([255 << 23; 4]);

    let int4 = i32x4::from([half4[0] as i32, half4[1] as i32, half4[2] as i32, half4[3] as i32]);
    let expmant = MASK_NO_SIGN & int4;
    let shifted = expmant << 13;
    let scaled = fx4(shifted) * MAGIC;
    let was_infnan = i32x4::simd_ge(expmant, WAS_INFNAN).to_int();
    let sign = (int4 ^ expmant) << 16;
    let infnanexp = was_infnan & EXP_INFNAN;
    let sign_inf = sign | infnanexp;
    let float4 = ix4(scaled) | sign_inf;
    return fx4(float4);
}

#[inline(always)]
pub(crate) const fn fx4(v: i32x4) -> f32x4 {
    return unsafe { mem::transmute(v) };
}

#[inline(always)]
pub(crate) const fn ix4(v: f32x4) -> i32x4 {
    return unsafe { mem::transmute(v) };
}

const_assert_eq!(mem::size_of::<f32x4>(), mem::size_of::<Vec3A>());

#[inline(always)]
pub(crate) fn fx4_from_vec3a(v: Vec3A) -> f32x4 {
    return unsafe { mem::transmute(v) };
}

#[inline(always)]
pub(crate) fn fx4_to_vec3a(v: f32x4) -> Vec3A {
    return unsafe { mem::transmute(v) };
}

const_assert_eq!(mem::size_of::<f32x4>(), mem::size_of::<Vec4>());

#[inline(always)]
pub(crate) fn fx4_from_vec4(v: Vec4) -> f32x4 {
    return unsafe { mem::transmute(v) };
}

#[inline(always)]
pub(crate) fn fx4_to_vec4(v: f32x4) -> Vec4 {
    return unsafe { mem::transmute(v) };
}

const_assert_eq!(mem::size_of::<f32x4>(), mem::size_of::<Quat>());

#[inline(always)]
pub(crate) fn fx4_from_quat(q: Quat) -> f32x4 {
    return unsafe { mem::transmute(q) };
}

#[inline(always)]
pub(crate) fn fx4_to_quat(q: f32x4) -> Quat {
    return unsafe { mem::transmute(q) };
}

#[inline(always)]
pub(crate) fn fx4_set_y(a: f32x4, b: f32x4) -> f32x4 {
    return simd_swizzle!(a, b, [0, 4, 2, 3]);
}

#[inline(always)]
pub(crate) fn fx4_set_z(a: f32x4, b: f32x4) -> f32x4 {
    return simd_swizzle!(a, b, [0, 1, 4, 3]);
}

#[inline(always)]
pub(crate) fn fx4_set_w(a: f32x4, b: f32x4) -> f32x4 {
    return simd_swizzle!(a, b, [0, 1, 2, 4]);
}

#[inline(always)]
pub(crate) fn fx4_splat_x(v: f32x4) -> f32x4 {
    return simd_swizzle!(v, [0, 0, 0, 0]);
}

#[inline(always)]
pub(crate) fn fx4_splat_y(v: f32x4) -> f32x4 {
    return simd_swizzle!(v, [1, 1, 1, 1]);
}

#[inline(always)]
pub(crate) fn fx4_splat_z(v: f32x4) -> f32x4 {
    return simd_swizzle!(v, [2, 2, 2, 2]);
}

#[inline(always)]
pub(crate) fn fx4_splat_w(v: f32x4) -> f32x4 {
    return simd_swizzle!(v, [3, 3, 3, 3]);
}

#[inline(always)]
pub(crate) fn ix4_splat_x(v: i32x4) -> i32x4 {
    return simd_swizzle!(v, [0, 0, 0, 0]);
}

#[inline(always)]
pub(crate) fn ix4_splat_y(v: i32x4) -> i32x4 {
    return simd_swizzle!(v, [1, 1, 1, 1]);
}

#[inline(always)]
pub(crate) fn ix4_splat_z(v: i32x4) -> i32x4 {
    return simd_swizzle!(v, [2, 2, 2, 2]);
}

#[inline(always)]
pub(crate) fn ix4_splat_w(v: i32x4) -> i32x4 {
    return simd_swizzle!(v, [3, 3, 3, 3]);
}

#[inline(always)]
pub(crate) fn fx4_sign(v: f32x4) -> i32x4 {
    // In some case, x86_64 and aarch64 may produce different sign NaN (+/-NaN) in same command.
    // So the result of `NaN & SIGN` may be different.
    // For cross-platform deterministic, we want to make sure the sign of NaN is always 0(+).
    return v.simd_lt(ZERO).to_int() & SIGN;
}

#[inline(always)]
pub(crate) fn fx4_and(v: f32x4, s: i32x4) -> f32x4 {
    return fx4(ix4(v) & s);
}

#[inline(always)]
pub(crate) fn fx4_or(v: f32x4, s: i32x4) -> f32x4 {
    return fx4(ix4(v) | s);
}

#[inline(always)]
pub(crate) fn fx4_xor(v: f32x4, s: i32x4) -> f32x4 {
    return fx4(ix4(v) ^ s);
}

#[inline(always)]
pub(crate) fn fx4_clamp_or_max(v: f32x4, min: f32x4, max: f32x4) -> f32x4 {
    // f32x4::clamp may produce NaN if self is NaN.
    // This implementation always returns max if self is NaN.
    return v.simd_min(max).simd_max(min);
}

#[inline(always)]
pub(crate) fn fx4_clamp_or_min(v: f32x4, min: f32x4, max: f32x4) -> f32x4 {
    // f32x4::clamp may produce NaN if self is NaN.
    // This implementation always returns min if self is NaN.
    return v.simd_max(min).simd_min(max);
}

#[inline(always)]
pub(crate) fn f32_clamp_or_max(v: f32, min: f32, max: f32) -> f32 {
    return v.min(max).max(min);
}

#[inline(always)]
pub(crate) fn f32_clamp_or_min(v: f32, min: f32, max: f32) -> f32 {
    return v.max(min).min(max);
}

pub(crate) fn fx4_sin_cos(v: f32x4) -> (f32x4, f32x4) {
    // Implementation based on Vec4.inl from the JoltPhysics
    // https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Math/Vec4.inl

    const N1: f32x4 = f32x4::from_array([1.5703125; 4]);
    const N2: f32x4 = f32x4::from_array([0.0004837512969970703125; 4]);
    const N3: f32x4 = f32x4::from_array([7.549789948768648e-8; 4]);

    const C1: f32x4 = f32x4::from_array([2.443315711809948e-5; 4]);
    const C2: f32x4 = f32x4::from_array([1.388731625493765e-3; 4]);
    const C3: f32x4 = f32x4::from_array([4.166664568298827e-2; 4]);

    const S1: f32x4 = f32x4::from_array([-1.9515295891e-4; 4]);
    const S2: f32x4 = f32x4::from_array([8.3321608736e-3; 4]);
    const S3: f32x4 = f32x4::from_array([1.6666654611e-1; 4]);

    // Make argument positive and remember sign for sin only since cos is symmetric around x (highest bit of a float is the sign bit)
    let mut sin_sign = fx4_sign(v);
    let mut x = v.abs();

    // x / (PI / 2) rounded to nearest int gives us the quadrant closest to x
    let quadrant = (FRAC_2_PI * x + FRAC_1_2).trunc();

    // Make x relative to the closest quadrant.
    // This does x = x - quadrant * PI / 2 using a two step Cody-Waite argument reduction.
    // This improves the accuracy of the result by avoiding loss of significant bits in the subtraction.
    // We start with x = x - quadrant * PI / 2, PI / 2 in hexadecimal notation is 0x3fc90fdb, we remove the lowest 16 bits to
    // get 0x3fc90000 (= 1.5703125) this means we can now multiply with a number of up to 2^16 without losing any bits.
    // This leaves us with: x = (x - quadrant * 1.5703125) - quadrant * (PI / 2 - 1.5703125).
    // PI / 2 - 1.5703125 in hexadecimal is 0x39fdaa22, stripping the lowest 12 bits we get 0x39fda000 (= 0.0004837512969970703125)
    // This leaves uw with: x = ((x - quadrant * 1.5703125) - quadrant * 0.0004837512969970703125) - quadrant * (PI / 2 - 1.5703125 - 0.0004837512969970703125)
    // See: https://stackoverflow.com/questions/42455143/sine-cosine-modular-extended-precision-arithmetic
    // After this we have x in the range [-PI / 4, PI / 4].
    x = ((x - quadrant * N1) - quadrant * N2) - quadrant * N3;

    // Calculate x2 = x^2
    let x2 = x * x;

    // Taylor expansion:
    // Cos(x) = 1 - x^2/2! + x^4/4! - x^6/6! + x^8/8! + ... = (((x2/8!- 1/6!) * x2 + 1/4!) * x2 - 1/2!) * x2 + 1
    let taylor_cos = ((C1 * x2 - C2) * x2 + C3) * x2 * x2 - FRAC_1_2 * x2 + ONE;
    // Sin(x) = x - x^3/3! + x^5/5! - x^7/7! + ... = ((-x2/7! + 1/5!) * x2 - 1/3!) * x2 * x + x
    let taylor_sin = ((S1 * x2 + S2) * x2 - S3) * x2 * x + x;

    // The lowest 2 bits of quadrant indicate the quadrant that we are in.
    // Let x be the original input value and x' our value that has been mapped to the range [-PI / 4, PI / 4].
    // since cos(x) = sin(x - PI / 2) and since we want to use the Taylor expansion as close as possible to 0,
    // we can alternate between using the Taylor expansion for sin and cos according to the following table:
    //
    // quadrant	 sin(x)		 cos(x)
    // XXX00b	 sin(x')	 cos(x')
    // XXX01b	 cos(x')	-sin(x')
    // XXX10b	-sin(x')	-cos(x')
    // XXX11b	-cos(x')	 sin(x')
    //
    // So: sin_sign = bit2, cos_sign = bit1 ^ bit2, bit1 determines if we use sin or cos Taylor expansion
    let quadrant_int: i32x4 = unsafe { quadrant.to_int_unchecked() };
    let bit1 = quadrant_int << 31;
    let bit2 = (quadrant_int << 30) & SIGN;

    // Select which one of the results is sin and which one is cos
    let cond = bit1.simd_eq(SIGN);
    let s = cond.select(taylor_cos, taylor_sin);
    let c = cond.select(taylor_sin, taylor_cos);

    // Update the signs
    sin_sign = sin_sign ^ bit2;
    let cos_sign = bit1 ^ bit2;

    // Correct the signs
    let out_sin = fx4_xor(s, sin_sign);
    let out_cos = fx4_xor(c, cos_sign);
    return (out_sin, out_cos);
}

#[inline]
pub fn f32_sin_cos(x: f32) -> (f32, f32) {
    let (sin, cos) = fx4_sin_cos(f32x4::splat(x));
    return (sin[0], cos[0]);
}

#[inline]
pub fn f32_sin(x: f32) -> f32 {
    let (sin, _) = fx4_sin_cos(f32x4::splat(x));
    return sin[0];
}

#[inline]
pub fn f32_cos(x: f32) -> f32 {
    let (_, cos) = fx4_sin_cos(f32x4::splat(x));
    return cos[0];
}

pub(crate) fn fx4_asin(v: f32x4) -> f32x4 {
    // Implementation based on Vec4.inl from the JoltPhysics
    // https://github.com/jrouwe/JoltPhysics/blob/master/Jolt/Math/Vec4.inl

    const N1: f32x4 = f32x4::from_array([4.2163199048e-2; 4]);
    const N2: f32x4 = f32x4::from_array([2.4181311049e-2; 4]);
    const N3: f32x4 = f32x4::from_array([4.5470025998e-2; 4]);
    const N4: f32x4 = f32x4::from_array([7.4953002686e-2; 4]);
    const N5: f32x4 = f32x4::from_array([1.6666752422e-1; 4]);

    // Make argument positive
    let asin_sign = fx4_sign(v);
    let mut a = v.abs();

    // ASin is not defined outside the range [-1, 1] but it often happens that a value is slightly above 1 so we just clamp here
    a = f32x4::simd_min(a, ONE);

    // When |x| <= 0.5 we use the asin approximation as is
    let z1 = a * a;
    let x1 = a;

    // When |x| > 0.5 we use the identity asin(x) = PI / 2 - 2 * asin(sqrt((1 - x) / 2))
    let z2 = FRAC_1_2 * (ONE - a);
    let x2 = z2.sqrt();

    // Select which of the two situations we have
    let greater = f32x4::simd_gt(a, FRAC_1_2);
    let mut z = greater.select(z2, z1);
    let x = greater.select(x2, x1);

    // Polynomial approximation of asin
    z = ((((N1 * z + N2) * z + N3) * z + N4) * z + N5) * z * x + x;

    // If |x| > 0.5 we need to apply the remainder of the identity above
    z = greater.select(FRAC_PI_2 - (z + z), z);

    // Put the sign back
    return fx4_xor(z, asin_sign);
}

#[inline]
pub(crate) fn fx4_acos(v: f32x4) -> f32x4 {
    const FRAC_PI_2: f32x4 = f32x4::from_array([core::f32::consts::FRAC_PI_2; 4]);
    return FRAC_PI_2 - fx4_asin(v);
}

#[inline]
pub fn f32_asin(x: f32) -> f32 {
    return fx4_asin(f32x4::splat(x))[0];
}

#[inline]
pub fn f32_acos(x: f32) -> f32 {
    return fx4_acos(f32x4::splat(x))[0];
}

#[inline]
pub(crate) fn fx4_lerp(from: f32x4, to: f32x4, alpha: f32x4) -> f32x4 {
    return alpha * (to - from) + from;
}

#[inline]
pub(crate) fn vec3_is_normalized(v: f32x4) -> bool {
    const MAX: f32 = 1.0 + 0.002;
    const MIN: f32 = 1.0 - 0.002;
    let len2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    return (MIN < len2) & (len2 < MAX);
}

#[inline]
pub(crate) fn vec3_length2_s(v: f32x4) -> f32x4 {
    return vec3_dot_s(v, v);
}

#[inline]
pub(crate) fn vec3_dot_s(a: f32x4, b: f32x4) -> f32x4 {
    let tmp = a * b;
    return tmp + simd_swizzle!(tmp, [1; 4]) + simd_swizzle!(tmp, [2; 4]);
}

pub(crate) fn vec3_cross(a: f32x4, b: f32x4) -> f32x4 {
    let shufa = simd_swizzle!(a, [1, 2, 0, 3]);
    let shufb = simd_swizzle!(b, [1, 2, 0, 3]);
    let shufc = a * shufb - b * shufa;
    return simd_swizzle!(shufc, [1, 2, 0, 3]);
}

pub(crate) fn quat_from_axis_angle(axis: f32x4, angle: f32x4) -> f32x4 {
    let half_angle = fx4_splat_x(angle) * FRAC_1_2;
    let (half_sin, half_cos) = fx4_sin_cos(half_angle);
    return fx4_set_w(axis * half_sin, half_cos);
}

pub(crate) fn quat_from_cos_angle(axis: f32x4, cos: f32x4) -> f32x4 {
    let half_cos2 = (ONE + cos) * FRAC_1_2;
    let half_sin2 = ONE - half_cos2;
    let half_cossin2 = fx4_set_y(half_cos2, half_sin2);
    let half_cossin = half_cossin2.sqrt();
    let half_sin = fx4_splat_y(half_cossin);
    return fx4_set_w(axis * half_sin, half_cossin);
}

pub(crate) fn quat_from_vectors(from: f32x4, to: f32x4) -> f32x4 {
    let norm_from_norm_to = (vec3_length2_s(from) * vec3_length2_s(to)).sqrt();
    let norm_from_norm_to_x = norm_from_norm_to[0];
    if norm_from_norm_to_x < 1.0e-6 {
        return QUAT_UNIT;
    }

    let real_part = norm_from_norm_to + vec3_dot_s(from, to);
    let real_part_x = real_part[0];

    let quat;
    if real_part_x < 1.0e-6 * norm_from_norm_to_x {
        if from[0].abs() > from[2].abs() {
            quat = f32x4::from_array([-from[1], from[0], 0.0, 0.0])
        } else {
            quat = f32x4::from_array([0.0, -from[2], from[1], 0.0])
        }
    } else {
        quat = fx4_set_w(vec3_cross(from, to), real_part)
    };

    return quat_normalize(quat);
}

#[inline]
pub(crate) fn quat_length2_s(q: f32x4) -> f32x4 {
    let q2 = (q * q).reduce_sum();
    return f32x4::splat(q2);
}

#[inline]
pub(crate) fn quat_normalize(q: f32x4) -> f32x4 {
    let q2 = q * q;
    let len2 = q2.reduce_sum();
    let inv_len = f32x4::splat(len2).sqrt().recip();
    return q * inv_len;
}

#[inline]
pub(crate) fn quat_transform_vector(q: f32x4, v: f32x4) -> f32x4 {
    let cross1 = fx4_splat_w(q) * v + vec3_cross(q, v);
    let cross2 = vec3_cross(q, cross1);
    return v + cross2 + cross2;
}

#[inline]
pub(crate) fn quat_mul(a: f32x4, b: f32x4) -> f32x4 {
    let p1 = simd_swizzle!(a, [3, 3, 3, 2]) * simd_swizzle!(b, [0, 1, 2, 2]);
    let p2 = simd_swizzle!(a, [0, 1, 2, 0]) * simd_swizzle!(b, [3, 3, 3, 0]);
    let p13 = simd_swizzle!(a, [1, 2, 0, 1]) * simd_swizzle!(b, [2, 0, 1, 1]) + p1;
    let p24 = p2 - simd_swizzle!(a, [2, 0, 1, 3]) * simd_swizzle!(b, [1, 2, 0, 3]);
    return fx4_xor(p13 + p24, SIGN_W);
}

#[inline]
pub(crate) fn quat_positive_w(q: f32x4) -> f32x4 {
    let s = fx4_splat_w(q).simd_lt(ZERO).to_int() & SIGN;
    return fx4_xor(q, s);
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_f16_to_f32() {
        assert_eq!(f16_to_f32(0b00111100_00000000), 1.0f32);
        assert_eq!(f16_to_f32(0b10111100_00000000), -1.0f32);
        assert_eq!(f16_to_f32(0b01000011_00000000), 3.5f32);
        assert_eq!(f16_to_f32(0b01111100_00000000), f32::INFINITY);
        assert_eq!(f16_to_f32(0b11111100_00000000), f32::NEG_INFINITY);
        assert_eq!(f16_to_f32(32791), -1.37090683e-06);
        assert_eq!(f16_to_f32(0), 0.0f32);
        assert_eq!(f16_to_f32(0x8000), 0.0f32);
        assert!(f16_to_f32(0xFFFF).is_nan());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_simd_f16_to_f32() {
        let half4 = [
            0b00111100_00000000,
            0b10111100_00000000,
            0b01000011_00000000,
            0b01111100_00000000,
        ];
        let float4 = simd_f16_to_f32(half4);
        assert_eq!(float4, f32x4::from_array([1.0f32, -1.0f32, 3.5f32, f32::INFINITY]));

        let half4 = [0b11111100_00000000, 0, 0x8000, 32791];
        let float4 = simd_f16_to_f32(half4);
        assert_eq!(
            float4,
            f32x4::from_array([f32::NEG_INFINITY, 0.0f32, 0.0f32, -1.37090683e-06])
        );

        let half4 = [0xFFFF, 0, 0, 0];
        let float4 = simd_f16_to_f32(half4);
        assert!(float4[0].is_nan());
    }

    #[test]
    #[wasm_bindgen_test]
    #[rustfmt::skip]
    fn test_matrix_invert() {
        let m = AosMat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );
        let m_inv = m.invert();
        assert_eq!(m_inv, AosMat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ));

        let m = AosMat4::new(
            5.0, 0.0, 0.0, 0.0,
            0.0, 5.0, 0.0, 0.0,
            0.0, 0.0, 5.0, 0.0,
            0.0, 0.0, 0.0, 5.0,
        );
        let m_inv = m.invert();
        assert_eq!(m_inv, AosMat4::new(
            0.2, 0.0, 0.0, 0.0,
            0.0, 0.2, 0.0, 0.0,
            0.0, 0.0, 0.2, 0.0,
            0.0, 0.0, 0.0, 0.2,
        ));

        let m = AosMat4::new(
            -1.82742070e-08, 0.988180459, 0.153295174, 0.0,
            1.17800290e-07, 0.153295159, -0.988180578, 0.0,
            -1.00000012, 0.0, -1.19209290e-07, 0.0,
            0.0, 0.0999999940, 0.0, 1.0,
        );
        let m_inv = m.invert();
        assert_eq!(
            m_inv,
            AosMat4::new(
                -1.82742035e-08, 1.17800262e-07, -0.999999881, 0.0,
                0.988180459, 0.153295159, 0.0, 0.0,
                0.153295144, -0.988180339, -1.19209261e-07, 0.0,
                -0.0988180414, -0.0153295146, 0.0, 1.0,
            )
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sin_cos() {
        const EPSILON: f32x4 = f32x4::from_array([2.0e-7; 4]);

        let (sin, cos) = fx4_sin_cos(f32x4::from_array([
            0.0,
            core::f32::consts::FRAC_PI_2,
            core::f32::consts::PI,
            -core::f32::consts::FRAC_PI_2,
        ]));
        assert!((sin - f32x4::from_array([0.0, 1.0, 0.0, -1.0]))
            .abs()
            .simd_lt(EPSILON)
            .all());
        assert!((cos - f32x4::from_array([1.0, 0.0, -1.0, 0.0]))
            .abs()
            .simd_lt(EPSILON)
            .all());

        let mut ms: f64 = 0.0;
        let mut mc: f64 = 0.0;

        let mut i = -100.0 * core::f32::consts::PI;
        while i < 100.0 * core::f32::consts::PI {
            let iv = f32x4::splat(i) + f32x4::from_array([0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4]);
            let (sin, cos) = fx4_sin_cos(iv);

            for i in 0..4 {
                let sin_std = (iv[i] as f64).sin();
                let ds = (sin[i] as f64 - sin_std).abs();
                ms = ms.max(ds);

                let cos_std = (iv[i] as f64).cos();
                let dc = (cos[i] as f64 - cos_std).abs();
                mc = mc.max(dc);
            }

            i += 1.0e-3;
        }

        assert!(ms < 2.5e-5);
        assert!(mc < 2.5e-5);
    }

    fn approx_eq(a: f32x4, b: f32, epsilon: f32) -> bool {
        return (a - f32x4::splat(b)).abs().simd_lt(f32x4::splat(epsilon)).all();
    }
    #[test]
    #[wasm_bindgen_test]
    fn test_asin() {
        assert_eq!(fx4_asin(f32x4::splat(0.0))[0], 0.0);
        assert_eq!(fx4_asin(f32x4::splat(1.0))[0], core::f32::consts::FRAC_PI_2);
        assert_eq!(fx4_asin(f32x4::splat(-1.0))[0], -core::f32::consts::FRAC_PI_2);
        assert_eq!(fx4_asin(f32x4::splat(1.1))[0], core::f32::consts::FRAC_PI_2);
        assert_eq!(fx4_asin(f32x4::splat(-1.1))[0], -core::f32::consts::FRAC_PI_2);

        let mut ma: f64 = 0.0;

        let mut i = -1.0;
        while i < 1.0 {
            let iv = f32x4::splat(i) + f32x4::from_array([0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4]).simd_min(f32x4::splat(1.0));
            let asin = fx4_asin(iv);

            for i in 0..4 {
                let asin_std = (iv[i] as f64).asin();
                let da = (asin[i] as f64 - asin_std).abs();
                ma = ma.max(da);
            }

            i += 1.0e-3;
        }

        assert!(ma < 2.0e-7);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_acos() {
        assert_eq!(fx4_acos(f32x4::splat(0.0))[0], core::f32::consts::FRAC_PI_2);
        assert_eq!(fx4_acos(f32x4::splat(1.0))[0], 0.0);
        assert_eq!(fx4_acos(f32x4::splat(-1.0))[0], core::f32::consts::PI);
        assert_eq!(fx4_acos(f32x4::splat(1.1))[0], 0.0);
        assert_eq!(fx4_acos(f32x4::splat(-1.1))[0], core::f32::consts::PI);

        let mut ma: f64 = 0.0;

        let mut i = -1.0;
        while i < 1.0 {
            let iv = f32x4::splat(i) + f32x4::from_array([0.0e-4, 2.5e-4, 5.0e-4, 7.5e-4]).simd_min(f32x4::splat(1.0));
            let acos = fx4_acos(iv);

            for i in 0..4 {
                let acos_std = (iv[i] as f64).acos();
                let da = (acos[i] as f64 - acos_std).abs();
                ma = ma.max(da);
            }

            i += 1.0e-3;
        }

        assert!(ma < 3.5e-7);
    }

    #[cfg(feature = "rkyv")]
    #[test]
    #[wasm_bindgen_test]
    fn test_rkyv() {
        use rkyv::Deserialize;

        let vec3 = SoaVec3::splat_col([2.0, 3.0, 4.0]);
        let bytes = rkyv::to_bytes::<_, 256>(&vec3).unwrap();
        let archived = rkyv::check_archived_root::<SoaVec3>(&bytes[..]).unwrap();
        assert_eq!(archived, &vec3);
        let vec3_de: SoaVec3 = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(vec3_de, vec3);

        let quat = SoaQuat::splat_col([2.0, 3.0, 4.0, 5.0]);
        let bytes = rkyv::to_bytes::<_, 256>(&quat).unwrap();
        let archived = rkyv::check_archived_root::<SoaQuat>(&bytes[..]).unwrap();
        assert_eq!(archived, &quat);
        let quat_de: SoaQuat = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(quat_de, quat);

        let transform = SoaTransform::new(
            SoaVec3::splat_col([9.0, 8.0, 7.0]),
            SoaQuat::splat_col([6.0, 5.0, 4.0, 3.0]),
            SoaVec3::splat_col([-1.0, -2.0, -3.0]),
        );
        let bytes = rkyv::to_bytes::<_, 256>(&transform).unwrap();
        let archived = rkyv::check_archived_root::<SoaTransform>(&bytes[..]).unwrap();
        assert_eq!(archived, &transform);
        let transform_de: SoaTransform = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(transform_de, transform);
    }

    #[cfg(feature = "serde")]
    #[test]
    #[wasm_bindgen_test]
    fn test_serde() {
        let vec3 = SoaVec3::splat_col([2.0, 3.0, 4.0]);
        let json = serde_json::to_string(&vec3).unwrap();
        let vec3_de: SoaVec3 = serde_json::from_str(&json).unwrap();
        assert_eq!(vec3_de, vec3);
        
        let quat = SoaQuat::splat_col([2.0, 3.0, 4.0, 5.0]);
        let json = serde_json::to_string(&quat).unwrap();
        let quat_de: SoaQuat = serde_json::from_str(&json).unwrap();
        assert_eq!(quat_de, quat);
    }
}
