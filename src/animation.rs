//!
//! Animation data structure definition.
//!

use glam::{Quat, Vec3, Vec4};
use std::alloc::{self, Layout};
use std::io::Read;
use std::simd::prelude::*;
use std::simd::*;
use std::{mem, slice};

use crate::archive::{Archive, ArchiveRead};
use crate::base::{align_ptr, align_usize, OzzError};
use crate::math::{f16_to_f32, fx4, ix4, simd_f16_to_f32, SoaQuat, SoaVec3, ONE, ZERO};

/// Float3 key for `Animation` track.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Float3Key([u16; 3]);

impl Float3Key {
    pub const fn new(value: [u16; 3]) -> Float3Key {
        Float3Key(value)
    }

    #[inline]
    pub fn decompress(&self) -> Vec3 {
        Vec3::new(f16_to_f32(self.0[0]), f16_to_f32(self.0[1]), f16_to_f32(self.0[2]))
    }

    #[inline]
    pub fn simd_decompress(k0: &Float3Key, k1: &Float3Key, k2: &Float3Key, k3: &Float3Key, soa: &mut SoaVec3) {
        soa.x = simd_f16_to_f32([k0.0[0], k1.0[0], k2.0[0], k3.0[0]]);
        soa.y = simd_f16_to_f32([k0.0[1], k1.0[1], k2.0[1], k3.0[1]]);
        soa.z = simd_f16_to_f32([k0.0[2], k1.0[2], k2.0[2], k3.0[2]]);
    }
}

impl ArchiveRead<Float3Key> for Float3Key {
    #[inline]
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<Float3Key, OzzError> {
        let value: [u16; 3] = [archive.read()?, archive.read()?, archive.read()?];
        Ok(Float3Key(value))
    }
}

/// Quaternion key for `Animation` track.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct QuaternionKey([u16; 3]);

impl QuaternionKey {
    pub const fn new(value: [u16; 3]) -> QuaternionKey {
        QuaternionKey(value)
    }

    #[inline]
    fn unpack(&self) -> (u16, u16, [u32; 3]) {
        let packed: u32 = ((self.0[0] as u32) >> 3) | ((self.0[1] as u32) << 13) | ((self.0[2] as u32) << 29);
        let bigest = self.0[0] & 0x3;
        let sign = (self.0[0] >> 2) & 0x1;
        let value = [packed & 0x7fff, (packed >> 15) & 0x7fff, (self.0[2] as u32) >> 1];
        (bigest, sign, value)
    }

    #[inline]
    pub fn decompress(&self) -> Quat {
        const MAPPING: [[usize; 4]; 4] = [[0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 0]];
        const SCALE: f32 = core::f32::consts::SQRT_2 / 32767.0;
        const OFFSET: f32 = -core::f32::consts::SQRT_2 / 2.0;

        let (largest, sign, value) = self.unpack();
        let mask = &MAPPING[largest as usize];
        let cmp_keys = [value[mask[0]], value[mask[1]], value[mask[2]], value[mask[3]]];

        let mut cpnt = Vec4::new(
            SCALE * (cmp_keys[0] as f32) + OFFSET,
            SCALE * (cmp_keys[1] as f32) + OFFSET,
            SCALE * (cmp_keys[2] as f32) + OFFSET,
            SCALE * (cmp_keys[3] as f32) + OFFSET,
        );
        cpnt[largest as usize] = 0.0;

        let dot = cpnt[0] * cpnt[0] + cpnt[1] * cpnt[1] + cpnt[2] * cpnt[2] + cpnt[3] * cpnt[3];
        let ww0 = f32::max(0.0, 1f32 - dot);
        let w0 = ww0.sqrt();
        let restored = if sign == 0 { w0 } else { -w0 };
        cpnt[largest as usize] = restored;
        Quat::from_vec4(cpnt)
    }

    #[rustfmt::skip]
    #[inline]
    pub fn simd_decompress(
        k0: &QuaternionKey,
        k1: &QuaternionKey,
        k2: &QuaternionKey,
        k3: &QuaternionKey,
        soa: &mut SoaQuat,
    ) {
        const MASK_F000:i32x4 = i32x4::from_array([-1i32, 0, 0, 0]);
        const MASK_0F00:i32x4 = i32x4::from_array([0, -1i32, 0, 0]);
        const MASK_00F0:i32x4 = i32x4::from_array([0, 0, -1i32, 0]);
        const MASK_000F:i32x4 = i32x4::from_array([0, 0, 0, -1i32]);

        const MAPPING: [[usize; 4]; 4] = [[0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 0]];

        const SCALE: f32x4 = f32x4::from_array([core::f32::consts::SQRT_2 / 32767.0; 4]);
        const OFFSET: f32x4 = f32x4::from_array([-core::f32::consts::SQRT_2 / 2.0; 4]);

        let (largest0, sign0, value0) = k0.unpack();
        let (largest1, sign1, value1) = k1.unpack();
        let (largest2, sign2, value2) = k2.unpack();
        let (largest3, sign3, value3) = k3.unpack();

        let m0 = &MAPPING[largest0 as usize];
        let m1 = &MAPPING[largest1 as usize];
        let m2 = &MAPPING[largest2 as usize];
        let m3 = &MAPPING[largest3 as usize];

        let cmp_keys: [f32x4; 4] = [
            f32x4::from_array([ value0[m0[0]] as f32, value1[m1[0]] as f32, value2[m2[0]] as f32, value3[m3[0]] as f32 ]),
            f32x4::from_array([ value0[m0[1]] as f32, value1[m1[1]] as f32, value2[m2[1]] as f32, value3[m3[1]] as f32 ]),
            f32x4::from_array([ value0[m0[2]] as f32, value1[m1[2]] as f32, value2[m2[2]] as f32, value3[m3[2]] as f32 ]),
            f32x4::from_array([ value0[m0[3]] as f32, value1[m1[3]] as f32, value2[m2[3]] as f32, value3[m3[3]] as f32 ]),
        ]; // TODO: simd int to float

        let mut cpnt = [
            SCALE * cmp_keys[0] + OFFSET,
            SCALE * cmp_keys[1] + OFFSET,
            SCALE * cmp_keys[2] + OFFSET,
            SCALE * cmp_keys[3] + OFFSET,
        ];
        cpnt[largest0 as usize] = fx4(ix4(cpnt[largest0 as usize]) & !MASK_F000);
        cpnt[largest1 as usize] = fx4(ix4(cpnt[largest1 as usize]) & !MASK_0F00);
        cpnt[largest2 as usize] = fx4(ix4(cpnt[largest2 as usize]) & !MASK_00F0);
        cpnt[largest3 as usize] = fx4(ix4(cpnt[largest3 as usize]) & !MASK_000F);

        let dot = cpnt[0] * cpnt[0] + cpnt[1] * cpnt[1] + cpnt[2] * cpnt[2] + cpnt[3] * cpnt[3];
        let ww0 =  f32x4::simd_max(ZERO, ONE - dot); // prevent NaN, different from C++ code
        let w0 = ww0.sqrt();
        let sign = i32x4::from_array([sign0 as i32, sign1 as i32, sign2 as i32, sign3 as i32]) << 31;
        let restored = ix4(w0) | sign;

        cpnt[largest0 as usize] = fx4(ix4(cpnt[largest0 as usize]) | (restored & MASK_F000));
        cpnt[largest1 as usize] = fx4(ix4(cpnt[largest1 as usize]) | (restored & MASK_0F00));
        cpnt[largest2 as usize] = fx4(ix4(cpnt[largest2 as usize]) | (restored & MASK_00F0));
        cpnt[largest3 as usize] = fx4(ix4(cpnt[largest3 as usize]) | (restored & MASK_000F));

        soa.x = cpnt[0];
        soa.y = cpnt[1];
        soa.z = cpnt[2];
        soa.w = cpnt[3];
    }
}

impl ArchiveRead<QuaternionKey> for QuaternionKey {
    #[inline]
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<QuaternionKey, OzzError> {
        let value: [u16; 3] = [archive.read()?, archive.read()?, archive.read()?];
        Ok(QuaternionKey(value))
    }
}

///
/// Defines a runtime skeletal animation clip.
///
/// The runtime animation data structure stores animation keyframes, for all the
/// joints of a skeleton.
///
/// For each transformation type (translation, rotation and scale), Animation
/// structure stores a single array of keyframes that contains all the tracks
/// required to animate all the joints of a skeleton, matching breadth-first
/// joints order of the runtime skeleton structure. In order to optimize cache
/// coherency when sampling the animation, Keyframes in this array are sorted by
/// time, then by track number.
///
#[derive(Debug)]
// #[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
// #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Animation {
    size: usize,
    duration: f32,
    num_tracks: u32,
    name: String,
    timepoints: *mut f32,
    timepoints_count: u32,

    translations: *mut Float3Key,
    translations_count: u32,
    t_ratios: *mut u16,
    t_previouses: *mut u16,
    t_iframe_interval: f32,
    t_iframe_entries: *mut u8,
    t_iframe_entries_count: u32,
    t_iframe_desc: *mut u32,
    t_iframe_desc_count: u32,

    rotations: *mut QuaternionKey,
    rotations_count: u32,
    r_ratios: *mut u16,
    r_previouses: *mut u16,
    r_iframe_interval: f32,
    r_iframe_entries: *mut u8,
    r_iframe_entries_count: u32,
    r_iframe_desc: *mut u32,
    r_iframe_desc_count: u32,

    scales: *mut Float3Key,
    scales_count: u32,
    s_ratios: *mut u16,
    s_previouses: *mut u16,
    s_iframe_interval: f32,
    s_iframe_entries: *mut u8,
    s_iframe_entries_count: u32,
    s_iframe_desc: *mut u32,
    s_iframe_desc_count: u32,
}

impl Drop for Animation {
    fn drop(&mut self) {
        if !self.timepoints.is_null() {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.size, mem::size_of::<f32>());
                alloc::dealloc(self.timepoints as *mut u8, layout);
            }
            self.timepoints = std::ptr::null_mut();
            self.translations = std::ptr::null_mut();
            self.t_ratios = std::ptr::null_mut();
            self.t_previouses = std::ptr::null_mut();
            self.t_iframe_entries = std::ptr::null_mut();
            self.t_iframe_desc = std::ptr::null_mut();
            self.rotations = std::ptr::null_mut();
            self.r_ratios = std::ptr::null_mut();
            self.r_previouses = std::ptr::null_mut();
            self.r_iframe_entries = std::ptr::null_mut();
            self.r_iframe_desc = std::ptr::null_mut();
            self.scales = std::ptr::null_mut();
            self.s_ratios = std::ptr::null_mut();
            self.s_previouses = std::ptr::null_mut();
            self.s_iframe_entries = std::ptr::null_mut();
            self.s_iframe_desc = std::ptr::null_mut();
        }
    }
}

/// Animation meta in `Archive`.
#[derive(Debug, Default, Clone)]
pub struct AnimationMeta {
    pub version: u32,
    pub duration: f32,
    pub num_tracks: u32,
    pub name: String,
    pub timepoints_count: u32,
    pub translations_count: u32,
    pub t_iframe_entries_count: u32,
    pub t_iframe_desc_count: u32,
    pub rotations_count: u32,
    pub r_iframe_entries_count: u32,
    pub r_iframe_desc_count: u32,
    pub scales_count: u32,
    pub s_iframe_entries_count: u32,
    pub s_iframe_desc_count: u32,
}

#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct AnimationRaw {
    pub duration: f32,
    pub num_tracks: u32,
    pub name: String,
    pub timepoints: Vec<f32>,

    pub translations: Vec<Float3Key>,
    pub t_ratios: Vec<u16>,
    pub t_previouses: Vec<u16>,
    pub t_iframe_interval: f32,
    pub t_iframe_entries: Vec<u8>,
    pub t_iframe_desc: Vec<u32>,

    pub rotations: Vec<QuaternionKey>,
    pub r_ratios: Vec<u16>,
    pub r_previouses: Vec<u16>,
    pub r_iframe_interval: f32,
    pub r_iframe_entries: Vec<u8>,
    pub r_iframe_desc: Vec<u32>,

    pub scales: Vec<Float3Key>,
    pub s_ratios: Vec<u16>,
    pub s_previouses: Vec<u16>,
    pub s_iframe_interval: f32,
    pub s_iframe_entries: Vec<u8>,
    pub s_iframe_desc: Vec<u32>,
}

unsafe impl Send for Animation {}
unsafe impl Sync for Animation {}

impl Animation {
    /// `Animation` resource file tag for `Archive`.
    #[inline]
    pub fn tag() -> &'static str {
        "ozz-animation"
    }

    /// `Animation` resource file version for `Archive`.
    #[inline]
    pub fn version() -> u32 {
        7
    }

    /// Reads an `AnimationMeta` from an `Archive`.
    pub fn read_meta(archive: &mut Archive<impl Read>) -> Result<AnimationMeta, OzzError> {
        if archive.read_tag()? != Self::tag() {
            return Err(OzzError::InvalidTag);
        }
        if archive.read_version()? != Self::version() {
            return Err(OzzError::InvalidVersion);
        }

        let duration: f32 = archive.read()?;
        let num_tracks: u32 = archive.read()?;
        let name_len: u32 = archive.read()?;
        let timepoints_count: u32 = archive.read()?;
        let translations_count: u32 = archive.read()?;
        let rotations_count: u32 = archive.read()?;
        let scales_count: u32 = archive.read()?;
        let t_iframe_entries_count: u32 = archive.read()?;
        let t_iframe_desc_count: u32 = archive.read()?;
        let r_iframe_entries_count: u32 = archive.read()?;
        let r_iframe_desc_count: u32 = archive.read()?;
        let s_iframe_entries_count: u32 = archive.read()?;
        let s_iframe_desc_count: u32 = archive.read()?;

        let mut name = String::new();
        if name_len != 0 {
            let buf = archive.read_vec(name_len as usize)?;
            name = String::from_utf8(buf).map_err(|e| e.utf8_error())?;
        }

        Ok(AnimationMeta {
            version: Self::version(),
            duration,
            num_tracks,
            name,
            timepoints_count,
            translations_count,
            t_iframe_entries_count,
            t_iframe_desc_count,
            rotations_count,
            r_iframe_entries_count,
            r_iframe_desc_count,
            scales_count,
            s_iframe_entries_count,
            s_iframe_desc_count,
        })
    }

    /// Reads an `Animation` from an `Archive`.
    pub fn from_archive(archive: &mut Archive<impl Read>) -> Result<Animation, OzzError> {
        let meta = Animation::read_meta(archive)?;
        let mut animation = Animation::new(meta);

        archive.read_slice(animation.timepoints_mut())?;
        let is_ratio_u8 = animation.timepoints().len() <= (u8::MAX as usize);

        if is_ratio_u8 {
            for i in 0..animation.t_ratios().len() {
                animation.t_ratios_mut()[i] = archive.read::<u8>()? as u16;
            }
        } else {
            archive.read_slice(animation.t_ratios_mut())?;
        }
        archive.read_slice(animation.t_previouses_mut())?;
        archive.read_slice(animation.t_iframe_entries_mut())?;
        archive.read_slice(animation.t_iframe_desc_mut())?;
        animation.t_iframe_interval = archive.read()?;
        archive.read_slice(animation.translations_mut())?;

        if is_ratio_u8 {
            for i in 0..animation.r_ratios().len() {
                animation.r_ratios_mut()[i] = archive.read::<u8>()? as u16;
            }
        } else {
            archive.read_slice(animation.r_ratios_mut())?;
        }
        archive.read_slice(animation.r_previouses_mut())?;
        archive.read_slice(animation.r_iframe_entries_mut())?;
        archive.read_slice(animation.r_iframe_desc_mut())?;
        animation.r_iframe_interval = archive.read()?;
        archive.read_slice(animation.rotations_mut())?;

        if is_ratio_u8 {
            for i in 0..animation.s_ratios().len() {
                animation.s_ratios_mut()[i] = archive.read::<u8>()? as u16;
            }
        } else {
            archive.read_slice(animation.s_ratios_mut())?;
        }
        archive.read_slice(animation.s_previouses_mut())?;
        archive.read_slice(animation.s_iframe_entries_mut())?;
        archive.read_slice(animation.s_iframe_desc_mut())?;
        animation.s_iframe_interval = archive.read()?;
        archive.read_slice(animation.scales_mut())?;
        Ok(animation)
    }

    /// Reads an `Animation` from a file path.
    #[cfg(not(feature = "wasm"))]
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Animation, OzzError> {
        let mut archive = Archive::from_path(path)?;
        Animation::from_archive(&mut archive)
    }

    /// Reads an `Animation` from a file path.
    #[cfg(all(feature = "wasm", feature = "nodejs"))]
    pub fn from_path(path: &str) -> Result<Animation, OzzError> {
        let mut archive = Archive::from_path(path)?;
        Animation::from_archive(&mut archive)
    }

    pub(crate) fn from_raw(raw: &AnimationRaw) -> Animation {
        let meta = AnimationMeta {
            version: Animation::version(),
            duration: raw.duration,
            num_tracks: raw.num_tracks,
            name: raw.name.clone(),
            timepoints_count: raw.timepoints.len() as u32,
            translations_count: raw.translations.len() as u32,
            t_iframe_entries_count: raw.t_iframe_entries.len() as u32,
            t_iframe_desc_count: raw.t_iframe_desc.len() as u32,
            rotations_count: raw.rotations.len() as u32,
            r_iframe_entries_count: raw.r_iframe_entries.len() as u32,
            r_iframe_desc_count: raw.r_iframe_desc.len() as u32,
            scales_count: raw.scales.len() as u32,
            s_iframe_entries_count: raw.s_iframe_entries.len() as u32,
            s_iframe_desc_count: raw.s_iframe_desc.len() as u32,
        };
        let mut animation = Animation::new(meta);
        animation.timepoints_mut().copy_from_slice(&raw.timepoints);

        animation.translations_mut().copy_from_slice(&raw.translations);
        animation.t_ratios_mut().copy_from_slice(&raw.t_ratios);
        animation.t_previouses_mut().copy_from_slice(&raw.t_previouses);
        animation.t_iframe_interval = raw.t_iframe_interval;
        animation.t_iframe_entries_mut().copy_from_slice(&raw.t_iframe_entries);
        animation.t_iframe_desc_mut().copy_from_slice(&raw.t_iframe_desc);

        animation.rotations_mut().copy_from_slice(&raw.rotations);
        animation.r_ratios_mut().copy_from_slice(&raw.r_ratios);
        animation.r_previouses_mut().copy_from_slice(&raw.r_previouses);
        animation.r_iframe_interval = raw.r_iframe_interval;
        animation.r_iframe_entries_mut().copy_from_slice(&raw.r_iframe_entries);
        animation.r_iframe_desc_mut().copy_from_slice(&raw.r_iframe_desc);

        animation.scales_mut().copy_from_slice(&raw.scales);
        animation.s_ratios_mut().copy_from_slice(&raw.s_ratios);
        animation.s_previouses_mut().copy_from_slice(&raw.s_previouses);
        animation.s_iframe_interval = raw.s_iframe_interval;
        animation.s_iframe_entries_mut().copy_from_slice(&raw.s_iframe_entries);
        animation.s_iframe_desc_mut().copy_from_slice(&raw.s_iframe_desc);
        animation
    }

    pub(crate) fn to_raw(&self) -> AnimationRaw {
        AnimationRaw {
            duration: self.duration,
            num_tracks: self.num_tracks,
            name: self.name.clone(),
            timepoints: self.timepoints().to_vec(),

            translations: self.translations().to_vec(),
            t_ratios: self.t_ratios().to_vec(),
            t_previouses: self.t_previouses().to_vec(),
            t_iframe_interval: self.t_iframe_interval,
            t_iframe_entries: self.t_iframe_entries().to_vec(),
            t_iframe_desc: self.t_iframe_desc().to_vec(),

            rotations: self.rotations().to_vec(),
            r_ratios: self.r_ratios().to_vec(),
            r_previouses: self.r_previouses().to_vec(),
            r_iframe_interval: self.r_iframe_interval,
            r_iframe_entries: self.r_iframe_entries().to_vec(),
            r_iframe_desc: self.r_iframe_desc().to_vec(),

            scales: self.scales().to_vec(),
            s_ratios: self.s_ratios().to_vec(),
            s_previouses: self.s_previouses().to_vec(),
            s_iframe_interval: self.s_iframe_interval,
            s_iframe_entries: self.s_iframe_entries().to_vec(),
            s_iframe_desc: self.s_iframe_desc().to_vec(),
        }
    }

    fn new(meta: AnimationMeta) -> Animation {
        let mut animation = Animation {
            size: 0,
            duration: meta.duration,
            num_tracks: meta.num_tracks,
            name: meta.name,
            timepoints: std::ptr::null_mut(),
            timepoints_count: meta.timepoints_count,

            translations: std::ptr::null_mut(),
            translations_count: meta.translations_count,
            t_ratios: std::ptr::null_mut(),
            t_previouses: std::ptr::null_mut(),
            t_iframe_interval: 0.0,
            t_iframe_entries: std::ptr::null_mut(),
            t_iframe_entries_count: meta.t_iframe_entries_count,
            t_iframe_desc: std::ptr::null_mut(),
            t_iframe_desc_count: meta.t_iframe_desc_count,

            rotations: std::ptr::null_mut(),
            rotations_count: meta.rotations_count,
            r_ratios: std::ptr::null_mut(),
            r_previouses: std::ptr::null_mut(),
            r_iframe_interval: 0.0,
            r_iframe_entries: std::ptr::null_mut(),
            r_iframe_entries_count: meta.r_iframe_entries_count,
            r_iframe_desc: std::ptr::null_mut(),
            r_iframe_desc_count: meta.r_iframe_desc_count,

            scales: std::ptr::null_mut(),
            scales_count: meta.scales_count,
            s_ratios: std::ptr::null_mut(),
            s_previouses: std::ptr::null_mut(),
            s_iframe_interval: 0.0,
            s_iframe_entries: std::ptr::null_mut(),
            s_iframe_entries_count: meta.s_iframe_entries_count,
            s_iframe_desc: std::ptr::null_mut(),
            s_iframe_desc_count: meta.s_iframe_desc_count,
        };

        const ALIGN: usize = mem::align_of::<f32>();
        animation.size = (animation.timepoints_count as usize) * mem::size_of::<f32>();

        animation.size += animation.translations_count as usize * mem::size_of::<Float3Key>() + // translations
            animation.translations_count as usize * mem::size_of::<u16>() + // ratios
            animation.translations_count as usize * mem::size_of::<u16>() + // previouses
            animation.t_iframe_entries_count as usize * mem::size_of::<u8>(); // t_iframe_entries
        animation.size = align_usize(animation.size, ALIGN);
        animation.size += animation.t_iframe_desc_count as usize * mem::size_of::<u32>(); // t_iframe_desc

        animation.size += animation.rotations_count as usize * mem::size_of::<QuaternionKey>() + // rotations
            animation.rotations_count as usize * mem::size_of::<u16>() + // r_ratios
            animation.rotations_count as usize * mem::size_of::<u16>() + // r_previouses
            animation.r_iframe_entries_count as usize * mem::size_of::<u8>(); // r_iframe_entries
        animation.size = align_usize(animation.size, ALIGN);
        animation.size += animation.r_iframe_desc_count as usize * mem::size_of::<u32>(); // r_iframe_desc

        animation.size += animation.scales_count as usize * mem::size_of::<Float3Key>() + // scales
            animation.scales_count as usize * mem::size_of::<u16>() + // s_ratios
            animation.scales_count as usize * mem::size_of::<u16>() + // s_previouses
            animation.s_iframe_entries_count as usize * mem::size_of::<u8>(); // s_iframe_entries
        animation.size = align_usize(animation.size, ALIGN);
        animation.size += animation.s_iframe_desc_count as usize * mem::size_of::<u32>(); // s_iframe_desc

        unsafe {
            let layout = Layout::from_size_align_unchecked(animation.size, mem::size_of::<f32>());
            let mut ptr = alloc::alloc(layout);

            animation.timepoints = ptr as *mut f32;
            ptr = ptr.add(animation.timepoints_count as usize * mem::size_of::<f32>());

            animation.translations = ptr as *mut Float3Key;
            ptr = ptr.add(animation.translations_count as usize * mem::size_of::<Float3Key>());
            animation.t_ratios = ptr as *mut u16;
            ptr = ptr.add(animation.translations_count as usize * mem::size_of::<u16>());
            animation.t_previouses = ptr as *mut u16;
            ptr = ptr.add(animation.translations_count as usize * mem::size_of::<u16>());
            animation.t_iframe_entries = ptr;
            ptr = ptr.add(animation.t_iframe_entries_count as usize);
            ptr = align_ptr(ptr, ALIGN);
            animation.t_iframe_desc = ptr as *mut u32;
            ptr = ptr.add(animation.t_iframe_desc_count as usize * mem::size_of::<u32>());

            animation.rotations = ptr as *mut QuaternionKey;
            ptr = ptr.add(animation.rotations_count as usize * mem::size_of::<QuaternionKey>());
            animation.r_ratios = ptr as *mut u16;
            ptr = ptr.add(animation.rotations_count as usize * mem::size_of::<u16>());
            animation.r_previouses = ptr as *mut u16;
            ptr = ptr.add(animation.rotations_count as usize * mem::size_of::<u16>());
            animation.r_iframe_entries = ptr;
            ptr = ptr.add(animation.r_iframe_entries_count as usize);
            ptr = align_ptr(ptr, ALIGN);
            animation.r_iframe_desc = ptr as *mut u32;
            ptr = ptr.add(animation.r_iframe_desc_count as usize * mem::size_of::<u32>());

            animation.scales = ptr as *mut Float3Key;
            ptr = ptr.add(animation.scales_count as usize * mem::size_of::<Float3Key>());
            animation.s_ratios = ptr as *mut u16;
            ptr = ptr.add(animation.scales_count as usize * mem::size_of::<u16>());
            animation.s_previouses = ptr as *mut u16;
            ptr = ptr.add(animation.scales_count as usize * mem::size_of::<u16>());
            animation.s_iframe_entries = ptr;
            ptr = ptr.add(animation.s_iframe_entries_count as usize);
            ptr = align_ptr(ptr, ALIGN);
            animation.s_iframe_desc = ptr as *mut u32;
            ptr = ptr.add(animation.s_iframe_desc_count as usize * mem::size_of::<u32>());

            assert_eq!(ptr, (animation.timepoints as *mut u8).add(animation.size));
        }
        animation
    }
}

/// Animation keyframes control structure.
#[derive(Debug, Default)]
pub struct KeyframesCtrl<'t> {
    pub ratios: &'t [u16],
    pub previouses: &'t [u16],
    pub iframe_entries: &'t [u8],
    pub iframe_desc: &'t [u32],
    pub iframe_interval: f32,
}

impl Animation {
    /// Gets the animation clip duration.
    #[inline]
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// Gets the number of animated tracks.
    #[inline]
    pub fn num_tracks(&self) -> usize {
        self.num_tracks as usize
    }

    /// Gets the number of animated tracks (aligned to 4 * SoA).
    #[inline]
    pub fn num_aligned_tracks(&self) -> usize {
        ((self.num_tracks as usize) + 3) & !0x3
    }

    /// Gets the number of SoA elements matching the number of tracks of `Animation`.
    /// This value is useful to allocate SoA runtime data structures.
    #[inline]
    pub fn num_soa_tracks(&self) -> usize {
        (self.num_tracks as usize).div_ceil(4)
    }

    /// Gets animation name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Gets the buffer of time points.
    #[inline]
    pub fn timepoints(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.timepoints, self.timepoints_count as usize) }
    }

    #[inline]
    fn timepoints_mut(&mut self) -> &mut [f32] {
        unsafe { slice::from_raw_parts_mut(self.timepoints, self.timepoints_count as usize) }
    }

    /// Gets the buffer of translation keys.
    #[inline]
    pub fn translations(&self) -> &[Float3Key] {
        unsafe { slice::from_raw_parts(self.translations, self.translations_count as usize) }
    }

    #[inline]
    fn translations_mut(&mut self) -> &mut [Float3Key] {
        unsafe { slice::from_raw_parts_mut(self.translations, self.translations_count as usize) }
    }

    #[inline]
    fn t_ratios(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.t_ratios, self.translations_count as usize) }
    }

    #[inline]
    fn t_ratios_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.t_ratios, self.translations_count as usize) }
    }

    #[inline]
    fn t_previouses(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.t_previouses, self.translations_count as usize) }
    }

    #[inline]
    fn t_previouses_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.t_previouses, self.translations_count as usize) }
    }

    #[inline]
    fn t_iframe_entries(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.t_iframe_entries, self.t_iframe_entries_count as usize) }
    }

    #[inline]
    fn t_iframe_entries_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.t_iframe_entries, self.t_iframe_entries_count as usize) }
    }

    #[inline]
    fn t_iframe_desc(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.t_iframe_desc, self.t_iframe_desc_count as usize) }
    }

    #[inline]
    fn t_iframe_desc_mut(&mut self) -> &mut [u32] {
        unsafe { slice::from_raw_parts_mut(self.t_iframe_desc, self.t_iframe_desc_count as usize) }
    }

    /// Gets the buffer of rotation keys.
    #[inline]
    pub fn rotations(&self) -> &[QuaternionKey] {
        unsafe { slice::from_raw_parts(self.rotations, self.rotations_count as usize) }
    }

    #[inline]
    fn rotations_mut(&mut self) -> &mut [QuaternionKey] {
        unsafe { slice::from_raw_parts_mut(self.rotations, self.rotations_count as usize) }
    }

    #[inline]
    fn r_ratios(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.r_ratios, self.rotations_count as usize) }
    }

    #[inline]
    fn r_ratios_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.r_ratios, self.rotations_count as usize) }
    }

    #[inline]
    fn r_previouses(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.r_previouses, self.rotations_count as usize) }
    }

    #[inline]
    fn r_previouses_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.r_previouses, self.rotations_count as usize) }
    }

    #[inline]
    fn r_iframe_entries(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.r_iframe_entries, self.r_iframe_entries_count as usize) }
    }

    #[inline]
    fn r_iframe_entries_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.r_iframe_entries, self.r_iframe_entries_count as usize) }
    }

    #[inline]
    fn r_iframe_desc(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.r_iframe_desc, self.r_iframe_desc_count as usize) }
    }

    #[inline]
    fn r_iframe_desc_mut(&mut self) -> &mut [u32] {
        unsafe { slice::from_raw_parts_mut(self.r_iframe_desc, self.r_iframe_desc_count as usize) }
    }

    /// Gets the buffer of scale keys.
    #[inline]
    pub fn scales(&self) -> &[Float3Key] {
        unsafe { slice::from_raw_parts(self.scales, self.scales_count as usize) }
    }

    #[inline]
    fn scales_mut(&mut self) -> &mut [Float3Key] {
        unsafe { slice::from_raw_parts_mut(self.scales, self.scales_count as usize) }
    }

    #[inline]
    fn s_ratios(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.s_ratios, self.scales_count as usize) }
    }

    #[inline]
    fn s_ratios_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.s_ratios, self.scales_count as usize) }
    }

    #[inline]
    fn s_previouses(&self) -> &[u16] {
        unsafe { slice::from_raw_parts(self.s_previouses, self.scales_count as usize) }
    }

    #[inline]
    fn s_previouses_mut(&mut self) -> &mut [u16] {
        unsafe { slice::from_raw_parts_mut(self.s_previouses, self.scales_count as usize) }
    }

    #[inline]
    fn s_iframe_entries(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.s_iframe_entries, self.s_iframe_entries_count as usize) }
    }

    #[inline]
    fn s_iframe_entries_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.s_iframe_entries, self.s_iframe_entries_count as usize) }
    }

    #[inline]
    fn s_iframe_desc(&self) -> &[u32] {
        unsafe { slice::from_raw_parts(self.s_iframe_desc, self.s_iframe_desc_count as usize) }
    }

    #[inline]
    fn s_iframe_desc_mut(&mut self) -> &mut [u32] {
        unsafe { slice::from_raw_parts_mut(self.s_iframe_desc, self.s_iframe_desc_count as usize) }
    }

    /// Gets the buffer of translation keys.
    #[inline]
    pub fn translations_ctrl(&self) -> KeyframesCtrl<'_> {
        unsafe {
            KeyframesCtrl {
                ratios: slice::from_raw_parts(self.t_ratios, self.translations_count as usize),
                previouses: slice::from_raw_parts(self.t_previouses, self.translations_count as usize),
                iframe_entries: slice::from_raw_parts(self.t_iframe_entries, self.t_iframe_entries_count as usize),
                iframe_desc: slice::from_raw_parts(self.t_iframe_desc, self.t_iframe_desc_count as usize),
                iframe_interval: self.t_iframe_interval,
            }
        }
    }

    /// Gets the buffer of rotation keys.
    #[inline]
    pub fn rotations_ctrl(&self) -> KeyframesCtrl<'_> {
        unsafe {
            KeyframesCtrl {
                ratios: slice::from_raw_parts(self.r_ratios, self.rotations_count as usize),
                previouses: slice::from_raw_parts(self.r_previouses, self.rotations_count as usize),
                iframe_entries: slice::from_raw_parts(self.r_iframe_entries, self.r_iframe_entries_count as usize),
                iframe_desc: slice::from_raw_parts(self.r_iframe_desc, self.r_iframe_desc_count as usize),
                iframe_interval: self.r_iframe_interval,
            }
        }
    }

    /// Gets the buffer of scale keys.
    #[inline]
    pub fn scales_ctrl(&self) -> KeyframesCtrl<'_> {
        unsafe {
            KeyframesCtrl {
                ratios: slice::from_raw_parts(self.s_ratios, self.scales_count as usize),
                previouses: slice::from_raw_parts(self.s_previouses, self.scales_count as usize),
                iframe_entries: slice::from_raw_parts(self.s_iframe_entries, self.s_iframe_entries_count as usize),
                iframe_desc: slice::from_raw_parts(self.s_iframe_desc, self.s_iframe_desc_count as usize),
                iframe_interval: self.s_iframe_interval,
            }
        }
    }
}

#[cfg(feature = "rkyv")]
pub struct ArchivedAnimation {
    pub duration: f32,
    pub num_tracks: u32,
    pub name: rkyv::string::ArchivedString,
    pub timepoints: rkyv::vec::ArchivedVec<f32>,

    pub translations: rkyv::vec::ArchivedVec<ArchivedFloat3Key>,
    pub t_ratios: rkyv::vec::ArchivedVec<u16>,
    pub t_previouses: rkyv::vec::ArchivedVec<u16>,
    pub t_iframe_interval: f32,
    pub t_iframe_entries: rkyv::vec::ArchivedVec<u8>,
    pub t_iframe_desc: rkyv::vec::ArchivedVec<u32>,

    pub rotations: rkyv::vec::ArchivedVec<ArchivedQuaternionKey>,
    pub r_ratios: rkyv::vec::ArchivedVec<u16>,
    pub r_previouses: rkyv::vec::ArchivedVec<u16>,
    pub r_iframe_interval: f32,
    pub r_iframe_entries: rkyv::vec::ArchivedVec<u8>,
    pub r_iframe_desc: rkyv::vec::ArchivedVec<u32>,

    pub scales: rkyv::vec::ArchivedVec<ArchivedFloat3Key>,
    pub s_ratios: rkyv::vec::ArchivedVec<u16>,
    pub s_previouses: rkyv::vec::ArchivedVec<u16>,
    pub s_iframe_interval: f32,
    pub s_iframe_entries: rkyv::vec::ArchivedVec<u8>,
    pub s_iframe_desc: rkyv::vec::ArchivedVec<u32>,
}

#[cfg(feature = "rkyv")]
const _: () = {
    use rkyv::ser::{ScratchSpace, Serializer};
    use rkyv::string::{ArchivedString, StringResolver};
    use rkyv::vec::{ArchivedVec, VecResolver};
    use rkyv::{from_archived, out_field, Archive, Deserialize, Fallible, Serialize};

    pub struct AnimationResolver {
        name: StringResolver,
        timepoints: VecResolver,

        translations: VecResolver,
        t_ratios: VecResolver,
        t_previouses: VecResolver,
        t_iframe_entries: VecResolver,
        t_iframe_desc: VecResolver,

        rotations: VecResolver,
        r_ratios: VecResolver,
        r_previouses: VecResolver,
        r_iframe_entries: VecResolver,
        r_iframe_desc: VecResolver,

        scales: VecResolver,
        s_ratios: VecResolver,
        s_previouses: VecResolver,
        s_iframe_entries: VecResolver,
        s_iframe_desc: VecResolver,
    }

    impl Archive for Animation {
        type Archived = ArchivedAnimation;
        type Resolver = AnimationResolver;

        unsafe fn resolve(&self, pos: usize, resolver: AnimationResolver, out: *mut ArchivedAnimation) {
            let (fp, fo) = out_field!(out.duration);
            f32::resolve(&self.duration, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.num_tracks);
            u32::resolve(&self.num_tracks, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.name);
            String::resolve(&self.name, pos + fp, resolver.name, fo);
            let (fp, fo) = out_field!(out.timepoints);
            ArchivedVec::resolve_from_slice(self.timepoints(), pos + fp, resolver.timepoints, fo);

            let (fp, fo) = out_field!(out.translations);
            ArchivedVec::resolve_from_slice(self.translations(), pos + fp, resolver.translations, fo);
            let (fp, fo) = out_field!(out.t_ratios);
            ArchivedVec::resolve_from_slice(self.t_ratios(), pos + fp, resolver.t_ratios, fo);
            let (fp, fo) = out_field!(out.t_previouses);
            ArchivedVec::resolve_from_slice(self.t_previouses(), pos + fp, resolver.t_previouses, fo);
            let (fp, fo) = out_field!(out.t_iframe_interval);
            f32::resolve(&self.t_iframe_interval, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.t_iframe_entries);
            ArchivedVec::resolve_from_slice(self.t_iframe_entries(), pos + fp, resolver.t_iframe_entries, fo);
            let (fp, fo) = out_field!(out.t_iframe_desc);
            ArchivedVec::resolve_from_slice(self.t_iframe_desc(), pos + fp, resolver.t_iframe_desc, fo);

            let (fp, fo) = out_field!(out.rotations);
            ArchivedVec::resolve_from_slice(self.rotations(), pos + fp, resolver.rotations, fo);
            let (fp, fo) = out_field!(out.r_ratios);
            ArchivedVec::resolve_from_slice(self.r_ratios(), pos + fp, resolver.r_ratios, fo);
            let (fp, fo) = out_field!(out.r_previouses);
            ArchivedVec::resolve_from_slice(self.r_previouses(), pos + fp, resolver.r_previouses, fo);
            let (fp, fo) = out_field!(out.r_iframe_interval);
            f32::resolve(&self.r_iframe_interval, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.r_iframe_entries);
            ArchivedVec::resolve_from_slice(self.r_iframe_entries(), pos + fp, resolver.r_iframe_entries, fo);
            let (fp, fo) = out_field!(out.r_iframe_desc);
            ArchivedVec::resolve_from_slice(self.r_iframe_desc(), pos + fp, resolver.r_iframe_desc, fo);

            let (fp, fo) = out_field!(out.scales);
            ArchivedVec::resolve_from_slice(self.scales(), pos + fp, resolver.scales, fo);
            let (fp, fo) = out_field!(out.s_ratios);
            ArchivedVec::resolve_from_slice(self.s_ratios(), pos + fp, resolver.s_ratios, fo);
            let (fp, fo) = out_field!(out.s_previouses);
            ArchivedVec::resolve_from_slice(self.s_previouses(), pos + fp, resolver.s_previouses, fo);
            let (fp, fo) = out_field!(out.s_iframe_interval);
            f32::resolve(&self.s_iframe_interval, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.s_iframe_entries);
            ArchivedVec::resolve_from_slice(self.s_iframe_entries(), pos + fp, resolver.s_iframe_entries, fo);
            let (fp, fo) = out_field!(out.s_iframe_desc);
            ArchivedVec::resolve_from_slice(self.s_iframe_desc(), pos + fp, resolver.s_iframe_desc, fo);
        }
    }

    impl<S: Serializer + ScratchSpace + ?Sized> Serialize<S> for Animation {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            Ok(AnimationResolver {
                name: ArchivedString::serialize_from_str(&self.name, serializer)?,
                timepoints: ArchivedVec::serialize_from_slice(self.timepoints(), serializer)?,
                translations: ArchivedVec::serialize_from_slice(self.translations(), serializer)?,
                t_ratios: ArchivedVec::serialize_from_slice(self.t_ratios(), serializer)?,
                t_previouses: ArchivedVec::serialize_from_slice(self.t_previouses(), serializer)?,
                t_iframe_entries: ArchivedVec::serialize_from_slice(self.t_iframe_entries(), serializer)?,
                t_iframe_desc: ArchivedVec::serialize_from_slice(self.t_iframe_desc(), serializer)?,
                rotations: ArchivedVec::serialize_from_slice(self.rotations(), serializer)?,
                r_ratios: ArchivedVec::serialize_from_slice(self.r_ratios(), serializer)?,
                r_previouses: ArchivedVec::serialize_from_slice(self.r_previouses(), serializer)?,
                r_iframe_entries: ArchivedVec::serialize_from_slice(self.r_iframe_entries(), serializer)?,
                r_iframe_desc: ArchivedVec::serialize_from_slice(self.r_iframe_desc(), serializer)?,
                scales: ArchivedVec::serialize_from_slice(self.scales(), serializer)?,
                s_ratios: ArchivedVec::serialize_from_slice(self.s_ratios(), serializer)?,
                s_previouses: ArchivedVec::serialize_from_slice(self.s_previouses(), serializer)?,
                s_iframe_entries: ArchivedVec::serialize_from_slice(self.s_iframe_entries(), serializer)?,
                s_iframe_desc: ArchivedVec::serialize_from_slice(self.s_iframe_desc(), serializer)?,
            })
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<Animation, D> for ArchivedAnimation {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<Animation, D::Error> {
            let archived = from_archived!(self);
            let mut animation = Animation::new(AnimationMeta {
                version: Animation::version(),
                duration: archived.duration,
                num_tracks: archived.num_tracks,
                name: archived.name.to_string(),
                timepoints_count: archived.timepoints.len() as u32,
                translations_count: archived.translations.len() as u32,
                t_iframe_entries_count: archived.t_iframe_entries.len() as u32,
                t_iframe_desc_count: archived.t_iframe_desc.len() as u32,
                rotations_count: archived.rotations.len() as u32,
                r_iframe_entries_count: archived.r_iframe_entries.len() as u32,
                r_iframe_desc_count: archived.r_iframe_desc.len() as u32,
                scales_count: archived.scales.len() as u32,
                s_iframe_entries_count: archived.s_iframe_entries.len() as u32,
                s_iframe_desc_count: archived.s_iframe_desc.len() as u32,
            });

            animation
                .timepoints_mut()
                .copy_from_slice(archived.timepoints.as_slice());

            for (idx, t) in archived.translations.iter().enumerate() {
                animation.translations_mut()[idx] = Float3Key(t.0);
            }
            animation.t_ratios_mut().copy_from_slice(archived.t_ratios.as_slice());
            animation
                .t_previouses_mut()
                .copy_from_slice(archived.t_previouses.as_slice());
            animation
                .t_iframe_entries_mut()
                .copy_from_slice(archived.t_iframe_entries.as_slice());
            animation
                .t_iframe_desc_mut()
                .copy_from_slice(archived.t_iframe_desc.as_slice());
            animation.t_iframe_interval = archived.t_iframe_interval;

            for (idx, r) in archived.rotations.iter().enumerate() {
                animation.rotations_mut()[idx] = QuaternionKey(r.0);
            }
            animation.r_ratios_mut().copy_from_slice(archived.r_ratios.as_slice());
            animation
                .r_previouses_mut()
                .copy_from_slice(archived.r_previouses.as_slice());
            animation
                .r_iframe_entries_mut()
                .copy_from_slice(archived.r_iframe_entries.as_slice());
            animation
                .r_iframe_desc_mut()
                .copy_from_slice(archived.r_iframe_desc.as_slice());
            animation.r_iframe_interval = archived.r_iframe_interval;

            for (idx, s) in archived.scales.iter().enumerate() {
                animation.scales_mut()[idx] = Float3Key(s.0);
            }
            animation.s_ratios_mut().copy_from_slice(archived.s_ratios.as_slice());
            animation
                .s_previouses_mut()
                .copy_from_slice(archived.s_previouses.as_slice());
            animation
                .s_iframe_entries_mut()
                .copy_from_slice(archived.s_iframe_entries.as_slice());
            animation
                .s_iframe_desc_mut()
                .copy_from_slice(archived.s_iframe_desc.as_slice());
            animation.s_iframe_interval = archived.s_iframe_interval;

            Ok(animation)
        }
    }
};

#[cfg(feature = "serde")]
const _: () = {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Animation {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw = self.to_raw();
            raw.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Animation {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Animation, D::Error> {
            let raw = AnimationRaw::deserialize(deserializer)?;
            Ok(Animation::from_raw(&raw))
        }
    }
};

#[allow(clippy::excessive_precision)]
#[cfg(test)]
mod tests {
    use wasm_bindgen_test::*;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_float3_key_decompress() {
        let res = Float3Key([11405, 34240, 31]).decompress();
        assert_eq!(res, Vec3::new(0.0711059570, -8.77380371e-05, 1.84774399e-06));

        let res = Float3Key([9839, 1, 0]).decompress();
        assert_eq!(res, Vec3::new(0.0251312255859375, 5.960464477539063e-8, 0.0));
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_simd_decompress_float3() {
        let k0 = Float3Key([11405, 34240, 31]);
        let k1 = Float3Key([9839, 1, 0]);
        let k2 = Float3Key([11405, 34240, 31]);
        let k3 = Float3Key([9839, 1, 0]);
        let mut soa = SoaVec3::default();
        Float3Key::simd_decompress(&k0, &k1, &k2, &k3, &mut soa);
        assert_eq!(
            soa,
            SoaVec3 {
                x: f32x4::from_array([0.0711059570, 0.0251312255859375, 0.0711059570, 0.0251312255859375]),
                y: f32x4::from_array([
                    -8.77380371e-05,
                    5.960464477539063e-8,
                    -8.77380371e-05,
                    5.960464477539063e-8
                ]),
                z: f32x4::from_array([1.84774399e-06, 0.0, 1.84774399e-06, 0.0]),
            }
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_decompress_quaternion() {
        let key = QuaternionKey([39974, 18396, 53990]);
        let quat = key.decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(-0.491480947, -0.508615375, -0.538519204, 0.457989037)
        );

        let key = QuaternionKey([38605, 19300, 55990]);
        let quat = key.decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(-0.498861253, -0.501123607, -0.498861253, 0.501148760)
        );

        let key = QuaternionKey([63843, 2329, 31255]);
        let quat = key.decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(-0.00912827253, 0.0251405239, -0.0326502919, 0.999108911)
        );

        let key = QuaternionKey([1579, 818, 33051]);
        let quat = key.decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(0.00852406025, 0.00882613659, 0.00610709190, 0.999906063)
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_simd_decompress_quaternion() {
        let key0 = QuaternionKey([39974, 18396, 53990]);
        let key1 = QuaternionKey([38605, 19300, 55990]);
        let key2 = QuaternionKey([63843, 2329, 31255]);
        let key3 = QuaternionKey([1579, 818, 33051]);
        let mut soa = SoaQuat::default();
        QuaternionKey::simd_decompress(&key0, &key1, &key2, &key3, &mut soa);
        assert_eq!(
            soa,
            SoaQuat {
                x: f32x4::from_array([-0.491480947, -0.498861253, -0.00912827253, 0.00852406025]),
                y: f32x4::from_array([-0.508615375, -0.501123607, 0.0251405239, 0.00882613659]),
                z: f32x4::from_array([-0.538519204, -0.498861253, -0.0326502919, 0.00610709190]),
                w: f32x4::from_array([0.457989037, 0.501148760, 0.999108911, 0.999906063]),
            }
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_read_animation() {
        let animation = Animation::from_path("./resource/playback/animation.ozz").unwrap();

        assert_eq!(animation.duration(), 8.60000038);
        assert_eq!(animation.num_tracks(), 67);
        assert_eq!(animation.name(), "crossarms".to_string());

        assert_eq!(animation.timepoints().len(), 252);
        assert_eq!(animation.timepoints().first().unwrap(), &0.0);
        assert_eq!(animation.timepoints().last().unwrap(), &1.0);

        assert_eq!(animation.translations_ctrl().ratios.len(), 178);
        assert_eq!(animation.translations_ctrl().ratios.first().unwrap(), &0);
        assert_eq!(animation.translations_ctrl().ratios.last().unwrap(), &251);
        assert_eq!(animation.translations_ctrl().previouses.len(), 178);
        assert_eq!(animation.translations_ctrl().previouses.first().unwrap(), &0);
        assert_eq!(animation.translations_ctrl().previouses.last().unwrap(), &1);
        assert_eq!(animation.translations_ctrl().iframe_interval, 1.0);
        assert_eq!(animation.translations_ctrl().iframe_entries.len(), 85);
        assert_eq!(animation.translations_ctrl().iframe_desc.len(), 2);

        assert_eq!(animation.rotations_ctrl().ratios.len(), 1748);
        assert_eq!(animation.rotations_ctrl().ratios.first().unwrap(), &0);
        assert_eq!(animation.rotations_ctrl().ratios.last().unwrap(), &251);
        assert_eq!(animation.rotations_ctrl().previouses.len(), 1748);
        assert_eq!(animation.rotations_ctrl().previouses.first().unwrap(), &0);
        assert_eq!(animation.rotations_ctrl().previouses.last().unwrap(), &6);
        assert_eq!(animation.rotations_ctrl().iframe_interval, 1.0);
        assert_eq!(animation.rotations_ctrl().iframe_entries.len(), 137);
        assert_eq!(animation.rotations_ctrl().iframe_desc.len(), 2);

        assert_eq!(animation.scales_ctrl().ratios.len(), 136);
        assert_eq!(animation.scales_ctrl().ratios.first().unwrap(), &0);
        assert_eq!(animation.scales_ctrl().ratios.last().unwrap(), &251);
        assert_eq!(animation.scales_ctrl().previouses.len(), 136);
        assert_eq!(animation.scales_ctrl().previouses.first().unwrap(), &0);
        assert_eq!(animation.scales_ctrl().previouses.last().unwrap(), &68);
        assert_eq!(animation.scales_ctrl().iframe_interval, 1.0);
        assert!(animation.scales_ctrl().iframe_entries.is_empty());
        assert!(animation.scales_ctrl().iframe_desc.is_empty());

        assert_eq!(animation.translations().len(), 178);
        assert_eq!(animation.translations().first().unwrap().0, [0, 15400, 43950]);
        assert_eq!(animation.translations().last().unwrap().0, [3659, 15400, 43933]);

        assert_eq!(animation.rotations().len(), 1748);
        assert_eq!(animation.rotations().first().unwrap().0, [39974, 18396, 53990]);
        assert_eq!(animation.rotations().last().unwrap().0, [63955, 2225, 31299]);

        assert_eq!(animation.scales().len(), 136);
        assert_eq!(animation.scales().first().unwrap().0, [15360, 15360, 15360]);
        assert_eq!(animation.scales().last().unwrap().0, [15360, 15360, 15360]);
    }

    #[cfg(feature = "rkyv")]
    #[test]
    #[wasm_bindgen_test]
    fn test_rkyv_animation() {
        use rkyv::ser::Serializer;
        use rkyv::Deserialize;

        let animation = Animation::from_path("./resource/playback/animation.ozz").unwrap();
        let mut serializer = rkyv::ser::serializers::AllocSerializer::<30720>::default();
        serializer.serialize_value(&animation).unwrap();
        let buf = serializer.into_serializer().into_inner();
        let archived = unsafe { rkyv::archived_root::<Animation>(&buf) };
        let mut deserializer = rkyv::Infallible;
        let animation2: Animation = archived.deserialize(&mut deserializer).unwrap();

        assert_eq!(animation.duration(), animation2.duration());
        assert_eq!(animation.num_tracks(), animation2.num_tracks());
        assert_eq!(animation.name(), animation2.name());
        assert_eq!(animation.timepoints(), animation2.timepoints());

        assert_eq!(
            animation.translations_ctrl().ratios,
            animation2.translations_ctrl().ratios
        );
        assert_eq!(
            animation.translations_ctrl().previouses,
            animation2.translations_ctrl().previouses
        );
        assert_eq!(
            animation.translations_ctrl().iframe_interval,
            animation2.translations_ctrl().iframe_interval
        );
        assert_eq!(
            animation.translations_ctrl().iframe_entries,
            animation2.translations_ctrl().iframe_entries
        );
        assert_eq!(
            animation.translations_ctrl().iframe_desc,
            animation2.translations_ctrl().iframe_desc
        );

        assert_eq!(animation.rotations_ctrl().ratios, animation2.rotations_ctrl().ratios);
        assert_eq!(
            animation.rotations_ctrl().previouses,
            animation2.rotations_ctrl().previouses
        );
        assert_eq!(
            animation.rotations_ctrl().iframe_interval,
            animation2.rotations_ctrl().iframe_interval
        );
        assert_eq!(
            animation.rotations_ctrl().iframe_entries,
            animation2.rotations_ctrl().iframe_entries
        );
        assert_eq!(
            animation.rotations_ctrl().iframe_desc,
            animation2.rotations_ctrl().iframe_desc
        );

        assert_eq!(animation.scales_ctrl().ratios, animation2.scales_ctrl().ratios);
        assert_eq!(animation.scales_ctrl().previouses, animation2.scales_ctrl().previouses);
        assert_eq!(
            animation.scales_ctrl().iframe_interval,
            animation2.scales_ctrl().iframe_interval
        );
        assert_eq!(
            animation.scales_ctrl().iframe_entries,
            animation2.scales_ctrl().iframe_entries
        );
        assert_eq!(
            animation.scales_ctrl().iframe_desc,
            animation2.scales_ctrl().iframe_desc
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    #[wasm_bindgen_test]
    fn test_serde_animation() {
        use serde_json;

        let animation = Animation::from_path("./resource/blend/animation1.ozz").unwrap();
        let josn = serde_json::to_vec(&animation).unwrap();
        let animation2: Animation = serde_json::from_slice(&josn).unwrap();

        assert_eq!(animation.duration(), animation2.duration());
        assert_eq!(animation.num_tracks(), animation2.num_tracks());
        assert_eq!(animation.name(), animation2.name());
        assert_eq!(animation.timepoints(), animation2.timepoints());

        assert_eq!(
            animation.translations_ctrl().ratios,
            animation2.translations_ctrl().ratios
        );
        assert_eq!(
            animation.translations_ctrl().previouses,
            animation2.translations_ctrl().previouses
        );
        assert_eq!(
            animation.translations_ctrl().iframe_interval,
            animation2.translations_ctrl().iframe_interval
        );
        assert_eq!(
            animation.translations_ctrl().iframe_entries,
            animation2.translations_ctrl().iframe_entries
        );
        assert_eq!(
            animation.translations_ctrl().iframe_desc,
            animation2.translations_ctrl().iframe_desc
        );

        assert_eq!(animation.rotations_ctrl().ratios, animation2.rotations_ctrl().ratios);
        assert_eq!(
            animation.rotations_ctrl().previouses,
            animation2.rotations_ctrl().previouses
        );
        assert_eq!(
            animation.rotations_ctrl().iframe_interval,
            animation2.rotations_ctrl().iframe_interval
        );
        assert_eq!(
            animation.rotations_ctrl().iframe_entries,
            animation2.rotations_ctrl().iframe_entries
        );
        assert_eq!(
            animation.rotations_ctrl().iframe_desc,
            animation2.rotations_ctrl().iframe_desc
        );

        assert_eq!(animation.scales_ctrl().ratios, animation2.scales_ctrl().ratios);
        assert_eq!(animation.scales_ctrl().previouses, animation2.scales_ctrl().previouses);
        assert_eq!(
            animation.scales_ctrl().iframe_interval,
            animation2.scales_ctrl().iframe_interval
        );
        assert_eq!(
            animation.scales_ctrl().iframe_entries,
            animation2.scales_ctrl().iframe_entries
        );
        assert_eq!(
            animation.scales_ctrl().iframe_desc,
            animation2.scales_ctrl().iframe_desc
        );
    }
}
