//!
//! Animation data structure definition.
//!

use glam::{Quat, Vec3, Vec4};
use std::io::Read;
use std::simd::prelude::*;
use std::simd::*;
use std::{mem, slice};

use crate::archive::{Archive, ArchiveRead};
use crate::base::OzzError;
use crate::math::{f16_to_f32, fx4, ix4, simd_f16_to_f32, SoaQuat, SoaVec3, ONE, ZERO};

/// Float3 key for `Animation` track.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Float3Key([u16; 3]);

impl Float3Key {
    pub const fn new(value: [u16; 3]) -> Float3Key {
        return Float3Key(value);
    }

    #[inline]
    pub fn decompress(&self) -> Vec3 {
        return Vec3::new(f16_to_f32(self.0[0]), f16_to_f32(self.0[1]), f16_to_f32(self.0[2]));
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
        return Ok(Float3Key(value));
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
        return QuaternionKey(value);
    }

    #[inline]
    fn unpack(&self) -> (u16, u16, [u32; 3]) {
        let packed: u32 = (self.0[0] as u32) >> 3 | (self.0[1] as u32) << 13 | (self.0[2] as u32) << 29;
        let bigest = self.0[0] & 0x3;
        let sign = (self.0[0] >> 2) & 0x1;
        let value = [packed & 0x7fff, (packed >> 15) & 0x7fff, (self.0[2] as u32) >> 1];
        return (bigest, sign, value);
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
        return Quat::from_vec4(cpnt);
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

        soa.x = unsafe { mem::transmute(cpnt[0]) };
        soa.y = unsafe { mem::transmute(cpnt[1]) };
        soa.z = unsafe { mem::transmute(cpnt[2]) };
        soa.w = unsafe { mem::transmute(cpnt[3]) };
    }
}

impl ArchiveRead<QuaternionKey> for QuaternionKey {
    #[inline]
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<QuaternionKey, OzzError> {
        let value: [u16; 3] = [archive.read()?, archive.read()?, archive.read()?];
        return Ok(QuaternionKey(value));
    }
}

/// Animation keyframes control structure.
#[derive(Debug, Default)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KeyframesCtrl {
    ratios: Vec<u8>,
    previouses: Vec<u16>,
    iframe_entries: Vec<u8>,
    iframe_desc: Vec<u32>,
    iframe_interval: f32,
}

impl KeyframesCtrl {
    pub fn new(
        ratios: Vec<u8>,
        previouses: Vec<u16>,
        iframe_entries: Vec<u8>,
        iframe_desc: Vec<u32>,
        iframe_interval: f32,
    ) -> KeyframesCtrl {
        return KeyframesCtrl {
            ratios,
            previouses,
            iframe_entries,
            iframe_desc,
            iframe_interval,
        };
    }

    #[inline]
    pub fn ratios_u8(&self) -> &[u8] {
        return &self.ratios;
    }

    #[inline]
    pub fn ratios_u16(&self) -> &[u16] {
        let data = self.ratios.as_ptr() as *const u16;
        let len = self.ratios.len() / 2;
        return unsafe { slice::from_raw_parts(data, len) };
    }

    #[inline]
    pub fn previouses(&self) -> &[u16] {
        return &self.previouses;
    }

    #[inline]
    pub fn iframe_entries(&self) -> &[u8] {
        return &self.iframe_entries;
    }

    #[inline]
    pub fn iframe_desc(&self) -> &[u32] {
        return &self.iframe_desc;
    }

    #[inline]
    pub fn iframe_interval(&self) -> f32 {
        return self.iframe_interval;
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
#[derive(Debug, Default)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Animation {
    duration: f32,
    num_tracks: usize,
    name: String,
    timepoints: Vec<f32>,
    translations_ctrl: KeyframesCtrl,
    rotations_ctrl: KeyframesCtrl,
    scales_ctrl: KeyframesCtrl,
    translations: Vec<Float3Key>,
    rotations: Vec<QuaternionKey>,
    scales: Vec<Float3Key>,
}

/// Animation meta in `Archive`.
#[derive(Debug, Clone)]
pub struct AnimationMeta {
    pub version: u32,
    pub duration: f32,
    pub num_tracks: u32,
    pub name: String,
    pub timepoints_count: u32,
    pub translation_count: u32,
    pub rotation_count: u32,
    pub scale_count: u32,
    pub t_iframe_entries_count: u32,
    pub t_iframe_desc_count: u32,
    pub r_iframe_entries_count: u32,
    pub r_iframe_desc_count: u32,
    pub s_iframe_entries_count: u32,
    pub s_iframe_desc_count: u32,
}

impl Animation {
    /// `Animation` resource file tag for `Archive`.
    #[inline]
    pub fn tag() -> &'static str {
        return "ozz-animation";
    }

    /// `Animation` resource file version for `Archive`.
    #[inline]
    pub fn version() -> u32 {
        return 7;
    }

    #[cfg(test)]
    pub(crate) fn from_raw(
        duration: f32,
        num_tracks: usize,
        name: String,
        timepoints: Vec<f32>,
        translations_ctrl: KeyframesCtrl,
        rotations_ctrl: KeyframesCtrl,
        scales_ctrl: KeyframesCtrl,
        translations: Vec<Float3Key>,
        rotations: Vec<QuaternionKey>,
        scales: Vec<Float3Key>,
    ) -> Animation {
        return Animation {
            duration,
            num_tracks,
            name,
            timepoints,
            translations_ctrl,
            rotations_ctrl,
            scales_ctrl,
            translations,
            rotations,
            scales,
        };
    }

    /// Reads an `AnimationMeta` from an `Archive`.
    pub fn read_meta(archive: &mut Archive<impl Read>) -> Result<AnimationMeta, OzzError> {
        if archive.tag() != Self::tag() {
            return Err(OzzError::InvalidTag);
        }
        if archive.version() != Self::version() {
            return Err(OzzError::InvalidVersion);
        }

        let duration: f32 = archive.read()?;
        let num_tracks: u32 = archive.read()?;
        let name_len: u32 = archive.read()?;
        let timepoints_count: u32 = archive.read()?;
        let translation_count: u32 = archive.read()?;
        let rotation_count: u32 = archive.read()?;
        let scale_count: u32 = archive.read()?;
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

        return Ok(AnimationMeta {
            version: archive.version(),
            duration,
            num_tracks,
            name,
            timepoints_count,
            translation_count,
            rotation_count,
            scale_count,
            t_iframe_entries_count,
            t_iframe_desc_count,
            r_iframe_entries_count,
            r_iframe_desc_count,
            s_iframe_entries_count,
            s_iframe_desc_count,
        });
    }

    /// Reads an `Animation` from an `Archive`.
    pub fn from_archive(archive: &mut Archive<impl Read>) -> Result<Animation, OzzError> {
        let meta = Animation::read_meta(archive)?;

        let timepoints: Vec<f32> = archive.read_vec(meta.timepoints_count as usize)?;
        let sizeof_ratio = if timepoints.len() <= u8::MAX as usize { 1 } else { 2 };
        let translations_ctrl = KeyframesCtrl {
            ratios: archive.read_vec((meta.translation_count * sizeof_ratio) as usize)?,
            previouses: archive.read_vec(meta.translation_count as usize)?,
            iframe_entries: archive.read_vec(meta.t_iframe_entries_count as usize)?,
            iframe_desc: archive.read_vec(meta.t_iframe_desc_count as usize)?,
            iframe_interval: archive.read()?,
        };
        let translations: Vec<Float3Key> = archive.read_vec(meta.translation_count as usize)?;
        let rotations_ctrl = KeyframesCtrl {
            ratios: archive.read_vec((meta.rotation_count * sizeof_ratio) as usize)?,
            previouses: archive.read_vec(meta.rotation_count as usize)?,
            iframe_entries: archive.read_vec(meta.r_iframe_entries_count as usize)?,
            iframe_desc: archive.read_vec(meta.r_iframe_desc_count as usize)?,
            iframe_interval: archive.read()?,
        };
        let rotations: Vec<QuaternionKey> = archive.read_vec(meta.rotation_count as usize)?;
        let scales_ctrl = KeyframesCtrl {
            ratios: archive.read_vec((meta.scale_count * sizeof_ratio) as usize)?,
            previouses: archive.read_vec(meta.scale_count as usize)?,
            iframe_entries: archive.read_vec(meta.s_iframe_entries_count as usize)?,
            iframe_desc: archive.read_vec(meta.s_iframe_desc_count as usize)?,
            iframe_interval: archive.read()?,
        };
        let scales: Vec<Float3Key> = archive.read_vec(meta.scale_count as usize)?;

        return Ok(Animation {
            duration: meta.duration,
            num_tracks: meta.num_tracks as usize,
            name: meta.name,
            timepoints,
            translations,
            rotations,
            scales,
            translations_ctrl,
            rotations_ctrl,
            scales_ctrl,
        });
    }

    /// Reads an `Animation` from a file path.
    #[cfg(not(feature = "wasm"))]
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Animation, OzzError> {
        let mut archive = Archive::from_path(path)?;
        return Animation::from_archive(&mut archive);
    }

    /// Reads an `Animation` from a file path.
    #[cfg(all(feature = "wasm", feature = "nodejs"))]
    pub fn from_path(path: &str) -> Result<Animation, OzzError> {
        let mut archive = Archive::from_path(path)?;
        return Animation::from_archive(&mut archive);
    }
}

impl Animation {
    /// Gets the animation clip duration.
    #[inline]
    pub fn duration(&self) -> f32 {
        return self.duration;
    }

    /// Gets the number of animated tracks.
    #[inline]
    pub fn num_tracks(&self) -> usize {
        return self.num_tracks;
    }

    /// Gets the number of animated tracks (aligned to 4 * SoA).
    #[inline]
    pub fn num_aligned_tracks(&self) -> usize {
        return (self.num_tracks + 3) & !0x3;
    }

    /// Gets the number of SoA elements matching the number of tracks of `Animation`.
    /// This value is useful to allocate SoA runtime data structures.
    #[inline]
    pub fn num_soa_tracks(&self) -> usize {
        return (self.num_tracks + 3) / 4;
    }

    /// Gets animation name.
    #[inline]
    pub fn name(&self) -> &str {
        return &self.name;
    }

    /// Gets the buffer of time points.
    #[inline]
    pub fn timepoints(&self) -> &[f32] {
        return &self.timepoints;
    }

    /// Gets the buffer of translation keys.
    #[inline]
    pub fn translations_ctrl(&self) -> &KeyframesCtrl {
        return &self.translations_ctrl;
    }

    /// Gets the buffer of rotation keys.
    #[inline]
    pub fn rotations_ctrl(&self) -> &KeyframesCtrl {
        return &self.rotations_ctrl;
    }

    /// Gets the buffer of scale keys.
    #[inline]
    pub fn scales_ctrl(&self) -> &KeyframesCtrl {
        return &self.scales_ctrl;
    }

    /// Gets the buffer of translation keys.
    #[inline]
    pub fn translations(&self) -> &[Float3Key] {
        return &self.translations;
    }

    /// Gets the buffer of rotation keys.
    #[inline]
    pub fn rotations(&self) -> &[QuaternionKey] {
        return &self.rotations;
    }

    /// Gets the buffer of scale keys.
    #[inline]
    pub fn scales(&self) -> &[Float3Key] {
        return &self.scales;
    }
}

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

        assert_eq!(animation.translations_ctrl().ratios_u8().len(), 178);
        assert_eq!(animation.translations_ctrl().ratios_u8().first().unwrap(), &0);
        assert_eq!(animation.translations_ctrl().ratios_u8().last().unwrap(), &251);
        assert_eq!(animation.translations_ctrl().previouses().len(), 178);
        assert_eq!(animation.translations_ctrl().previouses().first().unwrap(), &0);
        assert_eq!(animation.translations_ctrl().previouses().last().unwrap(), &1);
        assert!(animation.translations_ctrl().iframe_entries().is_empty());
        assert!(animation.translations_ctrl().iframe_desc().is_empty());
        assert_eq!(animation.translations_ctrl().iframe_interval, 1.0);

        assert_eq!(animation.rotations_ctrl().ratios_u8().len(), 1699);
        assert_eq!(animation.rotations_ctrl().ratios_u8().first().unwrap(), &0);
        assert_eq!(animation.rotations_ctrl().ratios_u8().last().unwrap(), &251);
        assert_eq!(animation.rotations_ctrl().previouses().len(), 1699);
        assert_eq!(animation.rotations_ctrl().previouses().first().unwrap(), &0);
        assert_eq!(animation.rotations_ctrl().previouses().last().unwrap(), &6);
        assert!(animation.rotations_ctrl().iframe_entries().is_empty());
        assert!(animation.rotations_ctrl().iframe_desc().is_empty());
        assert_eq!(animation.rotations_ctrl().iframe_interval, 1.0);

        assert_eq!(animation.scales_ctrl().ratios_u8().len(), 136);
        assert_eq!(animation.scales_ctrl().ratios_u8().first().unwrap(), &0);
        assert_eq!(animation.scales_ctrl().ratios_u8().last().unwrap(), &251);
        assert_eq!(animation.scales_ctrl().previouses().len(), 136);
        assert_eq!(animation.scales_ctrl().previouses().first().unwrap(), &0);
        assert_eq!(animation.scales_ctrl().previouses().last().unwrap(), &68);
        assert!(animation.scales_ctrl().iframe_entries().is_empty());
        assert!(animation.scales_ctrl().iframe_desc().is_empty());
        assert_eq!(animation.scales_ctrl().iframe_interval, 1.0);

        assert_eq!(animation.translations().len(), 178);
        assert_eq!(animation.translations().first().unwrap().0, [0, 15400, 43950]);
        assert_eq!(animation.translations().last().unwrap().0, [3659, 15400, 43933]);

        assert_eq!(animation.rotations().len(), 1699);
        assert_eq!(animation.rotations().first().unwrap().0, [39974, 18396, 53990]);
        assert_eq!(animation.rotations().last().unwrap().0, [65531, 65533, 30456]);

        assert_eq!(animation.scales().len(), 136);
        assert_eq!(animation.scales().first().unwrap().0, [15360, 15360, 15360]);
        assert_eq!(animation.scales().last().unwrap().0, [15360, 15360, 15360]);
    }
}
