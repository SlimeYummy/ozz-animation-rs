use glam::{Quat, Vec3, Vec4};
use std::io::Read;
use std::mem;
use std::simd::prelude::*;
use std::simd::*;

use crate::archive::{Archive, ArchiveRead};
use crate::base::OzzError;
use crate::math::{f16_to_f32, fx4, ix4, simd_f16_to_f32, SoaQuat, SoaVec3};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct Float3Key {
    pub ratio: f32,
    pub track: u16,
    pub value: [u16; 3],
}

impl Float3Key {
    pub fn new(ratio: f32, track: u16, value: [u16; 3]) -> Float3Key {
        return Float3Key { ratio, track, value };
    }

    pub fn decompress(&self) -> Vec3 {
        return Vec3::new(
            f16_to_f32(self.value[0]),
            f16_to_f32(self.value[1]),
            f16_to_f32(self.value[2]),
        );
    }

    pub fn simd_decompress(k0: &Float3Key, k1: &Float3Key, k2: &Float3Key, k3: &Float3Key, soa: &mut SoaVec3) {
        soa.x = simd_f16_to_f32([k0.value[0], k1.value[0], k2.value[0], k3.value[0]]);
        soa.y = simd_f16_to_f32([k0.value[1], k1.value[1], k2.value[1], k3.value[1]]);
        soa.z = simd_f16_to_f32([k0.value[2], k1.value[2], k2.value[2], k3.value[2]]);
    }
}

impl ArchiveRead<Float3Key> for Float3Key {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<Float3Key, OzzError> {
        let ratio: f32 = archive.read()?;
        let track: u16 = archive.read()?;
        let value: [u16; 3] = [archive.read()?, archive.read()?, archive.read()?];
        return Ok(Float3Key { ratio, track, value });
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct QuaternionKey {
    pub ratio: f32,
    // track: 13 => The track this key frame belongs to.
    // largest: 2 => The largest component of the quaternion.
    // sign: 1 => The sign of the largest component. 1 for negative.
    bit_field: u16,
    value: [i16; 3], // The quantized value of the 3 smallest components.
}

impl QuaternionKey {
    pub fn new(ratio: f32, bit_field: u16, value: [i16; 3]) -> QuaternionKey {
        return QuaternionKey {
            ratio,
            bit_field,
            value,
        };
    }

    pub fn track(&self) -> u16 {
        return self.bit_field >> 3;
    }

    pub fn largest(&self) -> u16 {
        return (self.bit_field & 0x6) >> 1;
    }

    pub fn sign(&self) -> u16 {
        return self.bit_field & 0x1;
    }

    pub fn decompress(&self) -> Quat {
        const MAPPING: [[usize; 4]; 4] = [[0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 0]];

        let mask = &MAPPING[self.largest() as usize];
        let mut cmp_keys = [
            self.value[mask[0]],
            self.value[mask[1]],
            self.value[mask[2]],
            self.value[mask[3]],
        ];
        cmp_keys[self.largest() as usize] = 0;

        const INT_2_FLOAT: f32 = 1.0f32 / (32767.0f32 * core::f32::consts::SQRT_2);
        let mut cpnt = Vec4::new(
            (cmp_keys[0] as f32) * INT_2_FLOAT,
            (cmp_keys[1] as f32) * INT_2_FLOAT,
            (cmp_keys[2] as f32) * INT_2_FLOAT,
            (cmp_keys[3] as f32) * INT_2_FLOAT,
        );

        let dot = cpnt[0] * cpnt[0] + cpnt[1] * cpnt[1] + cpnt[2] * cpnt[2] + cpnt[3] * cpnt[3];
        let ww0 = f32::max(1e-16f32, 1f32 - dot);
        let w0 = ww0.sqrt();
        let restored = if self.sign() == 0 { w0 } else { -w0 };

        cpnt[self.largest() as usize] = restored;
        return Quat::from_vec4(cpnt);
    }

    #[rustfmt::skip]
    pub fn simd_decompress(
        k0: &QuaternionKey,
        k1: &QuaternionKey,
        k2: &QuaternionKey,
        k3: &QuaternionKey,
        soa: &mut SoaQuat,
    ) {
        const INT_2_FLOAT: f32x4 = f32x4::from_array([1.0 / (32767.0 * core::f32::consts::SQRT_2); 4]);

        const ONE: f32x4 = f32x4::from_array([1.0; 4]);
        const SMALL: f32x4 = f32x4::from_array([1e-16; 4]);

        const MASK_F000:i32x4 = i32x4::from_array([-1i32, 0, 0, 0]);
        const MASK_0F00:i32x4 = i32x4::from_array([0, -1i32, 0, 0]);
        const MASK_00F0:i32x4 = i32x4::from_array([0, 0, -1i32, 0]);
        const MASK_000F:i32x4 = i32x4::from_array([0, 0, 0, -1i32]);

        const MAPPING: [[usize; 4]; 4] = [[0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 0]];

        let m0 = &MAPPING[k0.largest() as usize];
        let m1 = &MAPPING[k1.largest() as usize];
        let m2 = &MAPPING[k2.largest() as usize];
        let m3 = &MAPPING[k3.largest() as usize];

        let mut cmp_keys: [f32x4; 4] = [
            f32x4::from_array([ k0.value[m0[0]] as f32, k1.value[m1[0]] as f32, k2.value[m2[0]] as f32, k3.value[m3[0]] as f32 ]),
            f32x4::from_array([ k0.value[m0[1]] as f32, k1.value[m1[1]] as f32, k2.value[m2[1]] as f32, k3.value[m3[1]] as f32 ]),
            f32x4::from_array([ k0.value[m0[2]] as f32, k1.value[m1[2]] as f32, k2.value[m2[2]] as f32, k3.value[m3[2]] as f32 ]),
            f32x4::from_array([ k0.value[m0[3]] as f32, k1.value[m1[3]] as f32, k2.value[m2[3]] as f32, k3.value[m3[3]] as f32 ]),
        ]; // TODO: simd int to float
        cmp_keys[k0.largest() as usize][0] = 0.0f32;
        cmp_keys[k1.largest() as usize][1] = 0.0f32;
        cmp_keys[k2.largest() as usize][2] = 0.0f32;
        cmp_keys[k3.largest() as usize][3] = 0.0f32;

        let mut cpnt = [
            INT_2_FLOAT * cmp_keys[0],
            INT_2_FLOAT * cmp_keys[1],
            INT_2_FLOAT * cmp_keys[2],
            INT_2_FLOAT * cmp_keys[3],
        ];
        let dot = cpnt[0] * cpnt[0] + cpnt[1] * cpnt[1] + cpnt[2] * cpnt[2] + cpnt[3] * cpnt[3];
        let ww0 = f32x4::simd_max(SMALL, ONE - dot);
        let w0 = ww0.sqrt();
        let sign = i32x4::from_array([k0.sign() as i32, k1.sign() as i32, k2.sign() as i32, k3.sign() as i32]) << 31;
        let restored = ix4(w0) | sign;

        cpnt[k0.largest() as usize] = fx4(ix4(cpnt[k0.largest() as usize]) | (restored & MASK_F000));
        cpnt[k1.largest() as usize] = fx4(ix4(cpnt[k1.largest() as usize]) | (restored & MASK_0F00));
        cpnt[k2.largest() as usize] = fx4(ix4(cpnt[k2.largest() as usize]) | (restored & MASK_00F0));
        cpnt[k3.largest() as usize] = fx4(ix4(cpnt[k3.largest() as usize]) | (restored & MASK_000F));

        soa.x = unsafe { mem::transmute(cpnt[0]) };
        soa.y = unsafe { mem::transmute(cpnt[1]) };
        soa.z = unsafe { mem::transmute(cpnt[2]) };
        soa.w = unsafe { mem::transmute(cpnt[3]) };
    }
}

impl ArchiveRead<QuaternionKey> for QuaternionKey {
    fn read<R: Read>(archive: &mut Archive<R>) -> Result<QuaternionKey, OzzError> {
        let ratio: f32 = archive.read()?;
        let track: u16 = archive.read()?;
        let largest: u8 = archive.read()?;
        let sign: u8 = archive.read()?;
        let bit_field: u16 = ((track & 0x1FFF) << 3) | ((largest as u16 & 0x3) << 1) | (sign as u16 & 0x1);
        let value: [i16; 3] = [archive.read()?, archive.read()?, archive.read()?];
        return Ok(QuaternionKey {
            ratio,
            bit_field,
            value,
        });
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
pub struct Animation {
    duration: f32,
    num_tracks: usize,
    name: String,
    translations: Vec<Float3Key>,
    rotations: Vec<QuaternionKey>,
    scales: Vec<Float3Key>,
}

impl Animation {
    /// `Animation` resource file tag for `Archive`.
    #[inline]
    pub fn tag() -> &'static str {
        return "ozz-animation";
    }

    #[inline]
    /// `Animation` resource file version for `Archive`.
    pub fn version() -> u32 {
        return 6;
    }

    #[cfg(test)]
    pub(crate) fn from_raw(
        duration: f32,
        num_tracks: usize,
        name: String,
        translations: Vec<Float3Key>,
        rotations: Vec<QuaternionKey>,
        scales: Vec<Float3Key>,
    ) -> Animation {
        return Animation {
            duration,
            num_tracks,
            name,
            translations,
            rotations,
            scales,
        };
    }

    /// Reads an `Animation` from an `Archive`.
    pub fn from_archive(archive: &mut Archive<impl Read>) -> Result<Animation, OzzError> {
        if archive.tag() != Self::tag() {
            return Err(OzzError::InvalidTag);
        }
        if archive.version() != Self::version() {
            return Err(OzzError::InvalidVersion);
        }

        let duration: f32 = archive.read()?;
        let num_tracks: i32 = archive.read()?;
        let name_len: i32 = archive.read()?;
        let translation_count: i32 = archive.read()?;
        let rotation_count: i32 = archive.read()?;
        let scale_count: i32 = archive.read()?;

        let mut name = String::new();
        if name_len != 0 {
            let buf = archive.read_vec(name_len as usize)?;
            name = String::from_utf8(buf).map_err(|e| e.utf8_error())?;
        }
        let translations: Vec<Float3Key> = archive.read_vec(translation_count as usize)?;
        let rotations: Vec<QuaternionKey> = archive.read_vec(rotation_count as usize)?;
        let scales: Vec<Float3Key> = archive.read_vec(scale_count as usize)?;

        return Ok(Animation {
            duration,
            num_tracks: num_tracks as usize,
            name,
            translations,
            rotations,
            scales,
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

    /// Gets the buffer of translations keys.
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
        let res = Float3Key {
            ratio: 0.0,
            track: 0,
            value: [11405, 34240, 31],
        }
        .decompress();
        assert_eq!(res, Vec3::new(0.0711059570, -8.77380371e-05, 1.84774399e-06));

        let res = Float3Key {
            ratio: 0.0,
            track: 0,
            value: [9839, 1, 0],
        }
        .decompress();
        assert_eq!(res, Vec3::new(0.0251312255859375, 5.960464477539063e-8, 0.0));
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_simd_decompress_float3() {
        let k0 = Float3Key {
            ratio: 0.0,
            track: 0,
            value: [11405, 34240, 31],
        };
        let k1 = Float3Key {
            ratio: 0.0,
            track: 0,
            value: [9839, 1, 0],
        };
        let k2 = Float3Key {
            ratio: 0.0,
            track: 0,
            value: [11405, 34240, 31],
        };
        let k3 = Float3Key {
            ratio: 0.0,
            track: 0,
            value: [9839, 1, 0],
        };
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
    fn test_quaternion_key_decompress() {
        let quat = QuaternionKey {
            ratio: 0.0,
            bit_field: (3 << 1) | 0,
            value: [396, 409, 282],
        }
        .decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(
                0.008545618438802194,
                0.008826156417853781,
                0.006085516160965199,
                0.9999060145140845,
            )
        );

        let quat = QuaternionKey {
            ratio: 0.0,
            bit_field: (0 << 1) | 0,
            value: [5256, -14549, 25373],
        }
        .decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(
                0.767303715540273,
                0.11342366291501094,
                -0.3139651582478109,
                0.5475453955750709,
            )
        );

        let quat = QuaternionKey {
            ratio: 0.0,
            bit_field: (3 << 1) | 0,
            value: [0, 0, -195],
        }
        .decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(0.00000000, 0.00000000, -0.00420806976, 0.999991119)
        );

        let quat = QuaternionKey {
            ratio: 0.0,
            bit_field: (2 << 1) | 1,
            value: [-23255, -23498, 21462],
        }
        .decompress();
        assert_eq!(
            quat,
            Quat::from_xyzw(-0.501839280, -0.507083178, -0.525850952, 0.463146627)
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_simd_decompress_quaternion() {
        let quat0 = QuaternionKey {
            ratio: 0.0,
            bit_field: (3 << 1) | 0,
            value: [396, 409, 282],
        };
        let quat1 = QuaternionKey {
            ratio: 0.0,
            bit_field: (0 << 1) | 0,
            value: [5256, -14549, 25373],
        };
        let quat2 = QuaternionKey {
            ratio: 0.0,
            bit_field: (3 << 1) | 0,
            value: [0, 0, -195],
        };
        let quat3 = QuaternionKey {
            ratio: 0.0,
            bit_field: (2 << 1) | 1,
            value: [-23255, -23498, 21462],
        };
        let mut soa = SoaQuat::default();
        QuaternionKey::simd_decompress(&quat0, &quat1, &quat2, &quat3, &mut soa);
        assert_eq!(
            soa,
            SoaQuat {
                x: f32x4::from_array([0.0085456185, 0.7673037, 0.0, -0.5018393]),
                y: f32x4::from_array([0.008826156, 0.11342366, 0.0, -0.5070832]),
                z: f32x4::from_array([0.006085516, -0.31396517, -0.0042080698, -0.52585095]),
                w: f32x4::from_array([0.999906, 0.5475454, 0.9999911, 0.46314663]),
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

        let last = animation.translations().len() - 1;
        assert_eq!(animation.translations().len(), 178);
        assert_eq!(animation.translations[0].ratio, 0f32);
        assert_eq!(animation.translations[0].track, 0);
        assert_eq!(animation.translations[0].value, [0, 15400, 43950]);
        assert_eq!(animation.translations[last].ratio, 1f32);
        assert_eq!(animation.translations[last].track, 0);
        assert_eq!(animation.translations[last].value, [3659, 15400, 43933]);

        let last = animation.rotations().len() - 1;
        assert_eq!(animation.rotations().len(), 1678);
        assert_eq!(animation.rotations[0].ratio, 0f32);
        assert_eq!(animation.rotations[0].track(), 0);
        assert_eq!(animation.rotations[0].largest(), 2);
        assert_eq!(animation.rotations[0].sign(), 1);
        assert_eq!(animation.rotations[0].value, [-22775, -23568, 21224]);
        assert_eq!(animation.rotations[last].ratio, 1f32);
        assert_eq!(animation.rotations[last].track(), 63);
        assert_eq!(animation.rotations[last].largest(), 3);
        assert_eq!(animation.rotations[last].sign(), 0);
        assert_eq!(animation.rotations[last].value, [0, 0, -2311]);

        let last = animation.scales().len() - 1;
        assert_eq!(animation.scales().len(), 136);
        assert_eq!(animation.scales()[0].ratio, 0f32);
        assert_eq!(animation.scales()[0].track, 0);
        assert_eq!(animation.scales()[0].value, [15360, 15360, 15360]);
        assert_eq!(animation.scales()[last].ratio, 1f32);
        assert_eq!(animation.scales()[last].track, 67);
        assert_eq!(animation.scales()[last].value, [15360, 15360, 15360]);
    }
}
