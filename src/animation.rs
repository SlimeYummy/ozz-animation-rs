use crate::archive::{ArchiveReader, ArchiveTag, ArchiveVersion, IArchive};
use crate::math::OzzNumber;
use anyhow::{anyhow, Result};
use nalgebra::{Quaternion, Vector3, Vector4};
use std::marker::PhantomData;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Float3Key<N: OzzNumber> {
    ratio: f32,
    track: u16,
    value: [u16; 3],
    phantom: PhantomData<N>,
}

impl<N: OzzNumber> Float3Key<N> {
    pub fn new(ratio: f32, track: u16, value: [u16; 3]) -> Float3Key<N> {
        return Float3Key {
            ratio,
            track,
            value,
            phantom: PhantomData,
        };
    }

    pub fn ratio(&self) -> N {
        return N::parse_f32(self.ratio);
    }

    pub fn track(&self) -> u16 {
        return self.track;
    }

    pub fn decompress(&self) -> Vector3<N> {
        return Vector3::new(
            N::parse_f16(self.value[0]),
            N::parse_f16(self.value[1]),
            N::parse_f16(self.value[2]),
        );
    }
}

impl<N: OzzNumber> ArchiveReader<Float3Key<N>> for Float3Key<N> {
    fn read(archive: &mut IArchive) -> Result<Float3Key<N>> {
        let ratio: f32 = archive.read()?;
        let track: u16 = archive.read()?;
        let value: [u16; 3] = [archive.read()?, archive.read()?, archive.read()?];
        return Ok(Float3Key {
            ratio,
            track,
            value,
            phantom: PhantomData,
        });
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuaternionKey<N: OzzNumber> {
    ratio: f32,
    // track: 13 => The track this key frame belongs to.
    // largest: 2 => The largest component of the quaternion.
    // sign: 1 => The sign of the largest component. 1 for negative.
    bit_field: u16,
    value: [i16; 3], // The quantized value of the 3 smallest components.
    phantom: PhantomData<N>,
}

impl<N: OzzNumber> QuaternionKey<N> {
    pub fn new(ratio: f32, bit_field: u16, value: [i16; 3]) -> QuaternionKey<N> {
        return QuaternionKey {
            ratio,
            bit_field,
            value,
            phantom: PhantomData,
        };
    }

    pub fn ratio(&self) -> N {
        return N::parse_f32(self.ratio);
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

    pub fn decompress(&self) -> Quaternion<N> {
        const MAPPING: [[usize; 4]; 4] = [[0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 0, 2], [0, 1, 2, 0]];

        let mask = MAPPING[self.largest() as usize];
        let mut cmp_keys = Vector4::new(
            self.value[mask[0]],
            self.value[mask[1]],
            self.value[mask[2]],
            self.value[mask[3]],
        );
        cmp_keys[self.largest() as usize] = 0;

        let int2float = N::parse_f32(0.0000215798450022f32); // 1 ÷ (32767 × √2)
        let mut cpnt = Vector4::new(
            N::parse_i16(cmp_keys[0]) * int2float,
            N::parse_i16(cmp_keys[1]) * int2float,
            N::parse_i16(cmp_keys[2]) * int2float,
            N::parse_i16(cmp_keys[3]) * int2float,
        );

        let dot = cpnt[0] * cpnt[0] + cpnt[1] * cpnt[1] + cpnt[2] * cpnt[2] + cpnt[3] * cpnt[3];
        let ww0 = N::max(N::parse_f32(1e-16f32), N::one() - dot);
        let w0 = ww0.sqrt();
        let restored = if self.sign() == 0 { w0 } else { -w0 };

        cpnt[self.largest() as usize] = restored;
        return Quaternion::from(cpnt);
    }
}

impl<N: OzzNumber> ArchiveReader<QuaternionKey<N>> for QuaternionKey<N> {
    fn read(archive: &mut IArchive) -> Result<QuaternionKey<N>> {
        let ratio: f32 = archive.read()?;
        let track: u16 = archive.read()?;
        let largest: u8 = archive.read()?;
        let sign: u8 = archive.read()?;
        let bit_field: u16 =
            ((track & 0x1FFF) << 3) | ((largest as u16 & 0x3) << 1) | (sign as u16 & 0x1);
        let value: [i16; 3] = [archive.read()?, archive.read()?, archive.read()?];
        return Ok(QuaternionKey {
            ratio,
            bit_field,
            value,
            phantom: PhantomData,
        });
    }
}

#[derive(Debug)]
pub struct Animation<N: OzzNumber> {
    pub(crate) duration: N,
    pub(crate) num_tracks: usize,
    pub(crate) name: String,
    pub(crate) translations: Vec<Float3Key<N>>,
    pub(crate) rotations: Vec<QuaternionKey<N>>,
    pub(crate) scales: Vec<Float3Key<N>>,
}

impl<N: OzzNumber> ArchiveVersion for Animation<N> {
    fn version() -> u32 {
        return 6;
    }
}

impl<N: OzzNumber> ArchiveTag for Animation<N> {
    fn tag() -> &'static str {
        return "ozz-animation";
    }
}

impl<N: OzzNumber> ArchiveReader<Animation<N>> for Animation<N> {
    fn read(archive: &mut IArchive) -> Result<Animation<N>> {
        if !archive.test_tag::<Self>()? {
            return Err(anyhow!("Invalid tag"));
        }

        let version = archive.read_version()?;
        if version != Self::version() {
            return Err(anyhow!("Invalid version"));
        }

        let duration: N = N::parse_f32(archive.read()?);
        let num_tracks: i32 = archive.read()?;
        let name_len: i32 = archive.read()?;
        let translation_count: i32 = archive.read()?;
        let rotation_count: i32 = archive.read()?;
        let scale_count: i32 = archive.read()?;

        let name: String = archive.read_string(name_len as usize)?;
        let translations: Vec<Float3Key<N>> = archive.read_vec(translation_count as usize)?;
        let rotations: Vec<QuaternionKey<N>> = archive.read_vec(rotation_count as usize)?;
        let scales: Vec<Float3Key<N>> = archive.read_vec(scale_count as usize)?;

        return Ok(Animation {
            duration,
            num_tracks: num_tracks as usize,
            name,
            translations,
            rotations,
            scales,
        });
    }
}

impl<N: OzzNumber> Drop for Animation<N> {
    fn drop(&mut self) {}
}

impl<N: OzzNumber> Animation<N> {
    pub fn duration(&self) -> N {
        return self.duration;
    }

    pub fn num_tracks(&self) -> usize {
        return self.num_tracks;
    }

    pub fn num_aligned_tracks(&self) -> usize {
        return (self.num_tracks + 3) & !0x3;
    }

    pub fn name(&self) -> &str {
        return &self.name;
    }

    pub fn translations(&self) -> &[Float3Key<N>] {
        return &self.translations;
    }

    pub fn rotations(&self) -> &[QuaternionKey<N>] {
        return &self.rotations;
    }

    pub fn scales(&self) -> &[Float3Key<N>] {
        return &self.scales;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float3_key_decompress() {
        let res = Float3Key::<f32> {
            ratio: 0.0,
            track: 0,
            value: [11405, 34240, 31],
            phantom: PhantomData,
        }
        .decompress();
        assert_eq!(
            res,
            Vector3::new(0.0711059570, -8.77380371e-05, 1.84774399e-06)
        );

        let res = Float3Key::<f64> {
            ratio: 0.0,
            track: 0,
            value: [9839, 1, 0],
            phantom: PhantomData,
        }
        .decompress();
        assert_eq!(
            res,
            Vector3::new(0.0251312255859375, 5.960464477539063e-8, 0.0)
        );
    }

    #[test]
    fn test_quaternion_key_decompress() {
        let res = QuaternionKey::<f64> {
            ratio: 0.0,
            bit_field: (3 << 1) | 0,
            value: [396, 409, 282],
            phantom: PhantomData,
        }
        .decompress();
        assert_eq!(
            res,
            Quaternion::new(
                0.9999060145140845,
                0.008545618438802194,
                0.008826156417853781,
                0.006085516160965199
            )
        );

        let res = QuaternionKey::<f64> {
            ratio: 0.0,
            bit_field: (0 << 1) | 0,
            value: [5256, -14549, 25373],
            phantom: PhantomData,
        }
        .decompress();
        assert_eq!(
            res,
            Quaternion::new(
                0.5475453955750709,
                0.767303715540273,
                0.11342366291501094,
                -0.3139651582478109
            )
        );

        let res = QuaternionKey::<f32> {
            ratio: 0.0,
            bit_field: (3 << 1) | 0,
            value: [0, 0, -195],
            phantom: PhantomData,
        }
        .decompress();
        assert_eq!(
            res,
            Quaternion::new(0.999991119, 0.00000000, 0.00000000, -0.00420806976)
        );

        let res = QuaternionKey::<f32> {
            ratio: 0.0,
            bit_field: (2 << 1) | 1,
            value: [-23255, -23498, 21462],
            phantom: PhantomData,
        }
        .decompress();
        assert_eq!(
            res,
            Quaternion::new(0.463146627, -0.501839280, -0.507083178, -0.525850952)
        );
    }

    #[test]
    fn test_read_animation() {
        let mut archive = IArchive::new("./test_files/animation-simple.ozz").unwrap();
        let animation = Animation::<f32>::read(&mut archive).unwrap();

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
