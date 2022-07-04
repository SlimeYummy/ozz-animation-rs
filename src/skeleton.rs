use crate::archive::{ArchiveReader, ArchiveTag, ArchiveVersion, IArchive};
use crate::math::{OzzNumber, OzzTransform};
use anyhow::{anyhow, Result};
use nalgebra::{Quaternion, Vector3};
use std::collections::HashMap;
use std::mem;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoaTransform {
    pub translation_x: [f32; 4],
    pub translation_y: [f32; 4],
    pub translation_z: [f32; 4],

    pub rotation_x: [f32; 4],
    pub rotation_y: [f32; 4],
    pub rotation_z: [f32; 4],
    pub rotation_w: [f32; 4],

    pub scale_x: [f32; 4],
    pub scale_y: [f32; 4],
    pub scale_z: [f32; 4],
}

impl ArchiveReader<SoaTransform> for SoaTransform {
    fn read(archive: &mut IArchive) -> Result<SoaTransform> {
        const COUNT: usize = mem::size_of::<SoaTransform>() / mem::size_of::<f32>();
        let mut buffer = [0f32; COUNT];
        for idx in 0..COUNT {
            buffer[idx] = archive.read()?;
        }
        return Ok(unsafe { mem::transmute(buffer) });
    }
}

impl SoaTransform {
    pub fn aos_at(&self, idx: usize) -> OzzTransform<f32> {
        let translation = Vector3::new(
            self.translation_x[idx],
            self.translation_y[idx],
            self.translation_z[idx],
        );
        let rotation = Quaternion::new(
            self.rotation_x[idx],
            self.rotation_y[idx],
            self.rotation_z[idx],
            self.rotation_w[idx],
        );
        let scale = Vector3::new(self.scale_x[idx], self.scale_y[idx], self.scale_z[idx]);
        return OzzTransform {
            translation,
            rotation,
            scale,
        };
    }
}

#[derive(Debug)]
pub struct Skeleton<N: OzzNumber> {
    pub(crate) joint_bind_poses: Vec<OzzTransform<N>>,
    pub(crate) joint_parents: Vec<i16>,
    pub(crate) joint_names: HashMap<String, i16>,
}

impl<N: OzzNumber> ArchiveVersion for Skeleton<N> {
    fn version() -> u32 {
        return 2;
    }
}

impl<N: OzzNumber> ArchiveTag for Skeleton<N> {
    fn tag() -> &'static str {
        return "ozz-skeleton";
    }
}

impl<N: OzzNumber> ArchiveReader<Skeleton<N>> for Skeleton<N> {
    fn read(archive: &mut IArchive) -> Result<Skeleton<N>> {
        if !archive.test_tag::<Self>()? {
            return Err(anyhow!("Invalid tag"));
        }

        let version = archive.read_version()?;
        if version != Self::version() {
            return Err(anyhow!("Invalid version"));
        }

        let num_joints: i32 = archive.read()?;
        if num_joints == 0 {
            return Ok(Skeleton {
                joint_bind_poses: Vec::new(),
                joint_parents: Vec::new(),
                joint_names: HashMap::new(),
            });
        }

        let _char_count: i32 = archive.read()?;
        let mut joint_names = HashMap::with_capacity(num_joints as usize);
        for idx in 0..num_joints {
            joint_names.insert(archive.read_string(0)?, idx as i16);
        }

        let joint_parents: Vec<i16> = archive.read_vec(num_joints as usize)?;

        let mut joint_bind_poses: Vec<OzzTransform<f32>> = Vec::with_capacity(num_joints as usize);
        let mut aos_counter = 0;
        let soa_num_joints = (num_joints + 3) / 4;
        for _ in 0..soa_num_joints {
            let soa: SoaTransform = archive.read()?;
            for idx in 0..4 {
                if aos_counter < num_joints {
                    aos_counter += 1;
                    joint_bind_poses.push(soa.aos_at(idx));
                }
            }
        }

        let joint_bind_poses = joint_bind_poses
            .iter()
            .map(|t| OzzTransform::parse_f32(t))
            .collect();
        return Ok(Skeleton {
            joint_bind_poses,
            joint_parents,
            joint_names,
        });
    }
}

impl<N: OzzNumber> Skeleton<N> {
    #[inline(always)]
    pub fn max_joints() -> usize {
        return 1024;
    }

    #[inline(always)]
    pub fn max_soa_joints() -> usize {
        return (Self::max_joints() + 3) / 4;
    }

    #[inline(always)]
    pub fn no_parent() -> i16 {
        return -1;
    }

    pub fn num_joints(&self) -> usize {
        return self.joint_parents.len();
    }

    pub fn num_aligned_joints(&self) -> usize {
        return (self.num_joints() + 3) & !0x3;
    }

    pub fn joint_bind_poses(&self) -> &[OzzTransform<N>] {
        return &self.joint_bind_poses;
    }

    pub fn joint_parents(&self) -> &[i16] {
        return &self.joint_parents;
    }

    pub fn joint_parent(&self, idx: usize) -> i16 {
        return self.joint_parents[idx];
    }

    pub fn joint_names(&self) -> &HashMap<String, i16> {
        return &self.joint_names;
    }

    pub fn joint_by_name(&self, name: &str) -> Option<i16> {
        return self.joint_names.get(name).map(|idx| *idx);
    }

    pub fn index_joint(&self, idx: i16) -> Option<&OzzTransform<N>> {
        return self.joint_bind_poses.get(idx as usize);
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Quaternion, Vector3};

    use super::*;

    #[test]
    fn test_read_skeleton() {
        let mut archive = IArchive::new("./test_files/skeleton-simple.ozz").unwrap();
        let skeleton = Skeleton::<f32>::read(&mut archive).unwrap();

        assert_eq!(skeleton.joint_bind_poses().len(), 67);
        assert_eq!(
            skeleton.joint_bind_poses()[0].translation,
            Vector3::new(0.00000000, 1.04666960, -0.0151103791)
        );
        assert_eq!(
            skeleton.joint_bind_poses()[1].translation,
            Vector3::new(0.00000000, 0.00000000, 0.00000000)
        );
        assert_eq!(
            skeleton.joint_bind_poses()[66].translation,
            Vector3::new(0.0849116519, 2.77555750e-19, 0.00000000)
        );

        assert_eq!(
            skeleton.joint_bind_poses()[0].rotation,
            Quaternion::new(0.500000000, 0.500000000, 0.500000000, -0.500000000)
        );
        assert_eq!(
            skeleton.joint_bind_poses()[66].rotation,
            Quaternion::new(
                -1.05879131e-22,
                -1.30968591e-21,
                -1.97215226e-31,
                1.00000000
            )
        );

        assert_eq!(
            skeleton.joint_bind_poses()[0].scale,
            Vector3::new(1.0, 1.0, 1.0)
        );
        assert_eq!(
            skeleton.joint_bind_poses()[1].scale,
            Vector3::new(1.0, 1.0, 1.0)
        );
        assert_eq!(
            skeleton.joint_bind_poses()[66].scale,
            Vector3::new(1.0, 1.0, 1.0)
        );

        assert_eq!(skeleton.joint_parents().len(), 67);
        assert_eq!(skeleton.joint_parents()[0], -1);
        assert_eq!(skeleton.joint_parents()[66], 65);

        assert_eq!(skeleton.joint_names().len(), 67);
        assert_eq!(skeleton.joint_names()["Hips"], 0);
        assert_eq!(skeleton.joint_names()["Bip01 R Toe0Nub"], 66);
    }
}
