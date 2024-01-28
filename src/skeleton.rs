use std::collections::HashMap;
use std::path::Path;

use crate::archive::{ArchiveReader, ArchiveTag, ArchiveVersion, IArchive};
use crate::math::SoaTransform;
use crate::{DeterministicState, OzzError};

#[derive(Debug)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
pub struct Skeleton {
    pub(crate) joint_rest_poses: Vec<SoaTransform>,
    pub(crate) joint_parents: Vec<i16>,
    pub(crate) joint_names: HashMap<String, i16, DeterministicState>,
}

impl ArchiveVersion for Skeleton {
    fn version() -> u32 {
        return 2;
    }
}

impl ArchiveTag for Skeleton {
    fn tag() -> &'static str {
        return "ozz-skeleton";
    }
}

impl ArchiveReader<Skeleton> for Skeleton {
    fn read(archive: &mut IArchive) -> Result<Skeleton, OzzError> {
        if !archive.test_tag::<Self>()? {
            return Err(OzzError::InvalidTag);
        }

        let version = archive.read_version()?;
        if version != Self::version() {
            return Err(OzzError::InvalidVersion);
        }

        let num_joints: i32 = archive.read()?;
        if num_joints == 0 {
            return Ok(Skeleton {
                joint_rest_poses: Vec::new(),
                joint_parents: Vec::new(),
                joint_names: HashMap::with_hasher(DeterministicState::new()),
            });
        }

        let _char_count: i32 = archive.read()?;
        let mut joint_names = HashMap::with_capacity_and_hasher(num_joints as usize, DeterministicState::new());
        for idx in 0..num_joints {
            joint_names.insert(archive.read_string(0)?, idx as i16);
        }

        let joint_parents: Vec<i16> = archive.read_vec(num_joints as usize)?;

        let soa_num_joints = (num_joints + 3) / 4;
        let mut joint_rest_poses: Vec<SoaTransform> = Vec::with_capacity(soa_num_joints as usize);
        for _ in 0..soa_num_joints {
            joint_rest_poses.push(archive.read()?);
        }

        return Ok(Skeleton {
            joint_rest_poses,
            joint_parents,
            joint_names,
        });
    }
}

impl Skeleton {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Skeleton, OzzError> {
        let mut archive = IArchive::new(path)?;
        return Skeleton::read(&mut archive);
    }

    pub fn from_reader(reader: &mut IArchive) -> Result<Skeleton, OzzError> {
        return Skeleton::read(reader);
    }
}

impl Skeleton {
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

    pub fn num_soa_joints(&self) -> usize {
        return (self.joint_parents.len() + 3) / 4;
    }

    pub fn joint_rest_poses(&self) -> &[SoaTransform] {
        return &self.joint_rest_poses;
    }

    pub fn joint_parents(&self) -> &[i16] {
        return &self.joint_parents;
    }

    pub fn joint_parent(&self, idx: usize) -> i16 {
        return self.joint_parents[idx];
    }

    pub fn joint_names(&self) -> &HashMap<String, i16, DeterministicState> {
        return &self.joint_names;
    }

    pub fn joint_by_name(&self, name: &str) -> Option<i16> {
        return self.joint_names.get(name).map(|idx| *idx);
    }

    pub fn index_joint(&self, idx: i16) -> Option<&SoaTransform> {
        return self.joint_rest_poses.get(idx as usize);
    }

    pub fn is_leaf(&self, joint: i16) -> bool {
        let next = (joint + 1) as usize;
        return next == self.num_joints() || self.joint_parents()[next] != joint;
    }

    pub fn iter_depth_first<F>(&self, from: i16, mut f: F)
    where
        F: FnMut(i16, i16),
    {
        let mut i = if from < 0 { 0 } else { from } as usize;
        let mut process = i < self.num_joints();
        while process {
            f(i as i16, self.joint_parent(i));
            i += 1;
            process = i < self.num_joints() && self.joint_parent(i) >= from;
        }
    }

    pub fn iter_depth_first_reverse<F>(&self, mut f: F)
    where
        F: FnMut(i16, i16),
    {
        for i in (0..self.num_joints()).rev() {
            let parent = self.joint_parent(i);
            f(i as i16, parent);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::simd::prelude::*;

    use super::*;
    use crate::math::{SoaQuat, SoaVec3};

    #[test]
    fn test_read_skeleton() {
        let mut archive = IArchive::new("./resource/playback/skeleton.ozz").unwrap();
        let skeleton = Skeleton::read(&mut archive).unwrap();

        assert_eq!(skeleton.joint_rest_poses().len(), 17);
        assert_eq!(
            skeleton.joint_rest_poses()[0].translation,
            SoaVec3 {
                x: f32x4::from_array([-4.01047945e-10, 0.00000000, 0.0710870326, 0.110522307]),
                y: f32x4::from_array([1.04666960, 0.00000000, -8.79573781e-05, -7.82728166e-05]),
                z: f32x4::from_array([-0.0151103791, 0.00000000, 9.85883801e-08, -2.17094467e-10]),
            },
        );
        assert_eq!(
            skeleton.joint_rest_poses()[16].translation,
            SoaVec3 {
                x: f32x4::from_array([0.458143145, 0.117970668, 0.0849116519, 0.00000000]),
                y: f32x4::from_array([2.64545919e-09, 0.148304969, 0.00000000, 0.00000000]),
                z: f32x4::from_array([-4.97557555e-14, -7.47846236e-15, -1.77635680e-17, 0.00000000]),
            }
        );

        assert_eq!(
            skeleton.joint_rest_poses()[0].rotation,
            SoaQuat {
                x: f32x4::from_array([-0.500000000, -0.499999702, -1.41468570e-06, -3.05311332e-14]),
                y: f32x4::from_array([-0.500000000, -0.500000358, -6.93941161e-07, 1.70812796e-22]),
                z: f32x4::from_array([-0.500000000, -0.499999702, 0.000398159056, 1.08420217e-19]),
                w: f32x4::from_array([0.500000000, 0.500000358, 1.00000000, 1.00000000]),
            },
        );
        assert_eq!(
            skeleton.joint_rest_poses()[16].rotation,
            SoaQuat {
                x: f32x4::from_array([-2.20410801e-09, 4.11812209e-07, -6.55128745e-32, 0.00000000]),
                y: f32x4::from_array([4.60687737e-08, -4.11812152e-07, -1.30968591e-21, 0.00000000]),
                z: f32x4::from_array([0.0498105064, 0.707106829, -2.46519033e-32, 0.00000000]),
                w: f32x4::from_array([0.998758733, 0.707106769, 1.00000000, 1.00000000]),
            }
        );

        assert_eq!(
            skeleton.joint_rest_poses()[0].scale,
            SoaVec3 {
                x: f32x4::from_array([1.0, 1.0, 1.0, 1.0]),
                y: f32x4::from_array([1.0, 1.0, 1.0, 1.0]),
                z: f32x4::from_array([1.0, 1.0, 1.0, 1.0]),
            },
        );
        assert_eq!(
            skeleton.joint_rest_poses()[16].scale,
            SoaVec3 {
                x: f32x4::from_array([0.999999940, 1.0, 1.0, 1.0]),
                y: f32x4::from_array([0.999999940, 1.0, 1.0, 1.0]),
                z: f32x4::from_array([1.0, 1.0, 1.0, 1.0]),
            }
        );

        assert_eq!(skeleton.joint_parents().len(), 67);
        assert_eq!(skeleton.joint_parents()[0], -1);
        assert_eq!(skeleton.joint_parents()[66], 65);

        assert_eq!(skeleton.joint_names().len(), 67);
        assert_eq!(skeleton.joint_names()["Hips"], 0);
        assert_eq!(skeleton.joint_names()["Bip01 R Toe0Nub"], 66);
    }
}
