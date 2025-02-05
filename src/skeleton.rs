//!
//! Skeleton data structure definition.
//!

use bimap::BiHashMap;
use std::alloc::{self, Layout};
use std::io::Read;
use std::{mem, slice};

use crate::archive::Archive;
use crate::base::{DeterministicState, OzzError, OzzIndex};
use crate::math::SoaTransform;

/// Rexported `BiHashMap` in bimap crate.
pub type JointHashMap = BiHashMap<String, i16, DeterministicState, DeterministicState>;

///
/// This runtime skeleton data structure provides a const-only access to joint
/// hierarchy, joint names and rest-pose.
///
/// Joint names, rest-poses and hierarchy information are all stored in separate
/// arrays of data (as opposed to joint structures for the SkeletonRaw), in order
/// to closely match with the way runtime algorithms use them. Joint hierarchy is
/// packed as an array of parent jont indices (16 bits), stored in depth-first
/// order. This is enough to traverse the whole joint hierarchy. Use
/// iter_depth_first() to implement a depth-first traversal utility.
///
#[derive(Debug)]
pub struct Skeleton {
    size: usize,
    num_joints: u32,
    num_soa_joints: u32,
    joint_rest_poses: *mut SoaTransform,
    joint_names: JointHashMap,
    joint_parents: *mut i16,
}

impl Drop for Skeleton {
    fn drop(&mut self) {
        if !self.joint_rest_poses.is_null() {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.size, mem::align_of::<SoaTransform>());
                alloc::dealloc(self.joint_rest_poses as *mut u8, layout);
            }
            self.joint_rest_poses = std::ptr::null_mut();
            self.joint_parents = std::ptr::null_mut();
        }
    }
}

/// Skeleton meta in `Archive`.
#[derive(Debug)]
pub struct SkeletonMeta {
    pub version: u32,
    pub num_joints: u32,
    pub joint_names: JointHashMap,
    pub joint_parents: Vec<i16>,
}

#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct SkeletonRaw {
    pub joint_rest_poses: Vec<SoaTransform>,
    pub joint_parents: Vec<i16>,
    pub joint_names: JointHashMap,
}

impl Skeleton {
    /// `Skeleton` resource file tag for `Archive`.
    #[inline]
    pub fn tag() -> &'static str {
        "ozz-skeleton"
    }

    /// `Skeleton` resource file version for `Archive`.
    #[inline]
    pub fn version() -> u32 {
        2
    }

    /// Reads a `SkeletonMeta` from a reader.
    pub fn read_meta(archive: &mut Archive<impl Read>, with_joints: bool) -> Result<SkeletonMeta, OzzError> {
        if archive.tag() != Self::tag() {
            return Err(OzzError::InvalidTag);
        }
        if archive.version() != Self::version() {
            return Err(OzzError::InvalidVersion);
        }

        let num_joints: u32 = archive.read()?;
        if num_joints == 0 || !with_joints {
            return Ok(SkeletonMeta {
                version: Self::version(),
                num_joints,
                joint_names: BiHashMap::with_hashers(DeterministicState::new(), DeterministicState::new()),
                joint_parents: Vec::new(),
            });
        }

        let _char_count: u32 = archive.read()?;
        let mut joint_names = BiHashMap::with_capacity_and_hashers(
            num_joints as usize,
            DeterministicState::new(),
            DeterministicState::new(),
        );
        for idx in 0..num_joints {
            joint_names.insert(archive.read::<String>()?, idx as i16);
        }

        let joint_parents: Vec<i16> = archive.read_vec(num_joints as usize)?;

        Ok(SkeletonMeta {
            version: Self::version(),
            num_joints,
            joint_names,
            joint_parents,
        })
    }

    /// Reads a `Skeleton` from a reader.
    pub fn from_archive(archive: &mut Archive<impl Read>) -> Result<Skeleton, OzzError> {
        let meta = Skeleton::read_meta(archive, false)?;
        let mut skeleton = Skeleton::new(meta);

        let _char_count: u32 = archive.read()?;
        for idx in 0..skeleton.num_joints() {
            skeleton.joint_names.insert(archive.read::<String>()?, idx as i16);
        }

        archive.read_slice(skeleton.joint_parents_mut())?;
        archive.read_slice(skeleton.joint_rest_poses_mut())?;
        Ok(skeleton)
    }

    /// Reads a `Skeleton` from a file.
    #[cfg(not(feature = "wasm"))]
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Skeleton, OzzError> {
        let mut archive = Archive::from_path(path)?;
        Skeleton::from_archive(&mut archive)
    }

    // Only for wasm test in NodeJS environment.
    #[cfg(all(feature = "wasm", feature = "nodejs"))]
    pub fn from_path(path: &str) -> Result<Skeleton, OzzError> {
        let mut archive = Archive::from_path(path)?;
        Skeleton::from_archive(&mut archive)
    }

    pub(crate) fn from_raw(raw: &SkeletonRaw) -> Skeleton {
        let mut skeleton = Skeleton::new(SkeletonMeta {
            version: Self::version(),
            num_joints: raw.joint_parents.len() as u32,
            joint_names: BiHashMap::default(),
            joint_parents: Vec::new(),
        });
        skeleton.joint_rest_poses_mut().copy_from_slice(&raw.joint_rest_poses);
        skeleton.joint_parents_mut().copy_from_slice(&raw.joint_parents);
        skeleton.joint_names = raw.joint_names.clone();
        skeleton
    }

    pub(crate) fn to_raw(&self) -> SkeletonRaw {
        SkeletonRaw {
            joint_rest_poses: self.joint_rest_poses().to_vec(),
            joint_parents: self.joint_parents().to_vec(),
            joint_names: self.joint_names().clone(),
        }
    }

    fn new(meta: SkeletonMeta) -> Skeleton {
        let mut skeleton = Skeleton {
            size: 0,
            num_joints: meta.num_joints,
            num_soa_joints: meta.num_joints.div_ceil(4),
            joint_rest_poses: std::ptr::null_mut(),
            joint_parents: std::ptr::null_mut(),
            joint_names: BiHashMap::with_capacity_and_hashers(
                meta.num_joints as usize,
                DeterministicState::new(),
                DeterministicState::new(),
            ),
        };

        const ALIGN: usize = mem::align_of::<SoaTransform>();
        skeleton.size =
            mem::size_of::<SoaTransform>() * skeleton.num_soa_joints() + mem::size_of::<i16>() * skeleton.num_joints();

        unsafe {
            let layout = Layout::from_size_align_unchecked(skeleton.size, ALIGN);
            let mut ptr = alloc::alloc(layout);

            skeleton.joint_rest_poses = ptr as *mut SoaTransform;
            ptr = ptr.add(mem::size_of::<SoaTransform>() * skeleton.num_soa_joints());
            skeleton.joint_parents = ptr as *mut i16;
            ptr = ptr.add(mem::size_of::<i16>() * skeleton.num_joints());

            assert_eq!(ptr, (skeleton.joint_rest_poses as *mut u8).add(skeleton.size));
        }
        skeleton
    }
}

impl Skeleton {
    /// Gets the number of joints of `Skeleton`.
    #[inline]
    pub fn num_joints(&self) -> usize {
        self.num_joints as usize
    }

    /// Gets the number of joints of `Skeleton` (aligned to 4 * SoA).
    #[inline]
    pub fn num_aligned_joints(&self) -> usize {
        (self.num_joints() + 3) & !0x3
    }

    /// Gets the number of soa elements matching the number of joints of `Skeleton`.
    /// This value is useful to allocate SoA runtime data structures.
    #[inline]
    pub fn num_soa_joints(&self) -> usize {
        self.num_soa_joints as usize
    }

    /// Gets joint's rest poses. Rest poses are stored in soa format.
    #[inline]
    pub fn joint_rest_poses(&self) -> &[SoaTransform] {
        unsafe { slice::from_raw_parts(self.joint_rest_poses, self.num_soa_joints()) }
    }

    #[inline]
    fn joint_rest_poses_mut(&mut self) -> &mut [SoaTransform] {
        unsafe { slice::from_raw_parts_mut(self.joint_rest_poses, self.num_soa_joints()) }
    }

    /// Gets joint's name map.
    #[inline]
    pub fn joint_names(&self) -> &JointHashMap {
        &self.joint_names
    }

    /// Gets joint's index by name.
    #[inline]
    pub fn joint_by_name(&self, name: &str) -> Option<i16> {
        self.joint_names.get_by_left(name).copied()
    }

    /// Gets joint's name by index.
    #[inline]
    pub fn name_by_joint(&self, index: i16) -> Option<&str> {
        self.joint_names.get_by_right(&index).map(|s| s.as_str())
    }

    /// Gets joint's parent indices range.
    #[inline]
    pub fn joint_parents(&self) -> &[i16] {
        unsafe { slice::from_raw_parts(self.joint_parents, self.num_joints()) }
    }

    fn joint_parents_mut(&mut self) -> &mut [i16] {
        unsafe { slice::from_raw_parts_mut(self.joint_parents, self.num_joints()) }
    }

    /// Gets joint's parent by index.
    #[inline]
    pub fn joint_parent(&self, idx: impl OzzIndex) -> i16 {
        self.joint_parents()[idx.usize()]
    }

    /// Test if a joint is a leaf.
    ///
    /// * `joint` - `joint` must be in range [0, num joints].
    ///   Joint is a leaf if it's the last joint, or next joint's parent isn't `joint`.
    #[inline]
    pub fn is_leaf(&self, joint: impl OzzIndex) -> bool {
        let next = joint.usize() + 1;
        next == self.num_joints() || (self.joint_parents()[next] as i32 != joint.i32())
    }

    /// Iterates through the joint hierarchy in depth-first order.
    ///
    /// * `from` - The joint index to start from. If negative, the iteration starts from the root.
    /// * `f` - The function to call for each joint. The function takes arguments `(joint: i16, parent: i16)`.
    pub fn iter_depth_first<F>(&self, from: impl OzzIndex, mut f: F)
    where
        F: FnMut(i16, i16),
    {
        let mut i = if from.i32() < 0 { 0 } else { from.usize() };
        let mut process = i < self.num_joints();
        while process {
            f(i as i16, self.joint_parent(i));
            i += 1;
            process = i < self.num_joints() && (self.joint_parent(i) as i32 >= from.i32());
        }
    }

    /// Iterates through the joint hierarchy in reverse depth-first order.
    ///
    /// * `f` - The function to call for each joint. The function takes arguments `(joint: i16, parent: i16)`.
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

#[cfg(feature = "rkyv")]
pub struct ArchivedSkeleton {
    pub num_joints: u32,
    pub joint_rest_poses: rkyv::vec::ArchivedVec<SoaTransform>,
    pub joint_names: rkyv::vec::ArchivedVec<rkyv::collections::util::Entry<rkyv::string::ArchivedString, i16>>,
    pub joint_parents: rkyv::vec::ArchivedVec<i16>,
}

#[cfg(feature = "rkyv")]
const _: () = {
    use rkyv::collections::util::Entry;
    use rkyv::ser::{ScratchSpace, Serializer};
    use rkyv::vec::{ArchivedVec, VecResolver};
    use rkyv::{from_archived, out_field, Archive, Deserialize, Fallible, Serialize};

    pub struct SkeletonResolver {
        joint_rest_poses: VecResolver,
        joint_names: VecResolver,
        joint_parents: VecResolver,
    }

    impl Archive for Skeleton {
        type Archived = ArchivedSkeleton;
        type Resolver = SkeletonResolver;

        unsafe fn resolve(&self, pos: usize, resolver: SkeletonResolver, out: *mut ArchivedSkeleton) {
            let (fp, fo) = out_field!(out.num_joints);
            u32::resolve(&self.num_joints, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.joint_rest_poses);
            ArchivedVec::resolve_from_slice(self.joint_rest_poses(), pos + fp, resolver.joint_rest_poses, fo);
            let (fp, fo) = out_field!(out.joint_names);
            ArchivedVec::resolve_from_len(self.joint_names().len(), pos + fp, resolver.joint_names, fo);
            let (fp, fo) = out_field!(out.joint_parents);
            ArchivedVec::resolve_from_slice(self.joint_parents(), pos + fp, resolver.joint_parents, fo);
        }
    }

    impl<S: Serializer + ScratchSpace + ?Sized> Serialize<S> for Skeleton {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            serializer.align_for::<SoaTransform>()?;
            Ok(SkeletonResolver {
                joint_rest_poses: ArchivedVec::serialize_from_slice(self.joint_rest_poses(), serializer)?,
                joint_names: ArchivedVec::serialize_from_iter(
                    self.joint_names().iter().map(|(key, value)| Entry { key, value }),
                    serializer,
                )?,
                joint_parents: ArchivedVec::serialize_from_slice(self.joint_parents(), serializer)?,
            })
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<Skeleton, D> for ArchivedSkeleton {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<Skeleton, D::Error> {
            let archived = from_archived!(self);
            let mut skeleton = Skeleton::new(SkeletonMeta {
                version: Skeleton::version(),
                num_joints: archived.num_joints,
                joint_names: BiHashMap::default(),
                joint_parents: Vec::new(),
            });
            skeleton
                .joint_rest_poses_mut()
                .copy_from_slice(archived.joint_rest_poses.as_slice());

            skeleton.joint_names = JointHashMap::with_capacity_and_hashers(
                archived.joint_names.len(),
                DeterministicState::new(),
                DeterministicState::new(),
            );
            for entry in archived.joint_names.iter() {
                skeleton.joint_names.insert(entry.key.to_string(), entry.value);
            }

            skeleton
                .joint_parents_mut()
                .copy_from_slice(archived.joint_parents.as_slice());
            Ok(skeleton)
        }
    }
};

#[cfg(feature = "serde")]
const _: () = {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Skeleton {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let raw = self.to_raw();
            raw.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Skeleton {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Skeleton, D::Error> {
            let raw = SkeletonRaw::deserialize(deserializer)?;
            Ok(Skeleton::from_raw(&raw))
        }
    }
};

#[cfg(test)]
mod tests {
    use std::simd::prelude::*;
    use wasm_bindgen_test::*;

    use super::*;
    use crate::math::{SoaQuat, SoaVec3};

    #[allow(clippy::excessive_precision)]
    #[test]
    #[wasm_bindgen_test]
    fn test_read_skeleton() {
        let skeleton = Skeleton::from_path("./resource/playback/skeleton.ozz").unwrap();

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
        assert_eq!(skeleton.joint_by_name("Hips"), Some(0));
        assert_eq!(skeleton.joint_by_name("Bip01 R Toe0Nub"), Some(66));
    }

    #[cfg(feature = "rkyv")]
    #[test]
    #[wasm_bindgen_test]
    fn test_rkyv_skeleton() {
        use rkyv::ser::Serializer;
        use rkyv::Deserialize;

        let skeleton = Skeleton::from_path("./resource/playback/skeleton.ozz").unwrap();
        let mut serializer = rkyv::ser::serializers::AllocSerializer::<30720>::default();
        serializer.serialize_value(&skeleton).unwrap();
        let buf = serializer.into_serializer().into_inner();
        let archived = unsafe { rkyv::archived_root::<Skeleton>(&buf) };
        let mut deserializer = rkyv::Infallible;
        let skeleton2: Skeleton = archived.deserialize(&mut deserializer).unwrap();

        assert_eq!(skeleton.joint_rest_poses(), skeleton2.joint_rest_poses());
        assert_eq!(skeleton.joint_parents(), skeleton2.joint_parents());
        assert_eq!(skeleton.joint_names(), skeleton2.joint_names());
    }

    #[cfg(feature = "serde")]
    #[test]
    #[wasm_bindgen_test]
    fn test_serde_skeleton() {
        use serde_json;

        let skeleton = Skeleton::from_path("./resource/blend/skeleton.ozz").unwrap();
        let josn = serde_json::to_vec(&skeleton).unwrap();
        let skeleton2: Skeleton = serde_json::from_slice(&josn).unwrap();

        assert_eq!(skeleton.joint_rest_poses(), skeleton2.joint_rest_poses());
        assert_eq!(skeleton.joint_parents(), skeleton2.joint_parents());
        assert_eq!(skeleton.joint_names(), skeleton2.joint_names());
    }
}
