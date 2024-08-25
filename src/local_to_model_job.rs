//!
//! Local to Model Job.
//!

use glam::Mat4;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::base::{OzzBuf, OzzError, OzzIndex, OzzMutBuf, OzzObj, SKELETON_MAX_JOINTS, SKELETON_NO_PARENT};
use crate::math::{AosMat4, SoaMat4, SoaTransform};
use crate::skeleton::Skeleton;

///
/// Computes model-space joint matrices from local-space `SoaTransform`.
///
/// This job uses the skeleton to define joints parent-child hierarchy. The job
/// iterates through all joints to compute their transform relatively to the
/// skeleton root.
///
/// Job inputs is an array of SoaTransform objects (in local-space), ordered like
/// skeleton's joints. Job output is an array of matrices (in model-space),
/// ordered like skeleton's joints. Output are matrices, because the combination
/// of affine transformations can contain shearing or complex transformation
/// that cannot be represented as Transform object.
///
#[derive(Debug)]
pub struct LocalToModelJob<S = Rc<Skeleton>, I = Rc<RefCell<Vec<SoaTransform>>>, O = Rc<RefCell<Vec<Mat4>>>>
where
    S: OzzObj<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzMutBuf<Mat4>,
{
    skeleton: Option<S>,
    input: Option<I>,
    root: AosMat4,
    from: i32,
    to: i32,
    from_excluded: bool,
    output: Option<O>,
}

pub type LocalToModelJobRef<'t> = LocalToModelJob<&'t Skeleton, &'t [SoaTransform], &'t mut [Mat4]>;
pub type LocalToModelJobRc = LocalToModelJob<Rc<Skeleton>, Rc<RefCell<Vec<SoaTransform>>>, Rc<RefCell<Vec<Mat4>>>>;
pub type LocalToModelJobArc = LocalToModelJob<Arc<Skeleton>, Arc<RwLock<Vec<SoaTransform>>>, Arc<RwLock<Vec<Mat4>>>>;

impl<S, I, O> Default for LocalToModelJob<S, I, O>
where
    S: OzzObj<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzMutBuf<Mat4>,
{
    fn default() -> LocalToModelJob<S, I, O> {
        return LocalToModelJob {
            skeleton: None,
            input: None,
            root: AosMat4::identity(),
            from: SKELETON_NO_PARENT,
            to: SKELETON_MAX_JOINTS,
            from_excluded: false,
            output: None,
        };
    }
}

impl<S, I, O> LocalToModelJob<S, I, O>
where
    S: OzzObj<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzMutBuf<Mat4>,
{
    /// Gets skeleton of `LocalToModelJob`.
    #[inline]
    pub fn skeleton(&self) -> Option<&S> {
        return self.skeleton.as_ref();
    }

    /// Sets skeleton of `LocalToModelJob`.
    ///
    /// The Skeleton object describing the joint hierarchy used for local to model space conversion.
    #[inline]
    pub fn set_skeleton(&mut self, skeleton: S) {
        self.skeleton = Some(skeleton);
    }

    /// Clears skeleton of `LocalToModelJob`.
    #[inline]
    pub fn clear_skeleton(&mut self) {
        self.skeleton = None;
    }

    /// Gets input of `LocalToModelJob`.
    #[inline]
    pub fn input(&self) -> Option<&I> {
        return self.input.as_ref();
    }

    /// Sets input of `LocalToModelJob`.
    ///
    /// The input range that store local transforms.
    #[inline]
    pub fn set_input(&mut self, input: I) {
        self.input = Some(input);
    }

    /// Clears input of `LocalToModelJob`.
    #[inline]
    pub fn clear_input(&mut self) {
        self.input = None;
    }

    /// Gets root of `LocalToModelJob`.
    #[inline]
    pub fn root(&self) -> Mat4 {
        return self.root.into();
    }

    /// Sets root of `LocalToModelJob`.
    ///
    /// The root matrix will multiply to every model space matrices, None means an identity matrix.
    /// This can be used to directly compute world-space transforms for example.
    #[inline]
    pub fn set_root(&mut self, root: &Mat4) {
        self.root = (*root).into();
    }

    /// Gets from of `LocalToModelJob`.
    #[inline]
    pub fn from(&self) -> i32 {
        return self.from;
    }

    /// Sets from of `LocalToModelJob`.
    ///
    /// Defines "from" which joint the local-to-model conversion should start.
    ///
    /// Default value is `SKELETON_NO_PARENT`, meaning the whole hierarchy is updated.
    ///
    /// This parameter can be used to optimize update by limiting conversion to part of the joint hierarchy.
    /// Note that "from" parent should be a valid matrix, as it is going to be used as part of "from" joint
    /// hierarchy update.
    #[inline]
    pub fn set_from(&mut self, from: impl OzzIndex) {
        self.from = from.i32();
    }

    /// Gets to of `LocalToModelJob`.
    #[inline]
    pub fn to(&self) -> i32 {
        return self.to;
    }

    /// Sets to of `LocalToModelJob`.
    ///
    /// Defines "to" which joint the local-to-model conversion should go, "to" included.
    /// Update will end before "to" joint is reached if "to" is not partof the hierarchy starting from "from".
    ///
    /// Default value is `SKELETON_MAX_JOINTS`, meaning the hierarchy (starting from "from") is updated to
    /// the last joint.
    #[inline]
    pub fn set_to(&mut self, to: impl OzzIndex) {
        self.to = to.i32();
    }

    /// Gets from_excluded of `LocalToModelJob`.
    #[inline]
    pub fn from_excluded(&self) -> bool {
        return self.from_excluded;
    }

    /// Sets from_excluded of `LocalToModelJob`.
    ///
    /// If `true`, "from" joint is not updated during job execution. Update starts with all children of "from".
    ///
    /// Default value is `false`.
    ///
    /// This can be used to update a model-space transform independently from the local-space one.
    /// To do so: set "from" joint model-space transform matrix, and run this Job with "from_excluded" to update
    /// all "from" children.
    #[inline]
    pub fn set_from_excluded(&mut self, from_excluded: bool) {
        self.from_excluded = from_excluded;
    }

    /// Gets output of `LocalToModelJob`.
    #[inline]
    pub fn output(&self) -> Option<&O> {
        return self.output.as_ref();
    }

    /// Sets output of `LocalToModelJob`.
    ///
    /// The output range to be filled with model-space matrices.
    #[inline]
    pub fn set_output(&mut self, output: O) {
        self.output = Some(output);
    }

    /// Clears output of `LocalToModelJob`.
    #[inline]
    pub fn clear_output(&mut self) {
        self.output = None;
    }

    /// Validates `LocalToModelJob` parameters.
    pub fn validate(&self) -> bool {
        return (|| {
            let skeleton = self.skeleton.as_ref()?.obj();
            let input = self.input.as_ref()?.buf().ok()?;
            let output = self.output.as_ref()?.buf().ok()?;

            let mut ok = input.len() >= skeleton.num_soa_joints();
            ok &= output.len() >= skeleton.num_joints();
            return Some(ok);
        })()
        .unwrap_or(false);
    }

    /// Runs local to model job's task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        let skeleton = self.skeleton.as_ref().ok_or(OzzError::InvalidJob)?.obj();
        let input = self.input.as_ref().ok_or(OzzError::InvalidJob)?.buf()?;
        let mut output = self.output.as_mut().ok_or(OzzError::InvalidJob)?.mut_buf()?;

        let mut ok = input.len() >= skeleton.num_soa_joints();
        ok &= output.len() >= skeleton.num_joints();
        if !ok {
            return Err(OzzError::InvalidJob);
        }

        let begin = i32::max(0, self.from + (self.from_excluded as i32)) as usize;
        let end = i32::max(0, i32::min(self.to + 1, skeleton.num_joints() as i32)) as usize;

        let mut idx = begin;
        let mut process = idx < end && (!self.from_excluded || skeleton.joint_parent(idx) as i32 >= self.from);

        while process {
            let transform = &input[idx / 4];
            let soa_matrices = SoaMat4::from_affine(&transform.translation, &transform.rotation, &transform.scale);
            let aos_matrices = soa_matrices.to_aos();

            let soa_end = (idx + 4) & !3;
            while idx < soa_end && process {
                let parent = skeleton.joint_parent(idx);
                if parent as i32 == SKELETON_NO_PARENT {
                    output[idx] = AosMat4::mul(&self.root, &aos_matrices[idx & 3]).into();
                } else {
                    output[idx] = AosMat4::mul(&output[parent as usize].into(), &aos_matrices[idx & 3]).into();
                }

                idx += 1;
                process = idx < end && skeleton.joint_parent(idx) as i32 >= self.from;
            }
        }

        return Ok(());
    }
}

#[cfg(test)]
mod local_to_model_tests {
    use glam::Vec3;
    use wasm_bindgen_test::*;

    use super::*;
    use crate::base::DeterministicState;
    use crate::math::{SoaQuat, SoaVec3};
    use crate::skeleton::{JointHashMap, SkeletonRaw};

    #[test]
    #[wasm_bindgen_test]
    fn test_validity() {
        let skeleton = Rc::new(Skeleton::from_path("./resource/playback/skeleton.ozz").unwrap());
        let num_joints = skeleton.num_joints();

        // empty skeleton
        let mut job: LocalToModelJob = LocalToModelJob::default();
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // empty input
        let mut job: LocalToModelJob = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_output(Rc::new(RefCell::new(vec![Mat4::IDENTITY; num_joints])));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // empty output
        let mut job: LocalToModelJob = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_input(Rc::new(RefCell::new(vec![SoaTransform::default(); num_joints])));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid input
        let mut job = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_input(Rc::new(RefCell::new(vec![SoaTransform::default(); 1])));
        job.set_output(Rc::new(RefCell::new(vec![Mat4::IDENTITY; num_joints + 10])));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid output
        let mut job = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_input(Rc::new(RefCell::new(vec![SoaTransform::default(); num_joints + 10])));
        job.set_output(Rc::new(RefCell::new(vec![Mat4::IDENTITY; 1])));

        let mut job = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_input(Rc::new(RefCell::new(vec![SoaTransform::default(); num_joints])));
        job.set_output(Rc::new(RefCell::new(vec![Mat4::IDENTITY; num_joints])));

        // bad from
        job.from = SKELETON_MAX_JOINTS;
        assert!(job.run().is_ok());
        job.from = -SKELETON_MAX_JOINTS;
        assert!(job.run().is_ok());

        // bad to
        job.to = SKELETON_MAX_JOINTS;
        assert!(job.run().is_ok());
        job.to = -SKELETON_MAX_JOINTS;
        assert!(job.run().is_ok());
    }

    fn new_skeleton1() -> Rc<Skeleton> {
        // 6 joints
        //   j0
        //  /  \
        // j1  j3
        //  |  / \
        // j2 j4 j5
        return Rc::new(Skeleton::from_raw(&SkeletonRaw {
            joint_rest_poses: vec![
                SoaTransform {
                    translation: SoaVec3::splat_col([0.0; 3]),
                    rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                    scale: SoaVec3::splat_col([0.0; 3]),
                };
                2
            ],
            joint_names: (|| {
                let mut map = JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new());
                map.insert("j0".into(), 0);
                map.insert("j1".into(), 1);
                map.insert("j2".into(), 2);
                map.insert("j3".into(), 3);
                map.insert("j4".into(), 4);
                map.insert("j5".into(), 5);
                return map;
            })(),
            joint_parents: vec![-1, 0, 1, 0, 3, 3],
        }));
    }

    fn new_input1() -> Rc<RefCell<Vec<SoaTransform>>> {
        return Rc::new(RefCell::new(vec![
            SoaTransform {
                translation: SoaVec3::new([2.0, 0.0, 1.0, -2.0], [2.0, 0.0, 2.0, -2.0], [2.0, 0.0, 4.0, -2.0]),
                rotation: SoaQuat::new(
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.70710677, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.70710677, 1.0, 1.0],
                ),
                scale: SoaVec3::new([1.0, 1.0, 1.0, 10.0], [1.0, 1.0, 1.0, 10.0], [1.0, 1.0, 1.0, 10.0]),
            },
            SoaTransform {
                translation: SoaVec3::new([12.0, 0.0, 0.0, 0.0], [46.0, 0.0, 0.0, 0.0], [-12.0, 0.0, 0.0, 0.0]),
                rotation: SoaQuat::new(
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ),
                scale: SoaVec3::new([1.0, -0.1, 1.0, 1.0], [1.0, -0.1, 1.0, 1.0], [1.0, -0.1, 1.0, 1.0]),
            },
        ]));
    }

    fn new_skeleton2() -> Rc<Skeleton> {
        // 6 joints
        //       *
        //     /   \
        //   j0    j7
        //  /  \
        // j1  j3
        //  |  / \
        // j2 j4 j6
        //     |
        //    j5
        return Rc::new(Skeleton::from_raw(&SkeletonRaw {
            joint_rest_poses: vec![
                SoaTransform {
                    translation: SoaVec3::splat_col([0.0; 3]),
                    rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                    scale: SoaVec3::splat_col([0.0; 3]),
                };
                2
            ],
            joint_names: (|| {
                let mut map = JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new());
                map.insert("j0".into(), 0);
                map.insert("j1".into(), 1);
                map.insert("j2".into(), 2);
                map.insert("j3".into(), 3);
                map.insert("j4".into(), 4);
                map.insert("j5".into(), 5);
                map.insert("j6".into(), 6);
                map.insert("j7".into(), 7);
                return map;
            })(),
            joint_parents: vec![-1, 0, 1, 0, 3, 4, 3, -1],
        }));
    }

    fn new_input2() -> Rc<RefCell<Vec<SoaTransform>>> {
        return Rc::new(RefCell::new(vec![
            SoaTransform {
                translation: SoaVec3::new([2.0, 0.0, -2.0, 1.0], [2.0, 0.0, -2.0, 2.0], [2.0, 0.0, -2.0, 4.0]),
                rotation: SoaQuat::new(
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.70710677, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.70710677, 1.0, 1.0],
                ),
                scale: SoaVec3::new([1.0, 1.0, 10.0, 1.0], [1.0, 1.0, 10.0, 1.0], [1.0, 1.0, 10.0, 1.0]),
            },
            SoaTransform {
                translation: SoaVec3::new([12.0, 0.0, 3.0, 6.0], [46.0, 0.0, 4.0, 7.0], [-12.0, 0.0, 5.0, 8.0]),
                rotation: SoaQuat::new(
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ),
                scale: SoaVec3::new([1.0, -0.1, 1.0, 1.0], [1.0, -0.1, 1.0, 1.0], [1.0, -0.1, 1.0, 1.0]),
            },
        ]));
    }

    fn execute_test(
        skeleton: &Rc<Skeleton>,
        input: &Rc<RefCell<Vec<SoaTransform>>>,
        output: &Rc<RefCell<Vec<Mat4>>>,
        root: Option<Mat4>,
        from: Option<i32>,
        to: Option<i32>,
        from_excluded: bool,
        expected: &[Mat4],
        message: &str,
    ) {
        let mut job = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_input(input.clone());
        job.set_output(output.clone());
        job.set_root(&root.unwrap_or(Mat4::IDENTITY));
        job.from = from.unwrap_or(SKELETON_NO_PARENT);
        job.to = to.unwrap_or(SKELETON_MAX_JOINTS);
        job.from_excluded = from_excluded;

        job.run().unwrap();
        for idx in 0..skeleton.num_joints() {
            let a = output.as_ref().borrow()[idx];
            let b = expected[idx];
            assert!(a.abs_diff_eq(b, 2e-6f32), "{} joint={} {} {}", message, idx, a, b,);
        }
    }

    #[test]
    #[wasm_bindgen_test]
    #[rustfmt::skip]
    fn test_transformation() {
        let skeleton = new_skeleton1();
        let input = new_input1();

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 6]));
        execute_test(&skeleton, &input, &output, None, None, None, false, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 4.0, 1.0, 1.0]),
            Mat4::from_cols_array(&[10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 120.0, 460.0, -120.0, 1.0]),
            Mat4::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "transformation default");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 6]));
        let root = Mat4::from_translation(Vec3::new(4.0, 3.0, 2.0));
        execute_test(&skeleton, &input, &output, Some(root), None, None, false, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 5.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 5.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 7.0, 3.0, 1.0]),
            Mat4::from_cols_array(&[10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 4.0, 3.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 124.0, 463.0, -118.0, 1.0]),
            Mat4::from_cols_array(&[-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 4.0, 3.0, 2.0, 1.0]),
        ], "transformation root");
    }

    #[test]
    #[wasm_bindgen_test]
    #[rustfmt::skip]
    fn test_from_to() {
        let skeleton = new_skeleton2();
        let input = new_input2();

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        execute_test(&skeleton, &input, &output, None, None, None, false, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[-0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0]),
        ], "from_to from=* to=*");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(2), false, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to from=0 to=2");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(6), false, &vec![
            Mat4::from_cols_array(&[ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[ 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[ 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0]),
            Mat4::from_cols_array(&[ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[ -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0]),
            Mat4::from_cols_array(&[ 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to from=0 to=6");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(46), false, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[-0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to from=0 to=46");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(-99), false, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to from=0 to=-99");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        execute_test(&skeleton, &input, &output, None, Some(93), None, false, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to from=* to=93");
    }

    #[test]
    #[wasm_bindgen_test]
    #[rustfmt::skip]
    fn test_from_to_exclude() {
        let skeleton = new_skeleton2();
        let input = new_input2();

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        execute_test(&skeleton, &input, &output, None, None, None, true, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[-0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0]),
        ], "from_to_exclude from=* to=*");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        output.borrow_mut()[0] = Mat4::from_scale(Vec3::splat(2.0));
        execute_test(&skeleton, &input, &output, None, Some(0), None, true, &vec![
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[0.0, 0.0, -20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, -4.0, -4.0, 4.0, 1.0]),
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 4.0, 8.0, 1.0]),
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 26.0, 96.0, -16.0, 1.0]),
            Mat4::from_cols_array(&[-0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 26.0, 96.0, -16.0, 1.0]),
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 8.0, 12.0, 18.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to_exclude from=0 to=*");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        output.borrow_mut()[1] = Mat4::from_scale(Vec3::splat(2.0));
        execute_test(&skeleton, &input, &output, None, Some(1), None, true, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, -4.0, -4.0, -4.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to_exclude from=1 to=*");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        output.borrow_mut()[2] = Mat4::from_scale(Vec3::splat(2.0));
        execute_test(&skeleton, &input, &output, None, Some(2), None, true, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to_exclude from=2 to=*");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        output.borrow_mut()[7] = Mat4::from_scale(Vec3::splat(2.0));
        execute_test(&skeleton, &input, &output, None, Some(7), None, true, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to_exclude from=7 to=*");

        let output = Rc::new(RefCell::new(vec![Mat4::IDENTITY; 8]));
        output.borrow_mut()[6] = Mat4::from_scale(Vec3::splat(2.0));
        execute_test(&skeleton, &input, &output, None, Some(6), None, true, &vec![
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        ], "from_to_exclude from=6 to=*");
    }
}
