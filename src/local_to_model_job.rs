use glam::Mat4;
use std::cell::RefCell;
use std::rc::Rc;

use crate::base::{OzzBuf, OzzError, OzzRef, SKELETON_MAX_JOINTS, SKELETON_NO_PARENT};
use crate::math::{AosMat4, SoaMat4, SoaTransform};
use crate::skeleton::Skeleton;

#[derive(Debug)]
pub struct LocalToModelJob<S = Rc<Skeleton>, I = Rc<RefCell<Vec<SoaTransform>>>, O = Rc<RefCell<Vec<Mat4>>>>
where
    S: OzzRef<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzBuf<Mat4>,
{
    skeleton: Option<S>,
    input: Option<I>,
    output: Option<O>,
    verified: bool,
    root: AosMat4,
    from: i32,
    to: i32,
    from_excluded: bool,
}

impl<S, I, O> Default for LocalToModelJob<S, I, O>
where
    S: OzzRef<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzBuf<Mat4>,
{
    fn default() -> LocalToModelJob<S, I, O> {
        return LocalToModelJob {
            skeleton: None,
            input: None,
            output: None,
            verified: false,
            root: AosMat4::identity(),
            from: SKELETON_NO_PARENT,
            to: SKELETON_MAX_JOINTS,
            from_excluded: false,
        };
    }
}

impl<S, I, O> LocalToModelJob<S, I, O>
where
    S: OzzRef<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzBuf<Mat4>,
{
    pub fn new() -> LocalToModelJob {
        return LocalToModelJob::default();
    }

    pub fn skeleton(&self) -> Option<&S> {
        return self.skeleton.as_ref();
    }

    pub fn set_skeleton(&mut self, skeleton: S) {
        self.verified = false;
        self.skeleton = Some(skeleton);
    }

    pub fn clear_skeleton(&mut self) {
        self.verified = false;
        self.skeleton = None;
    }

    pub fn input(&self) -> Option<&I> {
        return self.input.as_ref();
    }

    pub fn set_input(&mut self, input: I) {
        self.verified = false;
        self.input = Some(input);
    }

    pub fn clear_input(&mut self) {
        self.verified = false;
        self.input = None;
    }

    pub fn output(&self) -> Option<&O> {
        return self.output.as_ref();
    }

    pub fn set_output(&mut self, output: O) {
        self.verified = false;
        self.output = Some(output);
    }

    pub fn clear_output(&mut self) {
        self.verified = false;
        self.output = None;
    }

    pub fn root(&self) -> Mat4 {
        return self.root.into();
    }

    pub fn set_root(&mut self, root: &Mat4) {
        self.root = (*root).into();
    }

    pub fn from(&self) -> i32 {
        return self.from;
    }

    pub fn set_from(&mut self, from: i32) {
        self.from = from;
    }

    pub fn to(&self) -> i32 {
        return self.to;
    }

    pub fn set_to(&mut self, to: i32) {
        self.to = to;
    }

    pub fn from_excluded(&self) -> bool {
        return self.from_excluded;
    }

    pub fn set_from_excluded(&mut self, from_excluded: bool) {
        self.from_excluded = from_excluded;
    }

    pub fn validate(&self) -> bool {
        let skeleton = match &self.skeleton {
            Some(skeleton) => skeleton,
            None => return false,
        };

        let input = match self.input.as_ref() {
            Some(input) => match input.vec() {
                Ok(input) => input,
                Err(_) => return false,
            },
            None => return false,
        };
        if input.len() < skeleton.num_soa_joints() {
            return false;
        }

        let output = match self.output.as_ref() {
            Some(output) => match output.vec() {
                Ok(output) => output,
                Err(_) => return false,
            },
            None => return false,
        };
        if output.len() < skeleton.num_joints() {
            return false;
        }

        return true;
    }

    pub fn run(&mut self) -> Result<(), OzzError> {
        if !self.verified {
            if !self.validate() {
                return Err(OzzError::InvalidJob);
            }
            self.verified = true;
        }

        let skeleton = self.skeleton.as_ref().unwrap();
        let input = self.input.as_ref().unwrap().vec()?;
        let mut output = self.output.as_mut().unwrap().vec_mut()?;

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
                if parent == Skeleton::no_parent() {
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
    use glam::{Mat4, Vec3};
    use std::collections::HashMap;

    use super::*;
    use crate::archive::{ArchiveReader, IArchive};
    use crate::base::DeterministicState;
    use crate::math::{SoaQuat, SoaVec3};
    use crate::skeleton::Skeleton;

    #[test]
    fn test_validity() {
        let mut archive = IArchive::new("./resource/playback/skeleton.ozz").unwrap();
        let skeleton = Rc::new(Skeleton::read(&mut archive).unwrap());
        let num_joints = skeleton.num_joints();

        // empty skeleton
        let job: LocalToModelJob = LocalToModelJob::default();
        assert!(!job.validate());

        // empty input
        let mut job: LocalToModelJob = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_output(Rc::new(RefCell::new(vec![Mat4::IDENTITY; num_joints])));
        assert!(!job.validate());

        // empty output
        let mut job: LocalToModelJob = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_input(Rc::new(RefCell::new(vec![SoaTransform::default(); num_joints])));
        assert!(!job.validate());

        // invalid input
        let mut job = LocalToModelJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_input(Rc::new(RefCell::new(vec![SoaTransform::default(); 1])));
        job.set_output(Rc::new(RefCell::new(vec![Mat4::IDENTITY; num_joints + 10])));
        assert!(!job.validate());

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
        return Rc::new(Skeleton {
            joint_rest_poses: vec![
                SoaTransform {
                    translation: SoaVec3::splat_col([0.0; 3]),
                    rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                    scale: SoaVec3::splat_col([0.0; 3]),
                };
                2
            ],
            joint_parents: vec![-1, 0, 1, 0, 3, 3],
            joint_names: (|| {
                let mut map = HashMap::with_hasher(DeterministicState::new());
                map.insert("j0".into(), 0);
                map.insert("j1".into(), 1);
                map.insert("j2".into(), 2);
                map.insert("j3".into(), 3);
                map.insert("j4".into(), 4);
                map.insert("j5".into(), 5);
                return map;
            })(),
        });
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
        return Rc::new(Skeleton {
            joint_rest_poses: vec![
                SoaTransform {
                    translation: SoaVec3::splat_col([0.0; 3]),
                    rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                    scale: SoaVec3::splat_col([0.0; 3]),
                };
                2
            ],
            joint_parents: vec![-1, 0, 1, 0, 3, 4, 3, -1],
            joint_names: (|| {
                let mut map = HashMap::with_hasher(DeterministicState::new());
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
        });
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
