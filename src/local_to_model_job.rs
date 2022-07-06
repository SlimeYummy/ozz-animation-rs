use crate::base::{OzzBuf, OzzBufX, OzzRes, OzzResX, SKELETON_MAX_JOINTS, SKELETON_NO_PARENT};
use crate::math::{ozz_matrix4_mul, ozz_matrix4_new, OzzNumber, OzzTransform};
use crate::skeleton::Skeleton;
use anyhow::{anyhow, Result};
use nalgebra::Matrix4;
use std::cell::RefCell;
use std::rc::Rc;

pub struct LocalToModelJob<N: OzzNumber> {
    skeleton: OzzResX<Skeleton<N>>,
    input: OzzBufX<OzzTransform<N>>,
    output: OzzBufX<Matrix4<N>>,
    root: Matrix4<N>,
    from: i32,
    to: i32,
    from_excluded: bool,
}

impl<N: OzzNumber> Default for LocalToModelJob<N> {
    fn default() -> LocalToModelJob<N> {
        return LocalToModelJob {
            skeleton: None,
            input: None,
            output: None,
            root: Matrix4::identity(),
            from: SKELETON_NO_PARENT,
            to: SKELETON_MAX_JOINTS,
            from_excluded: false,
        };
    }
}

impl<N: OzzNumber> LocalToModelJob<N> {
    pub fn new() -> LocalToModelJob<N> {
        return LocalToModelJob::default();
    }

    pub fn skeleton(&self) -> OzzResX<Skeleton<N>> {
        return self.skeleton.clone();
    }

    pub fn set_skeleton(&mut self, skeleton: &OzzRes<Skeleton<N>>) {
        self.skeleton = Some(skeleton.clone());
    }

    pub fn reset_skeleton(&mut self) {
        self.skeleton = None;
    }

    pub fn input(&self) -> OzzBufX<OzzTransform<N>> {
        return self.input.clone();
    }

    pub fn set_input(&mut self, input: &OzzBuf<OzzTransform<N>>) {
        self.input = Some(input.clone());
    }

    pub fn reset_input(&mut self) {
        self.input = None;
    }

    pub fn output(&self) -> OzzBufX<Matrix4<N>> {
        return self.output.clone();
    }

    pub fn set_output(&mut self, output: &Rc<RefCell<Vec<Matrix4<N>>>>) {
        self.output = Some(output.clone());
    }

    pub fn reset_output(&mut self) {
        self.output = None;
    }

    pub fn root(&self) -> Matrix4<N> {
        return self.root;
    }

    pub fn set_root(&mut self, root: Matrix4<N>) {
        self.root = root;
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
            Some(input) => input.as_ref().borrow(),
            None => return false,
        };
        if input.len() < skeleton.num_joints() {
            return false;
        }

        let output = match self.output.as_ref() {
            Some(output) => output.as_ref().borrow(),
            None => return false,
        };
        if output.len() < skeleton.num_joints() {
            return false;
        }

        return true;
    }

    pub fn run(&mut self) -> Result<()> {
        if !self.validate() {
            return Err(anyhow!("Invalid LocalToModelJob"));
        }

        let skeleton = self.skeleton.as_ref().unwrap();
        let input = &self.input.as_ref().unwrap().as_ref().borrow();
        let mut output = self.output.as_ref().unwrap().borrow_mut();

        let begin = i32::max(0, self.from + (self.from_excluded as i32)) as usize;
        let end = i32::max(0, i32::min(self.to + 1, skeleton.num_joints() as i32)) as usize;

        let mut idx = begin;
        if idx >= end {
            return Ok(());
        }
        if self.from_excluded && (skeleton.joint_parent(idx) as i32) < self.from {
            return Ok(());
        }

        loop {
            let matrix = ozz_matrix4_new(
                &input[idx].translation,
                &input[idx].rotation,
                &input[idx].scale,
            );

            let parent = skeleton.joint_parent(idx);
            if parent == Skeleton::<N>::no_parent() {
                output[idx] = ozz_matrix4_mul(&self.root, &matrix);
            } else {
                output[idx] = ozz_matrix4_mul(&output[parent as usize], &matrix);
            }

            idx += 1;
            if idx >= end || (skeleton.joint_parent(idx) as i32) < self.from {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod local_to_model_tests {
    use super::*;
    use crate::approx::abs_diff_eq;
    use crate::archive::{ArchiveReader, IArchive};
    use crate::skeleton::Skeleton;
    use maplit::hashmap;
    use nalgebra::{Quaternion, Vector3};

    #[test]
    fn test_validity() {
        let mut archive = IArchive::new("./test_files/skeleton-simple.ozz").unwrap();
        let skeleton = Rc::new(Skeleton::<f32>::read(&mut archive).unwrap());
        let num_joints = skeleton.num_joints();

        // empty skeleton
        let job = LocalToModelJob::<f32>::default();
        assert!(!job.validate());

        // empty input
        let mut job = LocalToModelJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.set_output(&Rc::new(RefCell::new(vec![
            Matrix4::identity();
            num_joints
        ])));
        assert!(!job.validate());

        // empty output
        let mut job = LocalToModelJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.set_input(&Rc::new(RefCell::new(vec![
            OzzTransform::default();
            num_joints
        ])));
        assert!(!job.validate());

        // invalid input
        let mut job = LocalToModelJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.set_input(&Rc::new(RefCell::new(vec![OzzTransform::default(); 1])));
        job.set_output(&Rc::new(RefCell::new(vec![
            Matrix4::identity();
            num_joints + 10
        ])));
        assert!(!job.validate());

        // invalid output
        let mut job = LocalToModelJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.set_input(&Rc::new(RefCell::new(vec![
            OzzTransform::default();
            num_joints + 10
        ])));
        job.set_output(&Rc::new(RefCell::new(vec![Matrix4::identity(); 1])));

        let mut job = LocalToModelJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.set_input(&Rc::new(RefCell::new(vec![
            OzzTransform::default();
            num_joints
        ])));
        job.set_output(&Rc::new(RefCell::new(vec![
            Matrix4::identity();
            num_joints
        ])));

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

    fn new_skeleton1() -> Rc<Skeleton<f32>> {
        // 6 joints
        //   j0
        //  /  \
        // j1  j3
        //  |  / \
        // j2 j4 j5
        return Rc::new(Skeleton {
            joint_bind_poses: vec![
                OzzTransform {
                    translation: Vector3::new(0.0, 0.0, 0.0),
                    rotation: Quaternion::new(0.0, 0.0, 0.0, 1.0),
                    scale: Vector3::new(0.0, 0.0, 0.0),
                };
                6
            ],
            joint_parents: vec![-1, 0, 1, 0, 3, 3],
            joint_names: hashmap! {
                "j0".to_string() => 0,
                "j1".to_string() => 1,
                "j2".to_string() => 2,
                "j3".to_string() => 3,
                "j4".to_string() => 4,
                "j5".to_string() => 5,
            },
        });
    }

    fn new_input1() -> Rc<RefCell<Vec<OzzTransform<f32>>>> {
        return Rc::new(RefCell::new(vec![
            OzzTransform {
                translation: Vector3::new(2.0, 2.0, 2.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(0.0, 0.0, 0.0),
                rotation: Quaternion::new(0.70710677, 0.0, 0.70710677, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(1.0, 2.0, 4.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(-2.0, -2.0, -2.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(10.0, 10.0, 10.0),
            },
            OzzTransform {
                translation: Vector3::new(12.0, 46.0, -12.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(0.0, 0.0, 0.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(-0.1, -0.1, -0.1),
            },
        ]));
    }

    fn new_skeleton2() -> Rc<Skeleton<f32>> {
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
            joint_bind_poses: vec![
                OzzTransform {
                    translation: Vector3::new(0.0, 0.0, 0.0),
                    rotation: Quaternion::new(0.0, 0.0, 0.0, 1.0),
                    scale: Vector3::new(0.0, 0.0, 0.0),
                };
                8
            ],
            joint_parents: vec![-1, 0, 1, 0, 3, 4, 3, -1],
            joint_names: hashmap! {
                "j0".to_string() => 0,
                "j1".to_string() => 1,
                "j2".to_string() => 2,
                "j3".to_string() => 3,
                "j4".to_string() => 4,
                "j5".to_string() => 5,
                "j6".to_string() => 6,
                "j7".to_string() => 7,
            },
        });
    }

    fn new_input2() -> Rc<RefCell<Vec<OzzTransform<f32>>>> {
        return Rc::new(RefCell::new(vec![
            OzzTransform {
                translation: Vector3::new(2.0, 2.0, 2.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(0.0, 0.0, 0.0),
                rotation: Quaternion::new(0.70710677, 0.0, 0.70710677, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(-2.0, -2.0, -2.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(10.0, 10.0, 10.0),
            },
            OzzTransform {
                translation: Vector3::new(1.0, 2.0, 4.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(12.0, 46.0, -12.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(0.0, 0.0, 0.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(-0.1, -0.1, -0.1),
            },
            OzzTransform {
                translation: Vector3::new(3.0, 4.0, 5.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
            OzzTransform {
                translation: Vector3::new(6.0, 7.0, 8.0),
                rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
                scale: Vector3::new(1.0, 1.0, 1.0),
            },
        ]));
    }

    fn execute_test(
        skeleton: &Rc<Skeleton<f32>>,
        input: &Rc<RefCell<Vec<OzzTransform<f32>>>>,
        output: &Rc<RefCell<Vec<Matrix4<f32>>>>,
        root: Option<Matrix4<f32>>,
        from: Option<i32>,
        to: Option<i32>,
        from_excluded: bool,
        expected: &[Matrix4<f32>],
        message: &str,
    ) {
        let mut job = LocalToModelJob::default();
        job.set_skeleton(skeleton);
        job.set_input(input);
        job.set_output(output);
        job.root = root.unwrap_or(Matrix4::identity());
        job.from = from.unwrap_or(SKELETON_NO_PARENT);
        job.to = to.unwrap_or(SKELETON_MAX_JOINTS);
        job.from_excluded = from_excluded;

        job.run().unwrap();
        for idx in 0..skeleton.num_joints() {
            let out = output.as_ref().borrow()[idx];
            assert!(
                abs_diff_eq!(out, expected[idx].transpose(), epsilon = 0.000002),
                "{} joint={}",
                message,
                idx
            );
        }
    }

    #[test]
    #[rustfmt::skip]
    fn test_transformation() {
        let skeleton = new_skeleton1();
        let input = new_input1();

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 6]));
        execute_test(&skeleton, &input, &output, None, None, None, false, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 4.0, 1.0, 1.0),
            Matrix4::new(10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 120.0, 460.0, -120.0, 1.0),
            Matrix4::new(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "transformation default");

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 6]));
        let root = Matrix4::new_translation(&Vector3::new(4.0, 3.0, 2.0));
        execute_test(&skeleton, &input, &output, Some(root), None, None, false, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 5.0, 4.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 5.0, 4.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 10.0, 7.0, 3.0, 1.0),
            Matrix4::new(10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 4.0, 3.0, 2.0, 1.0),
            Matrix4::new(10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 124.0, 463.0, -118.0, 1.0),
            Matrix4::new(-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 4.0, 3.0, 2.0, 1.0),
        ], "transformation root");
    }

    #[test]
    #[rustfmt::skip]
    fn test_from_to() {
        let skeleton = new_skeleton2();
        let input = new_input2();

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        execute_test(&skeleton, &input, &output, None, None, None, false, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new(-0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0),
        ], "from_to from=* to=*");
        
        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(2), false, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to from=0 to=2");
        
        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(6), false, &vec![
            Matrix4::new( 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new( 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new( 0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0,10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0),
            Matrix4::new( 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0),
            Matrix4::new( 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new( -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0,0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new( 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0),
            Matrix4::new( 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to from=0 to=6");
        
        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(46), false, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new(-0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to from=0 to=46");
        
        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        execute_test(&skeleton, &input, &output, None, Some(0), Some(-99), false, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to from=0 to=-99");
        
        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        execute_test(&skeleton, &input, &output, None, Some(93), None, false, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to from=* to=93");
    }

    #[test]
    #[rustfmt::skip]
    fn test_from_to_exclude() {
        let skeleton = new_skeleton2();
        let input = new_input2();

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        execute_test(&skeleton, &input, &output, None, None, None, true, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.0),
            Matrix4::new(0.0, 0.0, -10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 4.0, 6.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new(-0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 15.0, 50.0, -6.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 8.0, 11.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 6.0, 7.0, 8.0, 1.0),
        ], "from_to_exclude from=* to=*");

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        output.borrow_mut()[0] = Matrix4::new_scaling(2.0);
        execute_test(&skeleton, &input, &output, None, Some(0), None, true, &vec![
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(0.0, 0.0, -20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, -4.0, -4.0, 4.0, 1.0),
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 4.0, 8.0, 1.0),
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 26.0, 96.0, -16.0, 1.0),
            Matrix4::new(-0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 26.0, 96.0, -16.0, 1.0),
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 8.0, 12.0, 18.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to_exclude from=0 to=*");

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        output.borrow_mut()[1] = Matrix4::new_scaling(2.0);
        execute_test(&skeleton, &input, &output, None, Some(1), None, true, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, -4.0, -4.0, -4.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to_exclude from=1 to=*");

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        output.borrow_mut()[2] = Matrix4::new_scaling(2.0);
        execute_test(&skeleton, &input, &output, None, Some(2), None, true, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to_exclude from=2 to=*");

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        output.borrow_mut()[7] = Matrix4::new_scaling(2.0);
        execute_test(&skeleton, &input, &output, None, Some(7), None, true, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to_exclude from=7 to=*");

        let output = Rc::new(RefCell::new(vec![Matrix4::identity(); 8]));
        output.borrow_mut()[6] = Matrix4::new_scaling(2.0);
        execute_test(&skeleton, &input, &output, None, Some(6), None, true, &vec![
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            Matrix4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ], "from_to_exclude from=6 to=*");
    }

    // #[test]
    // fn test_local_to_model_run() {
    //     let mut archive = IArchive::new("./test_files/animation-simple.ozz").unwrap();
    //     let animation = Rc::new(Animation::read(&mut archive).unwrap());

    //     let mut archive = IArchive::new("./test_files/skeleton-simple.ozz").unwrap();
    //     let skeleton = Rc::new(Skeleton::read(&mut archive).unwrap());

    //     let mut sampling_job = SamplingJob::new(animation);
    //     let output = Rc::new(RefCell::new(vec![
    //         Matrix4::identity();
    //         skeleton.num_joints()
    //     ]));
    //     let mut local_to_model_job =
    //         LocalToModelJob::new(skeleton, sampling_job.output(), output).unwrap();

    //     let mut counter = 0;
    //     let mut ratio = 0f32;
    //     while ratio <= 1.0f32 {
    //         counter += 1;
    //         ratio += 0.005;
    //         sampling_job.run(ratio);
    //         local_to_model_job.run();

    //         let file_no = match counter {
    //             1 => 1,
    //             100 => 2,
    //             200 => 3,
    //             _ => continue,
    //         };

    //         let file = format!("./test_files/local_to_model/output_{}", file_no);
    //         let chunk: Vec<Matrix4<f32>> = read_chunk(&file).unwrap();
    //         let output = local_to_model_job.output().borrow();
    //         for idx in 0..output.len() {
    //             assert_abs_diff_eq!(chunk[idx], output[idx], epsilon = 0.0000005);
    //         }
    //     }
    // }
}
