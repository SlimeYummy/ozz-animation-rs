use crate::math::{ozz_matrix4_mul, ozz_matrix4_new, OzzNumber, OzzTransform};
use crate::skeleton::Skeleton;
use nalgebra::Matrix4;
use std::cell::RefCell;
use std::rc::Rc;

pub struct LocalToModelJob<N: OzzNumber> {
    skeleton: Rc<Skeleton<N>>,
    input: Rc<RefCell<Vec<OzzTransform<N>>>>,
    output: Rc<RefCell<Vec<Matrix4<N>>>>,
    root: Matrix4<N>,
}

impl<N: OzzNumber> LocalToModelJob<N> {
    pub fn new(
        skeleton: Rc<Skeleton<N>>,
        input: Rc<RefCell<Vec<OzzTransform<N>>>>,
    ) -> LocalToModelJob<N> {
        let num_joints = skeleton.num_joints();
        return LocalToModelJob {
            skeleton,
            input,
            output: Rc::new(RefCell::new(vec![Matrix4::identity(); num_joints])),
            root: Matrix4::identity(),
        };
    }

    pub fn output(&self) -> Rc<RefCell<Vec<Matrix4<N>>>> {
        return self.output.clone();
    }

    pub fn run(&mut self) {
        let input = self.input.borrow();
        let mut output = self.output.borrow_mut();

        let ske = self.skeleton.clone();

        for idx in 0..ske.num_joints() {
            let matrix = ozz_matrix4_new(
                &input[idx].translation,
                &input[idx].rotation,
                &input[idx].scale,
            );

            let parent = ske.joint_parents()[idx];
            if parent == Skeleton::<N>::no_parent() {
                output[idx] = ozz_matrix4_mul(&self.root, &matrix);
            } else {
                output[idx] = ozz_matrix4_mul(&output[parent as usize], &matrix);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::Animation;
    use crate::approx::assert_abs_diff_eq;
    use crate::archive::{ArchiveReader, IArchive};
    use crate::sampling_job::SamplingJob;
    use crate::skeleton::Skeleton;
    use crate::test_helper::read_chunk;

    #[test]
    fn test_local_to_model_job_run() {
        let mut archive = IArchive::new("./test_files/animation-simple.ozz").unwrap();
        let animation = Rc::new(Animation::read(&mut archive).unwrap());

        let mut archive = IArchive::new("./test_files/skeleton-simple.ozz").unwrap();
        let skeleton = Rc::new(Skeleton::read(&mut archive).unwrap());

        let mut sampling_job = SamplingJob::new(animation);
        let mut local_to_model_job = LocalToModelJob::new(skeleton, sampling_job.output());

        let mut counter = 0;
        let mut ratio = 0f32;
        while ratio <= 1.0f32 {
            counter += 1;
            ratio += 0.005;
            sampling_job.run(ratio);
            local_to_model_job.run();

            let file_no = match counter {
                1 => 1,
                100 => 2,
                200 => 3,
                _ => continue,
            };

            let file = format!("./test_files/local_to_model/output_{}", file_no);
            let chunk: Vec<Matrix4<f32>> = read_chunk(&file).unwrap();
            let output = local_to_model_job.output.borrow();
            for idx in 0..output.len() {
                assert_abs_diff_eq!(chunk[idx], output[idx], epsilon = 0.0000005);
            }
        }
    }
}
