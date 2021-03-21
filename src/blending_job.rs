use crate::math::{ozz_quat_dot, ozz_quat_normalize, OzzNumber, OzzTransform};
use crate::skeleton::Skeleton;
use std::cell::RefCell;
use std::rc::Rc;

pub struct BlendingLayer<N: OzzNumber> {
    input: Rc<RefCell<Vec<OzzTransform<N>>>>,
    weight: N,
    joint_weights: Vec<N>,
}

pub struct BlendingJob<N: OzzNumber> {
    skeleton: Rc<Skeleton<N>>,
    output: Rc<RefCell<Vec<OzzTransform<N>>>>,
    threshold: N,
    layers: Vec<BlendingLayer<N>>,
    // additive_layers: Vec<BlendingLayer>,
}

impl<N: OzzNumber> BlendingJob<N> {
    pub fn new(
        skeleton: Rc<Skeleton<N>>,
        layers: Vec<BlendingLayer<N>>,
        threshold: N,
    ) -> BlendingJob<N> {
        let num_joints = skeleton.num_joints();
        return BlendingJob {
            skeleton,
            output: Rc::new(RefCell::new(vec![Default::default(); num_joints])),
            threshold,
            layers,
        };
    }

    pub fn run(&mut self) {
        let (num_passes, accumulated_weight) = self.blend_layers();
        let accumulated_weight = self.blend_bind_pose(num_passes, accumulated_weight);
        self.normalize(accumulated_weight);
    }

    fn blend_layers(&mut self) -> (u32, N) {
        let mut output = self.output.borrow_mut();
        let bind_pose = self.skeleton.joint_bind_poses();

        let mut accumulated_weight = N::zero();
        let mut num_passes = 0;

        for layer in &self.layers {
            if layer.weight <= N::zero() {
                continue;
            }
            let input = layer.input.borrow();

            accumulated_weight += layer.weight;

            if !layer.joint_weights.is_empty() {
                // This layer has per-joint weights.
                panic!("not implement");
            } else {
                // This is a full layer.
                if num_passes == 0 {
                    for idx in 0..bind_pose.len() {
                        Self::blend_1st_pass(&input[idx], layer.weight, &mut output[idx]);
                    }
                } else {
                    for idx in 0..bind_pose.len() {
                        Self::blend_n_pass(&input[idx], layer.weight, &mut output[idx]);
                    }
                }
                num_passes += 1;
            }
        }

        return (num_passes, accumulated_weight);
    }

    fn blend_1st_pass(input: &OzzTransform<N>, weight: N, output: &mut OzzTransform<N>) {
        output.translation = input.translation.scale(weight);
        output.rotation.coords = input.rotation.coords.scale(weight);
        output.scale = input.scale.scale(weight);
    }

    fn blend_n_pass(input: &OzzTransform<N>, weight: N, output: &mut OzzTransform<N>) {
        output.translation += input.translation.scale(weight);
        let dot = ozz_quat_dot(&output.rotation, &input.rotation);
        let rotation = if dot > N::zero() {
            input.rotation
        } else {
            -input.rotation
        };
        output.rotation.coords += rotation.coords.scale(weight);
        output.scale += input.scale.scale(weight);
    }

    fn blend_bind_pose(&mut self, num_passes: u32, accumulated_weight: N) -> N {
        let mut output = self.output.borrow_mut();
        let bind_pose = self.skeleton.joint_bind_poses();

        let bp_weight = self.threshold - accumulated_weight;
        if bp_weight > N::zero() {
            if num_passes == 0 {
                for idx in 0..bind_pose.len() {
                    output[idx] = bind_pose[idx];
                }
                return accumulated_weight;
            } else {
                for idx in 0..bind_pose.len() {
                    Self::blend_n_pass(&bind_pose[idx], bp_weight, &mut output[idx]);
                }
                return self.threshold;
            }
        }
        return accumulated_weight;
    }

    fn normalize(&mut self, accumulated_weight: N) {
        let mut output = self.output.borrow_mut();

        let ratio = accumulated_weight.recip();
        for transform in output.iter_mut() {
            transform.translation.scale_mut(ratio);
            transform.rotation = ozz_quat_normalize(&transform.rotation);
            transform.scale.scale_mut(ratio);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::Animation;
    use crate::archive::{ArchiveReader, IArchive};
    use crate::sampling_job::SamplingJob;
    use crate::skeleton::Skeleton;
    use crate::test_helper::read_chunk;

    #[test]
    fn test_blending_job_run() {
        let mut archive = IArchive::new("./test_files/animation-blending-1.ozz").unwrap();
        let animation1 = Rc::new(Animation::read(&mut archive).unwrap());
        let mut archive = IArchive::new("./test_files/animation-blending-2.ozz").unwrap();
        let animation2 = Rc::new(Animation::read(&mut archive).unwrap());
        let mut archive = IArchive::new("./test_files/animation-blending-3.ozz").unwrap();
        let animation3 = Rc::new(Animation::read(&mut archive).unwrap());

        let mut archive = IArchive::new("./test_files/skeleton-blending.ozz").unwrap();
        let skeleton = Rc::new(Skeleton::read(&mut archive).unwrap());

        let mut sampling1_job = SamplingJob::new(animation1);
        let mut sampling2_job = SamplingJob::new(animation2);
        let mut sampling3_job = SamplingJob::new(animation3);
        let mut blending_job = BlendingJob::new(
            skeleton,
            vec![
                BlendingLayer {
                    input: sampling1_job.output(),
                    weight: 0.399999976f32,
                    joint_weights: Vec::new(),
                },
                BlendingLayer {
                    input: sampling2_job.output(),
                    weight: 0.600000024f32,
                    joint_weights: Vec::new(),
                },
                BlendingLayer {
                    input: sampling3_job.output(),
                    weight: 0.0f32,
                    joint_weights: Vec::new(),
                },
            ],
            0.1f32,
        );

        let mut counter = 0;
        let mut ratio = 0f32;
        while ratio <= 1.0f32 {
            counter += 1;
            ratio += 0.005;

            sampling1_job.run(ratio);
            sampling2_job.run(ratio);
            sampling3_job.run(ratio);

            let file_no = match counter {
                1 => 1,
                100 => 2,
                200 => 3,
                _ => continue,
            };

            let (num_passes, accumulated_weight) = blending_job.blend_layers();
            {
                let file = format!("./test_files/blending/blend_layers_{}", file_no);
                let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
                let output = blending_job.output.borrow();
                for idx in 0..output.len() {
                    assert_eq!(chunk[idx].translation, output[idx].translation);
                    assert_eq!(chunk[idx].rotation, output[idx].rotation);
                    assert_eq!(chunk[idx].scale, output[idx].scale);
                }
            }

            let accumulated_weight = blending_job.blend_bind_pose(num_passes, accumulated_weight);
            {
                let file = format!("./test_files/blending/blend_bind_pose_{}", file_no);
                let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
                let output = blending_job.output.borrow();
                for idx in 0..output.len() {
                    assert_eq!(chunk[idx].translation, output[idx].translation);
                    assert_eq!(chunk[idx].rotation, output[idx].rotation);
                    assert_eq!(chunk[idx].scale, output[idx].scale);
                }
            }

            blending_job.normalize(accumulated_weight);
            {
                let file = format!("./test_files/blending/normalize_{}", file_no);
                let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
                let output = blending_job.output.borrow();
                for idx in 0..output.len() {
                    assert_eq!(chunk[idx].translation, output[idx].translation);
                    assert_eq!(chunk[idx].rotation, output[idx].rotation);
                    assert_eq!(chunk[idx].scale, output[idx].scale);
                }
            }
        }
    }
}
