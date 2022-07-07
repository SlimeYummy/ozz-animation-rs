use crate::base::{OzzBuf, OzzBufX, OzzRes, OzzResX};
use crate::math::{ozz_quat_dot, ozz_quat_normalize, ozz_quat_xor_sign, OzzNumber, OzzTransform};
use crate::skeleton::Skeleton;
use anyhow::{anyhow, Result};
use nalgebra::{Quaternion, Vector3};
use std::iter;

#[derive(Debug, Clone)]
pub struct BlendingLayer<N: OzzNumber> {
    input: OzzBufX<OzzTransform<N>>,
    weight: N,
    joint_weights: Vec<N>,
}

#[derive(Debug)]
pub struct BlendingJob<N: OzzNumber> {
    skeleton: OzzResX<Skeleton<N>>,
    threshold: N,
    layers: Vec<BlendingLayer<N>>,
    additive_layers: Vec<BlendingLayer<N>>,
    output: OzzBufX<OzzTransform<N>>,

    num_passes: u32,
    num_partial_passes: u32,
    accumulated_weight: N,
    accumulated_weights: Vec<N>,
}

impl<N: OzzNumber> Default for BlendingJob<N> {
    fn default() -> BlendingJob<N> {
        return BlendingJob {
            skeleton: None,
            threshold: N::one() / N::from_i32(10).unwrap_or(N::one()), // todo: fix 0.1
            layers: Vec::new(),
            additive_layers: Vec::new(),
            output: None,

            num_passes: 0,
            num_partial_passes: 0,
            accumulated_weight: N::zero(),
            accumulated_weights: Vec::new(),
        };
    }
}

impl<N: OzzNumber> BlendingJob<N> {
    pub fn threshold(&self) -> N {
        return self.threshold;
    }

    pub fn set_threshold(&mut self, threshold: N) {
        self.threshold = threshold;
    }

    pub fn skeleton(&self) -> OzzResX<Skeleton<N>> {
        return self.skeleton.clone();
    }

    pub fn set_skeleton(&mut self, skeleton: &OzzRes<Skeleton<N>>) {
        let joint_bind_poses = skeleton.joint_bind_poses().len();
        if self.accumulated_weights.len() < joint_bind_poses {
            self.accumulated_weights.resize(joint_bind_poses, N::zero());
        }
        self.skeleton = Some(skeleton.clone());
    }

    pub fn reset_skeleton(&mut self) {
        self.skeleton = None;
    }

    pub fn output(&self) -> OzzBufX<OzzTransform<N>> {
        return self.output.clone();
    }

    pub fn set_output(&mut self, output: &OzzBuf<OzzTransform<N>>) {
        self.output = Some(output.clone());
    }

    pub fn reset_output(&mut self) {
        self.output = None;
    }

    pub fn layers(&self) -> &[BlendingLayer<N>] {
        return &self.layers;
    }

    pub fn layers_mut(&mut self) -> &mut Vec<BlendingLayer<N>> {
        return &mut self.layers;
    }

    pub fn additive_layers(&self) -> &[BlendingLayer<N>] {
        return &self.additive_layers;
    }

    pub fn additive_layers_mut(&mut self) -> &mut Vec<BlendingLayer<N>> {
        return &mut self.additive_layers;
    }

    pub fn validate(&self) -> bool {
        let skeleton = match &self.skeleton {
            Some(skeleton) => skeleton,
            None => return false,
        };

        if self.threshold <= N::zero() {
            return false;
        }

        let output = match self.output.as_ref() {
            Some(output) => output.as_ref().borrow(),
            None => return false,
        };
        if output.len() < skeleton.joint_bind_poses().len() {
            return false;
        }

        fn validate_layer<N: OzzNumber>(layer: &BlendingLayer<N>, joint_bind_poses: usize) -> bool {
            let input = match layer.input.as_ref() {
                Some(input) => input.as_ref().borrow(),
                None => return false,
            };
            if input.len() < joint_bind_poses {
                return false;
            }

            if !layer.joint_weights.is_empty() {
                return layer.joint_weights.len() >= joint_bind_poses;
            }
            return true;
        }

        return !iter::empty()
            .chain(&self.layers)
            .chain(&self.additive_layers)
            .any(|l| !validate_layer(l, skeleton.joint_bind_poses().len()));
    }

    pub fn run(&mut self) -> Result<()> {
        if !self.validate() {
            return Err(anyhow!("Invalid SamplingJob"));
        }

        self.num_partial_passes = 0;
        self.num_passes = 0;
        self.accumulated_weight = N::zero();

        self.blend_layers();
        self.blend_bind_pose();
        self.normalize();
        self.add_layers();
        return Ok(());
    }

    fn blend_layers(&mut self) {
        let mut output = self.output.as_ref().unwrap().borrow_mut();
        let bind_pose = self.skeleton.as_ref().unwrap().joint_bind_poses();

        for layer in &self.layers {
            let input = layer.input.as_ref().unwrap().borrow();
            if layer.weight <= N::zero() {
                continue;
            }
            self.accumulated_weight += layer.weight;

            if !layer.joint_weights.is_empty() {
                self.num_partial_passes += 1;
                if self.num_passes == 0 {
                    for idx in 0..bind_pose.len() {
                        let weight = layer.weight * N::max(layer.joint_weights[idx], N::zero());
                        self.accumulated_weights[idx] = weight;
                        Self::blend_1st_pass(&input[idx], weight, &mut output[idx]);
                    }
                } else {
                    for idx in 0..bind_pose.len() {
                        let weight = layer.weight * N::max(layer.joint_weights[idx], N::zero());
                        self.accumulated_weights[idx] += weight;
                        Self::blend_n_pass(&input[idx], weight, &mut output[idx]);
                    }
                }
                self.num_passes += 1;
            } else {
                if self.num_passes == 0 {
                    for idx in 0..bind_pose.len() {
                        self.accumulated_weights[idx] = layer.weight;
                        Self::blend_1st_pass(&input[idx], layer.weight, &mut output[idx]);
                    }
                } else {
                    for idx in 0..bind_pose.len() {
                        self.accumulated_weights[idx] += layer.weight;
                        Self::blend_n_pass(&input[idx], layer.weight, &mut output[idx]);
                    }
                }
                self.num_passes += 1;
            }
        }
    }

    fn blend_bind_pose(&mut self) {
        let mut output = self.output.as_ref().unwrap().borrow_mut();
        let joint_bind_poses = self.skeleton.as_ref().unwrap().joint_bind_poses();

        if self.num_partial_passes == 0 {
            let bp_weight = self.threshold - self.accumulated_weight;
            if bp_weight > N::zero() {
                if self.num_passes == 0 {
                    self.accumulated_weight = N::one();
                    for idx in 0..joint_bind_poses.len() {
                        output[idx] = joint_bind_poses[idx];
                    }
                } else {
                    self.accumulated_weight = self.threshold;
                    for idx in 0..joint_bind_poses.len() {
                        Self::blend_n_pass(&joint_bind_poses[idx], bp_weight, &mut output[idx]);
                    }
                }
            }
        } else {
            for idx in 0..joint_bind_poses.len() {
                let bp_weight = N::max(self.threshold - self.accumulated_weights[idx], N::zero());
                self.accumulated_weights[idx] =
                    N::max(self.threshold, self.accumulated_weights[idx]);
                Self::blend_n_pass(&joint_bind_poses[idx], bp_weight, &mut output[idx]);
            }
        }
    }

    fn normalize(&mut self) {
        let mut output = self.output.as_ref().unwrap().borrow_mut();
        let joint_bind_poses = self.skeleton.as_ref().unwrap().joint_bind_poses();

        if self.num_partial_passes == 0 {
            let ratio = self.accumulated_weight.recip();
            for idx in 0..joint_bind_poses.len() {
                output[idx].translation.scale_mut(ratio);
                output[idx].rotation = ozz_quat_normalize(&output[idx].rotation);
                output[idx].scale.scale_mut(ratio);
            }
        } else {
            for idx in 0..joint_bind_poses.len() {
                let ratio = self.accumulated_weights[idx].recip();
                output[idx].translation.scale_mut(ratio);
                output[idx].rotation = ozz_quat_normalize(&output[idx].rotation);
                output[idx].scale.scale_mut(ratio);
            }
        }
    }

    fn add_layers(&mut self) {
        let mut output = self.output.as_ref().unwrap().borrow_mut();
        let joint_bind_poses = self.skeleton.as_ref().unwrap().joint_bind_poses();

        for layer in &self.additive_layers {
            let input = layer.input.as_ref().unwrap().borrow();

            if layer.weight > N::zero() {
                if !layer.joint_weights.is_empty() {
                    for idx in 0..joint_bind_poses.len() {
                        let weight = layer.weight * N::max(layer.joint_weights[idx], N::zero());
                        Self::blend_add_pass(&input[idx], weight, &mut output[idx]);
                    }
                } else {
                    for idx in 0..joint_bind_poses.len() {
                        Self::blend_add_pass(&input[idx], layer.weight, &mut output[idx]);
                    }
                }
            } else if layer.weight < N::zero() {
                if !layer.joint_weights.is_empty() {
                    for idx in 0..joint_bind_poses.len() {
                        let weight = (-layer.weight) * N::max(layer.joint_weights[idx], N::zero());
                        Self::blend_sub_pass(&input[idx], weight, &mut output[idx]);
                    }
                } else {
                    for idx in 0..joint_bind_poses.len() {
                        Self::blend_sub_pass(&input[idx], -layer.weight, &mut output[idx]);
                    }
                }
            }
        }
    }

    fn blend_1st_pass(input: &OzzTransform<N>, weight: N, output: &mut OzzTransform<N>) {
        output.translation = input.translation.scale(weight);
        output.rotation.coords = input.rotation.coords.scale(weight);
        output.scale = input.scale.scale(weight);
    }

    fn blend_n_pass(input: &OzzTransform<N>, weight: N, output: &mut OzzTransform<N>) {
        output.translation += input.translation.scale(weight);
        let dot = ozz_quat_dot(&output.rotation, &input.rotation);
        let rotation = ozz_quat_xor_sign(dot.is_positive(), &input.rotation);
        output.rotation.coords += rotation.coords.scale(weight);
        output.scale += input.scale.scale(weight);
    }

    fn blend_add_pass(input: &OzzTransform<N>, weight: N, output: &mut OzzTransform<N>) {
        output.translation += input.translation.scale(weight);

        let rotation = ozz_quat_xor_sign(input.rotation.w.is_positive(), &input.rotation);
        let interp_quat = Quaternion::new(
            (rotation.w - N::one()) * weight + N::one(),
            rotation.i * weight,
            rotation.j * weight,
            rotation.k * weight,
        );
        let norm_interp_quat = ozz_quat_normalize(&interp_quat);
        output.rotation = norm_interp_quat * output.rotation;

        let one_minus_weight = Vector3::from_element(N::one() - weight);
        let tmp_weight = one_minus_weight + input.scale.scale(weight);
        output.scale = Vector3::component_mul(&output.scale, &tmp_weight);
    }

    fn blend_sub_pass(input: &OzzTransform<N>, weight: N, output: &mut OzzTransform<N>) {
        output.translation -= input.translation.scale(weight);

        let rotation = ozz_quat_xor_sign(input.rotation.w.is_positive(), &input.rotation);
        let interp_quat = Quaternion::new(
            (rotation.w - N::one()) * weight + N::one(),
            rotation.i * weight,
            rotation.j * weight,
            rotation.k * weight,
        );
        let norm_interp_quat = ozz_quat_normalize(&interp_quat);
        output.rotation = norm_interp_quat.conjugate() * output.rotation;

        let one_min_weight = N::one() - weight;
        let rcp_scale = Vector3::new(
            (input.scale.x * weight + one_min_weight).recip(),
            (input.scale.y * weight + one_min_weight).recip(),
            (input.scale.z * weight + one_min_weight).recip(),
        );
        output.scale = Vector3::component_mul(&output.scale, &rcp_scale);
    }
}

#[cfg(test)]
mod blending_tests {
    use super::*;
    use crate::approx::abs_diff_eq;
    use crate::archive::{ArchiveReader, IArchive};
    use crate::base::{ozz_buf, ozz_buf_x, ozz_res};
    use crate::skeleton::Skeleton;
    use std::collections::HashMap;
    use std::ops::Neg;

    #[test]
    fn test_validity() {
        let mut archive = IArchive::new("./test_files/skeleton-blending.ozz").unwrap();
        let skeleton = ozz_res(Skeleton::<f32>::read(&mut archive).unwrap());
        let num_bind_pose = skeleton.joint_bind_poses().len();
        let default_layer = BlendingLayer::<f32> {
            input: ozz_buf_x(vec![OzzTransform::default(); num_bind_pose]),
            weight: 0.5,
            joint_weights: Vec::new(),
        };

        // empty/default job
        let job = BlendingJob::<f32>::default();
        assert!(!job.validate());

        // invalid output
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        assert!(!job.validate());

        // layers are optional
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(job.validate());

        // invalid layer input, too small
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.layers_mut().push(default_layer.clone());
        job.layers_mut().push(BlendingLayer {
            input: ozz_buf_x(vec![]),
            weight: 0.5,
            joint_weights: Vec::new(),
        });
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(!job.validate());

        // invalid output range, smaller output
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.layers_mut().push(default_layer.clone());
        job.set_output(&ozz_buf(vec![OzzTransform::default(); 3]));
        assert!(!job.validate());

        // invalid threshold
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.set_threshold(0.0);
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(!job.validate());

        // invalid joint weights range
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.layers_mut().push(BlendingLayer {
            input: ozz_buf_x(vec![OzzTransform::default(); num_bind_pose]),
            weight: 0.5,
            joint_weights: vec![0.5; 1],
        });
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(!job.validate());

        // valid job
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.layers_mut().push(default_layer.clone());
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(job.validate());

        // valid joint weights range
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.layers_mut().push(default_layer.clone());
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(job.validate());

        // valid job, bigger output
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.layers_mut().push(default_layer.clone());
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose + 5]));
        assert!(job.validate());

        // valid additive job, no normal blending
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.additive_layers_mut().push(default_layer.clone());
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(job.validate());

        // valid additive job, with normal blending
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.layers_mut().push(default_layer.clone());
        job.additive_layers_mut().push(default_layer.clone());
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(job.validate());

        // invalid layer input range, too small
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.additive_layers_mut().push(BlendingLayer {
            input: ozz_buf_x(vec![OzzTransform::default(); 3]),
            weight: 0.5,
            joint_weights: Vec::new(),
        });
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(!job.validate());

        // valid additive job, with per-joint weights
        let mut job = BlendingJob::<f32>::default();
        job.set_skeleton(&skeleton);
        job.additive_layers_mut().push(BlendingLayer {
            input: ozz_buf_x(vec![OzzTransform::default(); num_bind_pose]),
            weight: 0.5,
            joint_weights: vec![0.5; num_bind_pose],
        });
        job.set_output(&ozz_buf(vec![OzzTransform::default(); num_bind_pose]));
        assert!(job.validate());
    }

    fn new_layers1() -> Vec<BlendingLayer<f32>> {
        let mut input1 = vec![OzzTransform::default(); 8];
        input1[0].translation = Vector3::new(0.0, 4.0, 8.0);
        input1[1].translation = Vector3::new(1.0, 5.0, 9.0);
        input1[2].translation = Vector3::new(2.0, 6.0, 10.0);
        input1[3].translation = Vector3::new(3.0, 7.0, 11.0);
        input1[4].translation = Vector3::new(12.0, 16.0, 20.0);
        input1[5].translation = Vector3::new(13.0, 17.0, 21.0);
        input1[6].translation = Vector3::new(14.0, 18.0, 22.0);
        input1[7].translation = Vector3::new(15.0, 19.0, 23.0);

        let mut input2 = input1.clone();
        input2.iter_mut().for_each(|transform| {
            transform.translation = transform.translation.neg();
        });

        return vec![
            BlendingLayer {
                input: ozz_buf_x(input1),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
            BlendingLayer {
                input: ozz_buf_x(input2),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
        ];
    }

    fn new_layers2() -> Vec<BlendingLayer<f32>> {
        let input1 = ozz_buf(vec![OzzTransform::<f32>::default(); 4]);
        input1.borrow_mut()[0].translation = Vector3::new(2.0, 6.0, 10.0);
        input1.borrow_mut()[1].translation = Vector3::new(3.0, 7.0, 11.0);
        input1.borrow_mut()[2].translation = Vector3::new(4.0, 8.0, 12.0);
        input1.borrow_mut()[3].translation = Vector3::new(5.0, 9.0, 13.0);

        let input2 = ozz_buf(vec![OzzTransform::<f32>::default(); 4]);
        input2.borrow_mut()[0].translation = Vector3::new(3.0, 7.0, 11.0);
        input2.borrow_mut()[1].translation = Vector3::new(4.0, 8.0, 12.0);
        input2.borrow_mut()[2].translation = Vector3::new(5.0, 9.0, 13.0);
        input2.borrow_mut()[3].translation = Vector3::new(6.0, 10.0, 14.0);

        return vec![
            BlendingLayer {
                input: Some(input1.clone()),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
            BlendingLayer {
                input: Some(input2.clone()),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
        ];
    }

    fn execute_test(
        skeleton: &OzzRes<Skeleton<f32>>,
        layers: Vec<BlendingLayer<f32>>,
        additive_layers: Vec<BlendingLayer<f32>>,
        expected_translations: Option<Vec<Vector3<f32>>>,
        expected_rotations: Option<Vec<Quaternion<f32>>>,
        expected_scales: Option<Vec<Vector3<f32>>>,
        message: &str,
    ) {
        let mut job = BlendingJob::default();
        job.set_skeleton(&skeleton);
        *job.layers_mut() = layers;
        *job.additive_layers_mut() = additive_layers;
        let joint_bind_poses = skeleton.joint_bind_poses().len();
        let output = ozz_buf(vec![OzzTransform::default(); joint_bind_poses]);
        job.set_output(&output);
        job.run().unwrap();

        for idx in 0..joint_bind_poses {
            let out = output.as_ref().borrow()[idx];

            if let Some(translations) = &expected_translations {
                assert!(
                    abs_diff_eq!(out.translation, translations[idx], epsilon = 0.000002),
                    "{} => translation idx:{} actual:{:?}, excepted:{:?}",
                    message,
                    idx,
                    out.translation,
                    translations[idx],
                );
            }
            if let Some(rotations) = &expected_rotations {
                assert!(
                    abs_diff_eq!(out.rotation, rotations[idx], epsilon = 0.0001),
                    "{} => rotation idx:{} actual:{:?}, excepted:{:?}",
                    message,
                    idx,
                    out.rotation,
                    rotations[idx],
                );
            }
            if let Some(scales) = &expected_scales {
                assert!(
                    abs_diff_eq!(out.scale, scales[idx], epsilon = 0.000002),
                    "{} => scale idx:{} actual:{:?}, excepted:{:?}",
                    message,
                    idx,
                    out.scale,
                    scales[idx],
                );
            }
        }
    }

    #[test]
    fn test_empty() {
        let mut joint_bind_poses = vec![
            OzzTransform {
                translation: Vector3::new(0.0, 4.0, 8.0),
                rotation: Quaternion::new(0.0, 0.0, 0.0, 1.0),
                scale: Vector3::new(0.0, 40.0, 80.0),
            },
            OzzTransform {
                translation: Vector3::new(1.0, 5.0, 9.0),
                rotation: Quaternion::new(0.0, 0.0, 0.0, 1.0),
                scale: Vector3::new(10.0, 50.0, 90.0),
            },
            OzzTransform {
                translation: Vector3::new(2.0, 6.0, 10.0),
                rotation: Quaternion::new(0.0, 0.0, 0.0, 1.0),
                scale: Vector3::new(20.0, 60.0, 100.0),
            },
            OzzTransform {
                translation: Vector3::new(3.0, 7.0, 11.0),
                rotation: Quaternion::new(0.0, 0.0, 0.0, 1.0),
                scale: Vector3::new(30.0, 70.0, 110.0),
            },
        ];
        joint_bind_poses.extend_from_within(0..4);
        joint_bind_poses.iter_mut().skip(4).for_each(|x| {
            x.translation = x.translation.scale(2.0);
            x.scale = x.scale.scale(2.0);
        });

        let skeleton = ozz_res(Skeleton {
            joint_bind_poses,
            joint_parents: vec![],
            joint_names: HashMap::new(),
        });

        execute_test(
            &skeleton,
            Vec::new(),
            Vec::new(),
            Some(vec![
                Vector3::new(0.0, 4.0, 8.0),
                Vector3::new(1.0, 5.0, 9.0),
                Vector3::new(2.0, 6.0, 10.0),
                Vector3::new(3.0, 7.0, 11.0),
                Vector3::new(0.0, 8.0, 16.0),
                Vector3::new(2.0, 10.0, 18.0),
                Vector3::new(4.0, 12.0, 20.0),
                Vector3::new(6.0, 14.0, 22.0),
            ]),
            None,
            Some(vec![
                Vector3::new(0.0, 40.0, 80.0),
                Vector3::new(10.0, 50.0, 90.0),
                Vector3::new(20.0, 60.0, 100.0),
                Vector3::new(30.0, 70.0, 110.0),
                Vector3::new(0.0, 80.0, 160.0),
                Vector3::new(20.0, 100.0, 180.0),
                Vector3::new(40.0, 120.0, 200.0),
                Vector3::new(60.0, 140.0, 220.0),
            ]),
            "empty",
        );
    }

    #[test]
    fn test_weight() {
        let mut joint_bind_poses = vec![OzzTransform::default(); 8];
        joint_bind_poses[0].scale = Vector3::new(0.0, 4.0, 8.0);
        joint_bind_poses[1].scale = Vector3::new(1.0, 5.0, 9.0);
        joint_bind_poses[2].scale = Vector3::new(2.0, 6.0, 10.0);
        joint_bind_poses[3].scale = Vector3::new(3.0, 7.0, 11.0);
        joint_bind_poses[4].scale = joint_bind_poses[0].scale.scale(2.0);
        joint_bind_poses[5].scale = joint_bind_poses[1].scale.scale(2.0);
        joint_bind_poses[6].scale = joint_bind_poses[2].scale.scale(2.0);
        joint_bind_poses[7].scale = joint_bind_poses[3].scale.scale(2.0);
        let skeleton = ozz_res(Skeleton {
            joint_bind_poses,
            joint_parents: vec![],
            joint_names: HashMap::new(),
        });

        let mut layers = new_layers1();

        layers[0].weight = -0.07;
        layers[1].weight = 1.0;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![
                Vector3::new(-0.0, -4.0, -8.0),
                Vector3::new(-1.0, -5.0, -9.0),
                Vector3::new(-2.0, -6.0, -10.0),
                Vector3::new(-3.0, -7.0, -11.0),
                Vector3::new(-12.0, -16.0, -20.0),
                Vector3::new(-13.0, -17.0, -21.0),
                Vector3::new(-14.0, -18.0, -22.0),
                Vector3::new(-15.0, -19.0, -23.0),
            ]),
            None,
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 8]),
            "weight - 1",
        );

        layers[0].weight = 1.0;
        layers[1].weight = 1.0e-23f32;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![
                Vector3::new(0.0, 4.0, 8.0),
                Vector3::new(1.0, 5.0, 9.0),
                Vector3::new(2.0, 6.0, 10.0),
                Vector3::new(3.0, 7.0, 11.0),
                Vector3::new(12.0, 16.0, 20.0),
                Vector3::new(13.0, 17.0, 21.0),
                Vector3::new(14.0, 18.0, 22.0),
                Vector3::new(15.0, 19.0, 23.0),
            ]),
            None,
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 8]),
            "weight - 2",
        );

        layers[0].weight = 0.5;
        layers[1].weight = 0.5;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![Vector3::new(0.0, 0.0, 0.0); 8]),
            None,
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 8]),
            "weight - 3",
        );
    }

    #[test]
    fn test_joint_weights() {
        let mut joint_bind_poses = vec![OzzTransform::default(); 8];
        joint_bind_poses[0].translation = Vector3::new(10.0, 14.0, 18.0);
        joint_bind_poses[1].translation = Vector3::new(11.0, 15.0, 19.0);
        joint_bind_poses[2].translation = Vector3::new(12.0, 16.0, 20.0);
        joint_bind_poses[3].translation = Vector3::new(13.0, 17.0, 21.0);
        joint_bind_poses[0].scale = Vector3::new(0.0, 4.0, 8.0);
        joint_bind_poses[1].scale = Vector3::new(1.0, 5.0, 9.0);
        joint_bind_poses[2].scale = Vector3::new(2.0, 6.0, 10.0);
        joint_bind_poses[3].scale = Vector3::new(3.0, 7.0, 11.0);
        joint_bind_poses[4].scale = joint_bind_poses[0].scale.scale(2.0);
        joint_bind_poses[5].scale = joint_bind_poses[1].scale.scale(2.0);
        joint_bind_poses[6].scale = joint_bind_poses[2].scale.scale(2.0);
        joint_bind_poses[7].scale = joint_bind_poses[3].scale.scale(2.0);
        let skeleton = ozz_res(Skeleton {
            joint_bind_poses,
            joint_parents: vec![],
            joint_names: HashMap::new(),
        });

        let mut layers = new_layers1();
        layers[0].joint_weights = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        layers[1].joint_weights = vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        layers[0].weight = 0.5;
        layers[1].weight = 0.5;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(-2.0, -6.0, -10.0),
                Vector3::new(13.0, 17.0, 21.0),
                Vector3::new(12.0, 16.0, 20.0),
                Vector3::new(-13.0, -17.0, -21.0),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0),
            ]),
            None,
            Some(vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(3.0, 7.0, 11.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
            ]),
            "joint weight - 1",
        );

        layers[0].weight = 0.0;
        layers[1].weight = 1.0;
        execute_test(
            &skeleton,
            layers,
            Vec::new(),
            Some(vec![
                Vector3::new(0.0, -4.0, -8.0),
                Vector3::new(-1.0, -5.0, -9.0),
                Vector3::new(-2.0, -6.0, -10.0),
                Vector3::new(13.0, 17.0, 21.0),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(-13.0, -17.0, -21.0),
                Vector3::new(-14.0, -18.0, -22.0),
                Vector3::new(-15.0, -19.0, -23.0),
            ]),
            None,
            Some(vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(3.0, 7.0, 11.0),
                Vector3::new(0.0, 8.0, 16.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
            ]),
            "joint weight - 2",
        );
    }

    fn new_skeleton1() -> OzzRes<Skeleton<f32>> {
        let mut joint_bind_poses = vec![OzzTransform::default(); 4];
        joint_bind_poses[0].scale = Vector3::new(0.0, 4.0, 8.0);
        joint_bind_poses[1].scale = Vector3::new(1.0, 5.0, 9.0);
        joint_bind_poses[2].scale = Vector3::new(2.0, 6.0, 10.0);
        joint_bind_poses[3].scale = Vector3::new(3.0, 7.0, 11.0);

        return ozz_res(Skeleton {
            joint_bind_poses,
            joint_parents: vec![],
            joint_names: HashMap::new(),
        });
    }

    #[test]
    fn test_normalize() {
        let skeleton = new_skeleton1();
        let mut layers = new_layers2();

        let input1 = layers[0].input.as_ref().unwrap().clone();
        input1.borrow_mut()[0].rotation = Quaternion::new(0.70710677, 0.70710677, 0.0, 0.0);
        input1.borrow_mut()[1].rotation = Quaternion::identity();
        input1.borrow_mut()[2].rotation = Quaternion::new(0.70710677, 0.0, 0.70710677, 0.0);
        input1.borrow_mut()[3].rotation = Quaternion::new(0.9238795, 0.382683432, 0.0, 0.0);

        let input2 = layers[1].input.as_ref().unwrap().clone();
        input2.borrow_mut()[0].rotation = Quaternion::identity();
        input2.borrow_mut()[1].rotation = Quaternion::new(0.70710677, 0.70710677, 0.0, 0.0);
        input2.borrow_mut()[2].rotation = Quaternion::new(0.0, -0.70710677, 0.0, -0.70710677);
        input2.borrow_mut()[3].rotation = Quaternion::new(-0.9238795, -0.382683432, 0.0, 0.0);

        layers[0].weight = 0.2;
        layers[1].weight = 0.3;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![
                Vector3::new(2.6, 6.6, 10.6),
                Vector3::new(3.6, 7.6, 11.6),
                Vector3::new(4.6, 8.6, 12.6),
                Vector3::new(5.6, 9.6, 13.6),
            ]),
            Some(vec![
                Quaternion::new(0.95224595, 0.30507791, 0.0, 0.0),
                Quaternion::new(0.88906217, 0.45761687, 0.0, 0.0),
                Quaternion::new(0.39229235, -0.58843851, 0.39229235, -0.58843851),
                Quaternion::new(0.92387962, 0.38268352, 0.0, 0.0),
            ]),
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 4]),
            "normalize - 1",
        );

        input1.borrow_mut()[0].translation = Vector3::new(5.0, 25.0, 45.0);
        input1.borrow_mut()[1].translation = Vector3::new(10.0, 30.0, 50.0);
        input1.borrow_mut()[2].translation = Vector3::new(15.0, 35.0, 55.0);
        input1.borrow_mut()[3].translation = Vector3::new(20.0, 40.0, 60.0);
        input2.borrow_mut()[0].translation = Vector3::new(10.0, 30.0, 50.0);
        input2.borrow_mut()[1].translation = Vector3::new(15.0, 35.0, 55.0);
        input2.borrow_mut()[2].translation = Vector3::new(20.0, 40.0, 60.0);
        input2.borrow_mut()[3].translation = Vector3::new(25.0, 45.0, 65.0);

        layers[0].weight = 2.0;
        layers[1].weight = 3.0;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![
                Vector3::new(8.0, 28.0, 48.0),
                Vector3::new(13.0, 33.0, 53.0),
                Vector3::new(18.0, 38.0, 58.0),
                Vector3::new(23.0, 43.0, 63.0),
            ]),
            Some(vec![
                Quaternion::new(0.95224595, 0.30507791, 0.0, 0.0),
                Quaternion::new(0.88906217, 0.45761687, 0.0, 0.0),
                Quaternion::new(0.39229235, -0.58843851, 0.39229235, -0.58843851),
                Quaternion::new(0.92387962, 0.38268352, 0.0, 0.0),
            ]),
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 4]),
            "normalize - 2",
        );

        layers[1].joint_weights = vec![1.0, -1.0, 2.0, 0.1];
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![
                Vector3::new(8.0, 28.0, 48.0),
                Vector3::new(10.0, 30.0, 50.0),
                Vector3::new(150.0, 310.0, 470.0).scale(1.0 / 8.0),
                Vector3::new(47.5, 93.5, 139.5).scale(1.0 / 2.3),
            ]),
            None,
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 4]),
            "normalize - 3",
        );
    }

    #[test]
    fn test_threshold() {
        let skeleton = new_skeleton1();
        let mut layers = new_layers2();

        layers[0].weight = 0.04;
        layers[1].weight = 0.06;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![
                Vector3::new(2.6, 6.6, 10.6),
                Vector3::new(3.6, 7.6, 11.6),
                Vector3::new(4.6, 8.6, 12.6),
                Vector3::new(5.6, 9.6, 13.6),
            ]),
            Some(vec![Quaternion::identity(); 4]),
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 4]),
            "threshold - 1",
        );

        layers[0].weight = 1.0e-27;
        layers[1].weight = 0.0;
        execute_test(
            &skeleton,
            layers.clone(),
            Vec::new(),
            Some(vec![Vector3::zeros(); 4]),
            Some(vec![Quaternion::identity(); 4]),
            Some(vec![
                Vector3::new(0.0, 4.0, 8.0),
                Vector3::new(1.0, 5.0, 9.0),
                Vector3::new(2.0, 6.0, 10.0),
                Vector3::new(3.0, 7.0, 11.0),
            ]),
            "threshold - 2",
        );
    }

    fn new_layers3() -> Vec<BlendingLayer<f32>> {
        let mut input1 = vec![OzzTransform::<f32>::default(); 4];
        input1[0].translation = Vector3::new(0.0, 4.0, 8.0);
        input1[1].translation = Vector3::new(1.0, 5.0, 9.0);
        input1[2].translation = Vector3::new(2.0, 6.0, 10.0);
        input1[3].translation = Vector3::new(3.0, 7.0, 11.0);
        input1[0].rotation = Quaternion::new(0.70710677, 0.70710677, 0.0, 0.0);
        input1[1].rotation = Quaternion::identity();
        input1[2].rotation = Quaternion::new(-0.70710677, 0.0, 0.70710677, 0.0);
        input1[3].rotation = Quaternion::new(0.9238795, 0.382683432, 0.0, 0.0);
        input1[0].scale = Vector3::new(12.0, 16.0, 20.0);
        input1[1].scale = Vector3::new(13.0, 17.0, 21.0);
        input1[2].scale = Vector3::new(14.0, 18.0, 22.0);
        input1[3].scale = Vector3::new(15.0, 19.0, 23.0);

        let input2 = input1
            .iter()
            .map(|x| OzzTransform {
                translation: x.translation.neg(),
                rotation: x.rotation.conjugate(),
                scale: x.scale.neg(),
            })
            .collect::<Vec<_>>();

        return vec![
            BlendingLayer {
                input: ozz_buf_x(input1),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
            BlendingLayer {
                input: ozz_buf_x(input2),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
        ];
    }

    #[test]
    fn test_additive_weight() {
        let skeleton = ozz_res(Skeleton {
            joint_bind_poses: vec![OzzTransform::default(); 4],
            joint_parents: vec![],
            joint_names: HashMap::new(),
        });

        let mut layers = new_layers3();
        layers.pop();
        let input1 = layers[0].input.as_ref().unwrap().borrow().clone();

        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(vec![Vector3::zeros(); 4]),
            Some(vec![Quaternion::identity(); 4]),
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 4]),
            "additive weight - 1",
        );

        layers[0].weight = 0.5;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(input1.iter().map(|x| x.translation.scale(0.5)).collect()),
            Some(vec![
                Quaternion::new(0.9238795, 0.3826834, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Quaternion::new(0.9238795, 0.0, -0.3826834, 0.0),
                Quaternion::new(0.98078528, 0.19509032, 0.0, 0.0),
            ]),
            Some(vec![
                Vector3::new(6.5, 8.5, 10.5),
                Vector3::new(7.0, 9.0, 11.0),
                Vector3::new(7.5, 9.5, 11.5),
                Vector3::new(8.0, 10.0, 12.0),
            ]),
            "additive weight - 2",
        );

        layers[0].weight = 1.0;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(input1.iter().map(|x| x.translation).collect()),
            Some(input1.iter().map(|x| x.rotation).collect()),
            Some(input1.iter().map(|x| x.scale).collect()),
            "additive weight - 3",
        );

        let mut layers = new_layers3();
        let input2 = layers[1].input.as_ref().unwrap().borrow().clone();

        layers[0].weight = 0.0;
        layers[1].weight = 1.0;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(input2.iter().map(|x| x.translation).collect()),
            Some(input2.iter().map(|x| x.rotation).collect()),
            Some(input2.iter().map(|x| x.scale).collect()),
            "additive weight - 4",
        );

        layers[0].weight = 1.0;
        layers[1].weight = 1.0;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(vec![Vector3::zeros(); 4]),
            Some(vec![Quaternion::identity(); 4]),
            Some(vec![
                Vector3::new(-144.0, -256.0, -400.0),
                Vector3::new(-169.0, -289.0, -441.0),
                Vector3::new(-196.0, -324.0, -484.0),
                Vector3::new(-225.0, -361.0, -529.0),
            ]),
            "additive weight - 5",
        );

        layers[0].weight = 0.5;
        layers[1].input = ozz_buf_x(input1.clone());
        layers[1].weight = -0.5;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(vec![Vector3::zeros(); 4]),
            Some(vec![Quaternion::identity(); 4]),
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 4]),
            "additive weight - 6",
        );
    }

    #[test]
    fn test_additive_joint_weight() {
        let skeleton = ozz_res(Skeleton {
            joint_bind_poses: vec![OzzTransform::default(); 4],
            joint_parents: vec![],
            joint_names: HashMap::new(),
        });

        let mut layers = new_layers3();
        layers.pop();
        layers[0].joint_weights = vec![1.0, 0.5, 0.0, -1.0];

        layers[0].weight = 0.0;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(vec![Vector3::zeros(); 4]),
            Some(vec![Quaternion::identity(); 4]),
            Some(vec![Vector3::new(1.0, 1.0, 1.0); 4]),
            "additive joint weight - 1",
        );

        layers[0].weight = 0.5;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(vec![
                Vector3::new(0.0, 2.0, 4.0),
                Vector3::new(0.25, 1.25, 2.25),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0),
            ]),
            Some(vec![
                Quaternion::new(0.92387950, 0.3826834, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
            ]),
            Some(vec![
                Vector3::new(6.5, 8.5, 10.5),
                Vector3::new(4.0, 5.0, 6.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
            ]),
            "additive joint weight - 2",
        );

        layers[0].weight = 1.0;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(vec![
                Vector3::new(0.0, 4.0, 8.0),
                Vector3::new(0.5, 2.5, 4.5),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0),
            ]),
            Some(vec![
                Quaternion::new(0.70710677, 0.70710677, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
            ]),
            Some(vec![
                Vector3::new(12.0, 16.0, 20.0),
                Vector3::new(7.0, 9.0, 11.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
            ]),
            "additive joint weight - 3",
        );

        layers[0].weight = -1.0;
        execute_test(
            &skeleton,
            vec![],
            layers.clone(),
            Some(vec![
                Vector3::new(0.0, -4.0, -8.0),
                Vector3::new(-0.5, -2.5, -4.5),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 0.0),
            ]),
            Some(vec![
                Quaternion::new(0.70710677, -0.70710677, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
                Quaternion::new(1.0, 0.0, 0.0, 0.0),
            ]),
            Some(vec![
                Vector3::new(1.0 / 12.0, 1.0 / 16.0, 1.0 / 20.0),
                Vector3::new(1.0 / 7.0, 1.0 / 9.0, 1.0 / 11.0),
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(1.0, 1.0, 1.0),
            ]),
            "additive joint weight - 3",
        );
    }

    // #[test]
    // fn test_blending_job_run() {
    //     let mut archive = IArchive::new("./test_files/animation-blending-1.ozz").unwrap();
    //     let animation1 = Rc::new(Animation::read(&mut archive).unwrap());
    //     let mut archive = IArchive::new("./test_files/animation-blending-2.ozz").unwrap();
    //     let animation2 = Rc::new(Animation::read(&mut archive).unwrap());
    //     let mut archive = IArchive::new("./test_files/animation-blending-3.ozz").unwrap();
    //     let animation3 = Rc::new(Animation::read(&mut archive).unwrap());

    //     let mut archive = IArchive::new("./test_files/skeleton-blending.ozz").unwrap();
    //     let skeleton = Rc::new(Skeleton::read(&mut archive).unwrap());

    //     let mut sampling1_job = SamplingJob::new(animation1);
    //     let mut sampling2_job = SamplingJob::new(animation2);
    //     let mut sampling3_job = SamplingJob::new(animation3);
    //     let mut blending_job = BlendingJob::new(
    //         skeleton,
    //         vec![
    //             BlendingLayer {
    //                 input: sampling1_job.output(),
    //                 weight: 0.399999976f32,
    //                 joint_weights: Vec::new(),
    //             },
    //             BlendingLayer {
    //                 input: sampling2_job.output(),
    //                 weight: 0.600000024f32,
    //                 joint_weights: Vec::new(),
    //             },
    //             BlendingLayer {
    //                 input: sampling3_job.output(),
    //                 weight: 0.0f32,
    //                 joint_weights: Vec::new(),
    //             },
    //         ],
    //         0.1f32,
    //     );

    //     let mut counter = 0;
    //     let mut ratio = 0f32;
    //     while ratio <= 1.0f32 {
    //         counter += 1;
    //         ratio += 0.005;

    //         sampling1_job.run(ratio);
    //         sampling2_job.run(ratio);
    //         sampling3_job.run(ratio);

    //         let file_no = match counter {
    //             1 => 1,
    //             100 => 2,
    //             200 => 3,
    //             _ => continue,
    //         };

    //         let (num_passes, accumulated_weight) = blending_job.blend_layers();
    //         {
    //             let file = format!("./test_files/blending/blend_layers_{}", file_no);
    //             let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
    //             let output = blending_job.output.borrow();
    //             for idx in 0..output.len() {
    //                 assert_eq!(chunk[idx].translation, output[idx].translation);
    //                 assert_eq!(chunk[idx].rotation, output[idx].rotation);
    //                 assert_eq!(chunk[idx].scale, output[idx].scale);
    //             }
    //         }

    //         let accumulated_weight = blending_job.blend_bind_pose(num_passes, accumulated_weight);
    //         {
    //             let file = format!("./test_files/blending/blend_bind_pose_{}", file_no);
    //             let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
    //             let output = blending_job.output.borrow();
    //             for idx in 0..output.len() {
    //                 assert_eq!(chunk[idx].translation, output[idx].translation);
    //                 assert_eq!(chunk[idx].rotation, output[idx].rotation);
    //                 assert_eq!(chunk[idx].scale, output[idx].scale);
    //             }
    //         }

    //         blending_job.normalize(accumulated_weight);
    //         {
    //             let file = format!("./test_files/blending/normalize_{}", file_no);
    //             let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
    //             let output = blending_job.output.borrow();
    //             for idx in 0..output.len() {
    //                 assert_eq!(chunk[idx].translation, output[idx].translation);
    //                 assert_eq!(chunk[idx].rotation, output[idx].rotation);
    //                 assert_eq!(chunk[idx].scale, output[idx].scale);
    //             }
    //         }
    //     }
    // }
}
