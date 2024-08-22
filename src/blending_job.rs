//!
//! Blending job.
//!

use glam::Vec4;
use std::cell::RefCell;
use std::rc::Rc;
use std::simd::prelude::*;
use std::sync::{Arc, RwLock};

use crate::base::{OzzBuf, OzzError, OzzMutBuf, OzzObj};
use crate::math::{fx4_from_vec4, fx4_sign, SoaQuat, SoaTransform, SoaVec3};
use crate::skeleton::Skeleton;

const ZERO: f32x4 = f32x4::from_array([0.0; 4]);
const ONE: f32x4 = f32x4::from_array([1.0; 4]);

/// Defines a layer of blending input data (local space transforms) and parameters (weights).
#[derive(Debug, Clone)]
pub struct BlendingLayer<I: OzzBuf<SoaTransform>> {
    /// Buffer to store local space transforms, that are usually outputted from a `SamplingJob`.
    pub transform: I,

    /// Blending weight of this layer. Negative values are considered as 0.
    /// Normalization is performed during the blending stage so weight can be in
    /// any range, even though range 0.0-1.0 is optimal.
    pub weight: f32,

    /// Optional buffer to store blending weight for each joint in this layer.
    /// If both `joint_weights` are empty in `BlendingJob` then per joint weight blending is disabled.
    /// When a layer doesn't specifies per joint weights, then it is implicitly considered as being 1.0.
    /// This default value is a reference value for the normalization process, which implies that the
    /// range of values for joint weights should be 0.0-1.0.
    /// Negative weight values are considered as 0, but positive ones aren't clamped because they could
    /// exceed 1.0 if all layers contains valid joint weights.
    pub joint_weights: Vec<Vec4>,
}

impl<I: OzzBuf<SoaTransform>> BlendingLayer<I> {
    pub fn new(transform: I) -> BlendingLayer<I> {
        return BlendingLayer {
            transform,
            weight: 0.0,
            joint_weights: Vec::new(),
        };
    }

    pub fn with_weight(transform: I, weight: f32) -> BlendingLayer<I> {
        return BlendingLayer {
            transform,
            weight,
            joint_weights: Vec::new(),
        };
    }

    pub fn with_joint_weights(transform: I, joint_weights: Vec<Vec4>) -> BlendingLayer<I> {
        return BlendingLayer {
            transform,
            weight: 0.0,
            joint_weights,
        };
    }

    fn joint_weight(&self, idx: usize) -> f32x4 {
        return fx4_from_vec4(self.joint_weights[idx]);
    }
}

/// Blending context for storing intermediate blending data in `BlendingJob`.
#[derive(Debug, Clone)]
pub struct BlendingContext {
    num_passes: u32,
    num_partial_passes: u32,
    accumulated_weight: f32,
    accumulated_weights: Vec<f32x4>,
}

impl Default for BlendingContext {
    fn default() -> BlendingContext {
        return BlendingContext {
            num_passes: 0,
            num_partial_passes: 0,
            accumulated_weight: 0.0,
            accumulated_weights: Vec::new(),
        };
    }
}

impl BlendingContext {
    /// New blending context with a given soa joints.
    pub fn new(soa_joints: usize) -> BlendingContext {
        return BlendingContext {
            num_passes: 0,
            num_partial_passes: 0,
            accumulated_weight: 0.0,
            accumulated_weights: vec![f32x4::splat(0.0); soa_joints],
        };
    }
}

///
/// `BlendingJob` is in charge of blending (mixing) multiple poses
/// (the result of a sampled animation) according to their respective weight,
/// into one output pose.
///
/// The number of transforms/joints blended by the job is defined by the number
/// of transforms of the rest pose (note that this is a SoA format). This means
/// that all buffers must be at least as big as the rest pose buffer.
///
/// Partial animation blending is supported through optional joint weights that
/// can be specified with layers joint_weights buffer. Unspecified joint weights
/// are considered as a unit weight of 1.0, allowing to mix full and partial
/// blend operations in a single pass.
///
#[derive(Debug)]
pub struct BlendingJob<S = Rc<Skeleton>, I = Rc<RefCell<Vec<SoaTransform>>>, O = Rc<RefCell<Vec<SoaTransform>>>>
where
    S: OzzObj<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzMutBuf<SoaTransform>,
{
    skeleton: Option<S>,
    context: Option<BlendingContext>,
    threshold: f32,
    layers: Vec<BlendingLayer<I>>,
    additive_layers: Vec<BlendingLayer<I>>,
    output: Option<O>,
}

pub type BlendingJobRef<'t> = BlendingJob<&'t Skeleton, &'t [SoaTransform], &'t mut SoaTransform>;
pub type BlendingJobRc<'t> = BlendingJob<Rc<Skeleton>, Rc<RefCell<Vec<SoaTransform>>>, Rc<RefCell<Vec<SoaTransform>>>>;
pub type BlendingJobArc = BlendingJob<Arc<Skeleton>, Arc<RwLock<Vec<SoaTransform>>>, Arc<RwLock<Vec<SoaTransform>>>>;

impl<S, I, O> Default for BlendingJob<S, I, O>
where
    S: OzzObj<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzMutBuf<SoaTransform>,
{
    fn default() -> BlendingJob<S, I, O> {
        return BlendingJob {
            skeleton: None,
            context: Some(BlendingContext::default()),
            threshold: 0.1,
            layers: Vec::new(),
            additive_layers: Vec::new(),
            output: None,
        };
    }
}

impl<S, I, O> BlendingJob<S, I, O>
where
    S: OzzObj<Skeleton>,
    I: OzzBuf<SoaTransform>,
    O: OzzMutBuf<SoaTransform>,
{
    /// Gets skeleton of `BlendingJob`.
    #[inline]
    pub fn skeleton(&self) -> Option<&S> {
        return self.skeleton.as_ref();
    }

    /// Set skeleton of `BlendingJob`.
    ///
    /// The skeleton that will be used during job.
    ///
    /// The rest pose size of this skeleton defines the number of transforms to blend.
    ///
    /// It is used when the accumulated weight for a bone on all layers is
    /// less than the threshold value, in order to fall back on valid transforms.
    #[inline]
    pub fn set_skeleton(&mut self, skeleton: S) {
        self.skeleton = Some(skeleton);
    }

    /// Clears skeleton of `BlendingJob`.
    #[inline]
    pub fn clear_skeleton(&mut self) {
        self.skeleton = None;
    }

    /// Gets context of `BlendingJob`. See [BlendingContext].
    #[inline]
    pub fn context(&self) -> Option<&BlendingContext> {
        return self.context.as_ref();
    }

    /// Sets context of `BlendingJob`. See [BlendingContext].
    #[inline]
    pub fn set_context(&mut self, ctx: BlendingContext) {
        self.context = Some(ctx);
    }

    /// Clears context of `BlendingJob`. See [BlendingContext].
    #[inline]
    pub fn clear_context(&mut self) {
        self.context = None;
    }

    /// Takes context of `BlendingJob`. See [BlendingContext].
    #[inline]
    pub fn take_context(&mut self) -> Option<BlendingContext> {
        return self.context.take();
    }

    /// Gets threshold of `BlendingJob`.
    #[inline]
    pub fn threshold(&self) -> f32 {
        return self.threshold;
    }

    /// Set threshold of `BlendingJob`.
    ///
    /// The job blends the rest pose to the output when the accumulated weight of
    /// all layers is less than this threshold value.
    /// Must be greater than 0.0.
    #[inline]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Gets layers of `BlendingJob`.
    #[inline]
    pub fn layers(&self) -> &[BlendingLayer<I>] {
        return &self.layers;
    }

    /// Gets mutable layers of `BlendingJob`.
    ///
    /// Job input layers, can be empty or nullptr. The range of layers that must be blended.
    #[inline]
    pub fn layers_mut(&mut self) -> &mut Vec<BlendingLayer<I>> {
        return &mut self.layers;
    }

    /// Gets additive layers of `BlendingJob`.
    #[inline]
    pub fn additive_layers(&self) -> &[BlendingLayer<I>] {
        return &self.additive_layers;
    }

    /// Gets mutable additive layers of `BlendingJob`.
    ///
    /// Job input additive layers, can be empty or nullptr. The range of layers that must be added to the output.
    #[inline]
    pub fn additive_layers_mut(&mut self) -> &mut Vec<BlendingLayer<I>> {
        return &mut self.additive_layers;
    }

    /// Gets output of `BlendingJob`.
    #[inline]
    pub fn output(&self) -> Option<&O> {
        return self.output.as_ref();
    }

    /// Sets output of `BlendingJob`.
    ///
    /// The range of output transforms to be filled with blended layer transforms during job execution.
    #[inline]
    pub fn set_output(&mut self, output: O) {
        self.output = Some(output);
    }

    /// Clears output of `BlendingJob`.
    #[inline]
    pub fn clear_output(&mut self) {
        self.output = None;
    }

    /// Validates `BlendingJob` parameters.
    pub fn validate(&self) -> bool {
        return (|| {
            let skeleton = self.skeleton.as_ref()?.obj();
            let _ctx = self.context.as_ref()?;
            let output = self.output.as_ref()?.buf().ok()?;

            let mut ok = self.threshold > 0.0;
            ok &= output.len() >= skeleton.num_soa_joints();

            for layer in &self.layers {
                ok &= layer.transform.buf().ok()?.len() >= skeleton.num_soa_joints();
                if !layer.joint_weights.is_empty() {
                    ok &= layer.joint_weights.len() >= skeleton.num_soa_joints();
                }
            }

            for layer in &self.additive_layers {
                ok &= layer.transform.buf().ok()?.len() >= skeleton.num_soa_joints();
                if !layer.joint_weights.is_empty() {
                    ok &= layer.joint_weights.len() >= skeleton.num_soa_joints();
                }
            }

            return Some(ok);
        })()
        .unwrap_or(false);
    }

    /// Runs job's blending task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        let skeleton = self.skeleton.as_ref().ok_or(OzzError::InvalidJob)?.obj();
        let ctx = self.context.as_mut().ok_or(OzzError::InvalidJob)?;
        let mut output = self.output.as_mut().ok_or(OzzError::InvalidJob)?.mut_buf()?;

        let mut ok = self.threshold > 0.0;
        ok &= output.len() >= skeleton.num_soa_joints();
        if !ok {
            return Err(OzzError::InvalidJob);
        }

        ctx.num_partial_passes = 0;
        ctx.num_passes = 0;
        ctx.accumulated_weight = 0.0;
        if ctx.accumulated_weights.len() < skeleton.num_soa_joints() {
            ctx.accumulated_weights.resize(skeleton.num_soa_joints(), ZERO);
        }

        Self::blend_layers(skeleton, ctx, &self.layers, &mut output)?;
        Self::blend_rest_pose(skeleton, ctx, self.threshold, &mut output);
        Self::normalize(skeleton, ctx, &mut output);
        Self::add_layers(skeleton, &self.additive_layers, &mut output)?;
        return Ok(());
    }

    fn blend_layers(
        skeleton: &Skeleton,
        ctx: &mut BlendingContext,
        layers: &[BlendingLayer<I>],
        output: &mut [SoaTransform],
    ) -> Result<(), OzzError> {
        let num_soa_joints = skeleton.num_soa_joints();

        for layer in layers {
            let transform = layer.transform.buf()?;
            if transform.len() < skeleton.num_soa_joints() {
                return Err(OzzError::InvalidJob);
            }
            if !layer.joint_weights.is_empty() && layer.joint_weights.len() < skeleton.num_soa_joints() {
                return Err(OzzError::InvalidJob);
            }

            if layer.weight <= 0.0 {
                continue;
            }
            ctx.accumulated_weight += layer.weight;
            let layer_weight = f32x4::splat(layer.weight);

            if !layer.joint_weights.is_empty() {
                ctx.num_partial_passes += 1;

                if ctx.num_passes == 0 {
                    for idx in 0..num_soa_joints {
                        let weight = layer_weight * layer.joint_weight(idx).simd_max(ZERO);
                        ctx.accumulated_weights[idx] = weight;
                        Self::blend_1st_pass(&transform[idx], weight, &mut output[idx]);
                    }
                } else {
                    for idx in 0..num_soa_joints {
                        let weight = layer_weight * layer.joint_weight(idx).simd_max(ZERO);
                        ctx.accumulated_weights[idx] += weight;
                        Self::blend_n_pass(&transform[idx], weight, &mut output[idx]);
                    }
                }
                ctx.num_passes += 1;
            } else {
                if ctx.num_passes == 0 {
                    for idx in 0..num_soa_joints {
                        ctx.accumulated_weights[idx] = layer_weight;
                        Self::blend_1st_pass(&transform[idx], layer_weight, &mut output[idx]);
                    }
                } else {
                    for idx in 0..num_soa_joints {
                        ctx.accumulated_weights[idx] += layer_weight;
                        Self::blend_n_pass(&transform[idx], layer_weight, &mut output[idx]);
                    }
                }
                ctx.num_passes += 1;
            }
        }

        return Ok(());
    }

    fn blend_rest_pose(skeleton: &Skeleton, ctx: &mut BlendingContext, threshold: f32, output: &mut [SoaTransform]) {
        let joint_rest_poses = skeleton.joint_rest_poses();

        if ctx.num_partial_passes == 0 {
            let bp_weight = threshold - ctx.accumulated_weight;
            if bp_weight > 0.0 {
                if ctx.num_passes == 0 {
                    ctx.accumulated_weight = 1.0;
                    for idx in 0..joint_rest_poses.len() {
                        output[idx] = joint_rest_poses[idx];
                    }
                } else {
                    ctx.accumulated_weight = threshold;
                    let simd_bp_weight = f32x4::splat(bp_weight);
                    for idx in 0..joint_rest_poses.len() {
                        Self::blend_n_pass(&joint_rest_poses[idx], simd_bp_weight, &mut output[idx]);
                    }
                }
            }
        } else {
            let simd_threshold = f32x4::splat(threshold);
            for idx in 0..joint_rest_poses.len() {
                let bp_weight = (simd_threshold - ctx.accumulated_weights[idx]).simd_max(ZERO);
                ctx.accumulated_weights[idx] = simd_threshold.simd_max(ctx.accumulated_weights[idx]);
                Self::blend_n_pass(&joint_rest_poses[idx], bp_weight, &mut output[idx]);
            }
        }
    }

    fn normalize(skeleton: &Skeleton, ctx: &mut BlendingContext, output: &mut [SoaTransform]) {
        let joint_rest_poses = skeleton.joint_rest_poses();

        if ctx.num_partial_passes == 0 {
            let ratio = f32x4::splat(ctx.accumulated_weight.recip());
            for idx in 0..joint_rest_poses.len() {
                let dest = &mut output[idx];
                dest.translation = dest.translation.mul_num(ratio);
                dest.rotation = dest.rotation.normalize();
                dest.scale = dest.scale.mul_num(ratio);
            }
        } else {
            for idx in 0..joint_rest_poses.len() {
                let dest = &mut output[idx];
                let ratio = ctx.accumulated_weights[idx].recip();
                dest.translation = dest.translation.mul_num(ratio);
                dest.rotation = dest.rotation.normalize();
                dest.scale = dest.scale.mul_num(ratio);
            }
        }
    }

    fn add_layers(
        skeleton: &Skeleton,
        layers: &[BlendingLayer<I>],
        output: &mut [SoaTransform],
    ) -> Result<(), OzzError> {
        let joint_rest_poses = skeleton.joint_rest_poses();

        for layer in layers {
            let transform = layer.transform.buf()?;
            if transform.len() < skeleton.num_soa_joints() {
                return Err(OzzError::InvalidJob);
            }
            if !layer.joint_weights.is_empty() && layer.joint_weights.len() < skeleton.num_soa_joints() {
                return Err(OzzError::InvalidJob);
            }

            if layer.weight > 0.0 {
                let layer_weight = f32x4::splat(layer.weight);

                if !layer.joint_weights.is_empty() {
                    for idx in 0..joint_rest_poses.len() {
                        let weight = layer_weight * layer.joint_weight(idx).simd_max(ZERO);
                        let one_minus_weight = ONE - weight;
                        Self::blend_add_pass(&transform[idx], weight, one_minus_weight, &mut output[idx]);
                    }
                } else {
                    let one_minus_weight = ONE - layer_weight;
                    for idx in 0..joint_rest_poses.len() {
                        Self::blend_add_pass(&transform[idx], layer_weight, one_minus_weight, &mut output[idx]);
                    }
                }
            } else if layer.weight < 0.0 {
                let layer_weight = f32x4::splat(-layer.weight);

                if !layer.joint_weights.is_empty() {
                    for idx in 0..joint_rest_poses.len() {
                        let weight = layer_weight * layer.joint_weight(idx).simd_max(ZERO);
                        let one_minus_weight = ONE - weight;
                        Self::blend_sub_pass(&transform[idx], weight, one_minus_weight, &mut output[idx]);
                    }
                } else {
                    let one_minus_weight = ONE - layer_weight;
                    for idx in 0..joint_rest_poses.len() {
                        Self::blend_sub_pass(&transform[idx], layer_weight, one_minus_weight, &mut output[idx]);
                    }
                }
            }
        }

        return Ok(());
    }

    #[inline(always)]
    fn blend_1st_pass(input: &SoaTransform, weight: f32x4, output: &mut SoaTransform) {
        output.translation = input.translation.mul_num(weight);
        output.rotation = input.rotation.mul_num(weight);
        output.scale = input.scale.mul_num(weight);
    }

    #[inline(always)]
    fn blend_n_pass(input: &SoaTransform, weight: f32x4, output: &mut SoaTransform) {
        output.translation = output.translation.add(&input.translation.mul_num(weight));
        let dot = output.rotation.dot(&input.rotation);
        let rotation = input.rotation.xor_num(fx4_sign(dot));
        output.rotation = output.rotation.add(&rotation.mul_num(weight));
        output.scale = output.scale.add(&input.scale.mul_num(weight));
    }

    #[inline(always)]
    fn blend_add_pass(input: &SoaTransform, weight: f32x4, soa_one_minus_weight: f32x4, output: &mut SoaTransform) {
        output.translation = output.translation.add(&input.translation.mul_num(weight));

        let rotation = input.rotation.positive_w();
        let interp_quat = SoaQuat {
            x: rotation.x * weight,
            y: rotation.y * weight,
            z: rotation.z * weight,
            w: (rotation.w - ONE) * weight + ONE,
        };
        output.rotation = output.rotation.mul(&interp_quat.normalize());

        output.scale = SoaVec3 {
            x: output.scale.x * (soa_one_minus_weight + input.scale.x * weight),
            y: output.scale.y * (soa_one_minus_weight + input.scale.y * weight),
            z: output.scale.z * (soa_one_minus_weight + input.scale.z * weight),
        }
    }

    #[inline(always)]
    fn blend_sub_pass(input: &SoaTransform, weight: f32x4, one_minus_weight: f32x4, output: &mut SoaTransform) {
        output.translation = output.translation.sub(&input.translation.mul_num(weight));

        let rotation = input.rotation.positive_w();
        let interp_quat = SoaQuat {
            x: rotation.x * weight,
            y: rotation.y * weight,
            z: rotation.z * weight,
            w: (rotation.w - ONE) * weight + ONE,
        };
        output.rotation = output.rotation.mul(&interp_quat.normalize().conjugate());

        output.scale = SoaVec3 {
            x: output.scale.x * (input.scale.x * weight + one_minus_weight).recip(),
            y: output.scale.y * (input.scale.y * weight + one_minus_weight).recip(),
            z: output.scale.z * (input.scale.z * weight + one_minus_weight).recip(),
        };
    }
}

#[cfg(test)]
mod blending_tests {
    use std::mem;
    use wasm_bindgen_test::*;

    use super::*;
    use crate::base::DeterministicState;
    use crate::skeleton::JointHashMap;

    const IDENTITY: SoaTransform = SoaTransform {
        translation: SoaVec3::splat_col([0.0; 3]),
        rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
        scale: SoaVec3::splat_col([1.0; 3]),
    };

    fn make_buf<T>(v: Vec<T>) -> Rc<RefCell<Vec<T>>> {
        return Rc::new(RefCell::new(v));
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_validity() {
        let skeleton = Rc::new(Skeleton::from_path("./resource/skeleton-blending.ozz").unwrap());
        let num_bind_pose = skeleton.num_soa_joints();
        let default_layer = BlendingLayer {
            transform: make_buf(vec![SoaTransform::default(); num_bind_pose]),
            weight: 0.5,
            joint_weights: Vec::new(),
        };

        // empty/default job
        let mut job: BlendingJob = BlendingJob::default();
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid output
        let mut job: BlendingJob = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // layers are optional
        let mut job: BlendingJob = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(job.validate());
        assert!(job.run().is_ok());

        // invalid layer input, too small
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.layers_mut().push(default_layer.clone());
        job.layers_mut().push(BlendingLayer {
            transform: make_buf(vec![]),
            weight: 0.5,
            joint_weights: Vec::new(),
        });
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid output range, smaller output
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.layers_mut().push(default_layer.clone());
        job.set_output(make_buf(vec![SoaTransform::default(); 3]));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid threshold
        let mut job: BlendingJob = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.set_threshold(0.0);
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid joint weights range
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.layers_mut().push(BlendingLayer {
            transform: make_buf(vec![SoaTransform::default(); num_bind_pose]),
            weight: 0.5,
            joint_weights: vec![Vec4::splat(0.5); 1],
        });
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // valid job
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.layers_mut().push(default_layer.clone());
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(job.validate());
        assert!(job.run().is_ok());

        // valid joint weights range
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.layers_mut().push(default_layer.clone());
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(job.validate());
        assert!(job.run().is_ok());

        // valid job, bigger output
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.layers_mut().push(default_layer.clone());
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose + 5]));
        assert!(job.validate());
        assert!(job.run().is_ok());

        // valid additive job, no normal blending
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.additive_layers_mut().push(default_layer.clone());
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(job.validate());
        assert!(job.run().is_ok());

        // valid additive job, with normal blending
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.layers_mut().push(default_layer.clone());
        job.additive_layers_mut().push(default_layer.clone());
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(job.validate());
        assert!(job.run().is_ok());

        // invalid layer input range, too small
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.additive_layers_mut().push(BlendingLayer {
            transform: make_buf(vec![SoaTransform::default(); 3]),
            weight: 0.5,
            joint_weights: Vec::new(),
        });
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // valid additive job, with per-joint weights
        let mut job = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        job.additive_layers_mut().push(BlendingLayer {
            transform: make_buf(vec![SoaTransform::default(); num_bind_pose]),
            weight: 0.5,
            joint_weights: vec![Vec4::splat(0.5); num_bind_pose],
        });
        job.set_output(make_buf(vec![SoaTransform::default(); num_bind_pose]));
        assert!(job.validate());
        assert!(job.run().is_ok());
    }

    fn new_layers(
        input1: Vec<SoaTransform>,
        weights1: Vec<Vec4>,
        input2: Vec<SoaTransform>,
        weights2: Vec<Vec4>,
    ) -> Vec<BlendingLayer<Rc<RefCell<Vec<SoaTransform>>>>> {
        return vec![
            BlendingLayer {
                transform: make_buf(input1),
                weight: 0.0,
                joint_weights: weights1,
            },
            BlendingLayer {
                transform: make_buf(input2),
                weight: 0.0,
                joint_weights: weights2,
            },
        ];
    }

    fn execute_test(
        skeleton: &Rc<Skeleton>,
        layers: Vec<BlendingLayer<Rc<RefCell<Vec<SoaTransform>>>>>,
        additive_layers: Vec<BlendingLayer<Rc<RefCell<Vec<SoaTransform>>>>>,
        expected_translations: Vec<SoaVec3>,
        expected_rotations: Vec<SoaQuat>,
        expected_scales: Vec<SoaVec3>,
        message: &str,
    ) {
        let mut job: BlendingJob = BlendingJob::default();
        job.set_skeleton(skeleton.clone());
        *job.layers_mut() = layers;
        *job.additive_layers_mut() = additive_layers;
        let joint_rest_poses = skeleton.num_soa_joints();
        let output = make_buf(vec![SoaTransform::default(); joint_rest_poses]);
        job.set_output(output.clone());
        job.run().unwrap();

        for idx in 0..joint_rest_poses {
            let out = output.as_ref().borrow()[idx];

            if !expected_translations.is_empty() {
                let a: [Vec4; 3] = unsafe { mem::transmute(out.translation) };
                let b: [Vec4; 3] = unsafe { mem::transmute(expected_translations[idx]) };
                assert!(
                    a[0].abs_diff_eq(b[0], 2e-6f32)
                        && a[1].abs_diff_eq(b[1], 2e-6f32)
                        && a[2].abs_diff_eq(b[2], 2e-6f32),
                    "{} => translation idx:{} actual:{:?}, excepted:{:?}",
                    message,
                    idx,
                    out.translation,
                    expected_translations[idx],
                );
            }
            if !expected_rotations.is_empty() {
                let a: [Vec4; 4] = unsafe { mem::transmute(out.rotation) };
                let b: [Vec4; 4] = unsafe { mem::transmute(expected_rotations[idx]) };
                assert!(
                    a[0].abs_diff_eq(b[0], 0.0001)
                        && a[1].abs_diff_eq(b[1], 0.0001)
                        && a[2].abs_diff_eq(b[2], 0.0001)
                        && a[3].abs_diff_eq(b[3], 0.0001),
                    "{} => rotation idx:{} actual:{:?}, excepted:{:?}",
                    message,
                    idx,
                    out.rotation,
                    expected_rotations[idx],
                );
            }
            if !expected_scales.is_empty() {
                let a: [Vec4; 3] = unsafe { mem::transmute(out.scale) };
                let b: [Vec4; 3] = unsafe { mem::transmute(expected_scales[idx]) };
                assert!(
                    a[0].abs_diff_eq(b[0], 2e-6f32)
                        && a[1].abs_diff_eq(b[1], 2e-6f32)
                        && a[2].abs_diff_eq(b[2], 2e-6f32),
                    "{} => scale idx:{} actual:{:?}, excepted:{:?}",
                    message,
                    idx,
                    out.scale,
                    expected_scales[idx],
                );
            }
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_empty() {
        let mut joint_rest_poses = vec![SoaTransform::default(); 2];
        joint_rest_poses[0].translation =
            SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]);
        joint_rest_poses[0].rotation = SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]);
        joint_rest_poses[0].scale = SoaVec3::new(
            [0.0, 10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0, 70.0],
            [80.0, 90.0, 100.0, 110.0],
        );
        joint_rest_poses[1].translation = joint_rest_poses[0].translation.mul_num(f32x4::splat(2.0));
        joint_rest_poses[1].rotation = SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]);
        joint_rest_poses[1].scale = joint_rest_poses[0].scale.mul_num(f32x4::splat(2.0));

        let skeleton = Rc::new(Skeleton::from_raw(
            joint_rest_poses,
            vec![0; 8],
            JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new()),
        ));

        execute_test(
            &skeleton,
            Vec::new(),
            Vec::new(),
            vec![
                SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]),
                SoaVec3::new([0.0, 2.0, 4.0, 6.0], [8.0, 10.0, 12.0, 14.0], [16.0, 18.0, 20.0, 22.0]),
            ],
            vec![],
            vec![
                SoaVec3::new(
                    [0.0, 10.0, 20.0, 30.0],
                    [40.0, 50.0, 60.0, 70.0],
                    [80.0, 90.0, 100.0, 110.0],
                ),
                SoaVec3::new(
                    [00.0, 20.0, 40.0, 60.0],
                    [80.0, 100.0, 120.0, 140.0],
                    [160.0, 180.0, 200.0, 220.0],
                ),
            ],
            "empty",
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_weight() {
        let mut input1 = vec![IDENTITY; 2];
        input1[0].translation = SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]);
        input1[1].translation = SoaVec3::new(
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0],
        );
        let mut input2 = vec![IDENTITY; 2];
        input2[0].translation = input1[0].translation.neg();
        input2[1].translation = input1[1].translation.neg();
        let mut layers = new_layers(input1, vec![], input2, vec![]);

        let rest_poses = vec![
            SoaTransform {
                translation: SoaVec3::splat_col([0.0, 0.0, 0.0]),
                rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                scale: SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]),
            },
            SoaTransform {
                translation: SoaVec3::splat_col([0.0, 0.0, 0.0]),
                rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                scale: SoaVec3::new(
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                ),
            },
        ];
        let skeleton = Rc::new(Skeleton::from_raw(
            rest_poses,
            vec![0; 8],
            JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new()),
        ));

        {
            layers[0].weight = -0.07;
            layers[1].weight = 1.0;
            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![
                    SoaVec3::new(
                        [-0.0, -1.0, -2.0, -3.0],
                        [-4.0, -5.0, -6.0, -7.0],
                        [-8.0, -9.0, -10.0, -11.0],
                    ),
                    SoaVec3::new(
                        [-12.0, -13.0, -14.0, -15.0],
                        [-16.0, -17.0, -18.0, -19.0],
                        [-20.0, -21.0, -22.0, -23.0],
                    ),
                ],
                vec![],
                vec![SoaVec3::splat_col([1.0, 1.0, 1.0]); 2],
                "weight - 1",
            );
        }

        {
            layers[0].weight = 1.0;
            layers[1].weight = 1.0e-23f32;
            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![
                    SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]),
                    SoaVec3::new(
                        [12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0],
                    ),
                ],
                vec![],
                vec![SoaVec3::splat_col([1.0, 1.0, 1.0]); 2],
                "weight - 2",
            );
        }

        {
            layers[0].weight = 0.5;
            layers[1].weight = 0.5;
            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![SoaVec3::splat_col([0.0, 0.0, 0.0]); 2],
                vec![],
                vec![SoaVec3::splat_col([1.0, 1.0, 1.0]); 2],
                "weight - 3",
            );
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_joint_weights() {
        let mut input1 = vec![IDENTITY; 2];
        input1[0].translation = SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]);
        input1[1].translation = SoaVec3::new(
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0],
        );
        let mut input2 = vec![IDENTITY; 2];
        input2[0].translation = input1[0].translation.neg();
        input2[1].translation = input1[1].translation.neg();
        let weights1 = vec![Vec4::new(1.0, 1.0, 0.0, 0.0), Vec4::new(1.0, 0.0, 1.0, 1.0)];
        let weights2 = vec![Vec4::new(1.0, 1.0, 1.0, 0.0), Vec4::new(0.0, 1.0, 1.0, 1.0)];
        let mut layers = new_layers(input1, weights1, input2, weights2);

        let rest_poses = vec![
            SoaTransform {
                translation: SoaVec3::new(
                    [10.0, 11.0, 12.0, 13.0],
                    [14.0, 15.0, 16.0, 17.0],
                    [18.0, 19.0, 20.0, 21.0],
                ),
                rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                scale: SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]),
            },
            SoaTransform {
                translation: SoaVec3::splat_col([0.0, 0.0, 0.0]),
                rotation: SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]),
                scale: SoaVec3::new([0.0, 2.0, 4.0, 6.0], [8.0, 10.0, 12.0, 14.0], [16.0, 18.0, 20.0, 22.0]),
            },
        ];
        let skeleton = Rc::new(Skeleton::from_raw(
            rest_poses,
            vec![0; 8],
            JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new()),
        ));

        {
            layers[0].weight = 0.5;
            layers[1].weight = 0.5;
            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![
                    SoaVec3::new([0.0, 0.0, -2.0, 13.0], [0.0, 0.0, -6.0, 17.0], [0.0, 0.0, -10.0, 21.0]),
                    SoaVec3::new(
                        [12.0, -13.0, 0.0, 0.0],
                        [16.0, -17.0, 0.0, 0.0],
                        [20.0, -21.0, 0.0, 0.0],
                    ),
                ],
                vec![],
                vec![
                    SoaVec3::new([1.0, 1.0, 1.0, 3.0], [1.0, 1.0, 1.0, 7.0], [1.0, 1.0, 1.0, 11.0]),
                    SoaVec3::splat_col([1.0, 1.0, 1.0]),
                ],
                "joint weight - 1",
            );
        }

        {
            layers[0].weight = 0.0;
            layers[1].weight = 1.0;
            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![
                    SoaVec3::new(
                        [-0.0, -1.0, -2.0, 13.0],
                        [-4.0, -5.0, -6.0, 17.0],
                        [-8.0, -9.0, -10.0, 21.0],
                    ),
                    SoaVec3::new(
                        [0.0, -13.0, -14.0, -15.0],
                        [0.0, -17.0, -18.0, -19.0],
                        [0.0, -21.0, -22.0, -23.0],
                    ),
                ],
                vec![],
                vec![
                    SoaVec3::new([1.0, 1.0, 1.0, 3.0], [1.0, 1.0, 1.0, 7.0], [1.0, 1.0, 1.0, 11.0]),
                    SoaVec3::new([0.0, 1.0, 1.0, 1.0], [8.0, 1.0, 1.0, 1.0], [16.0, 1.0, 1.0, 1.0]),
                ],
                "joint weight - 2",
            );
        }
    }

    fn new_skeleton1() -> Rc<Skeleton> {
        let mut joint_rest_poses = vec![IDENTITY];
        joint_rest_poses[0].scale = SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]);

        return Rc::new(Skeleton::from_raw(
            joint_rest_poses,
            vec![0; 4],
            JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new()),
        ));
    }
    #[test]
    #[wasm_bindgen_test]
    fn test_normalize() {
        let skeleton = new_skeleton1();

        let mut input1 = vec![IDENTITY; 1];
        input1[0].rotation = SoaQuat::new(
            [0.70710677, 0.0, 0.0, 0.382683432],
            [0.0, 0.0, 0.70710677, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.70710677, 1.0, 0.70710677, 0.9238795],
        );
        let mut input2 = vec![IDENTITY; 1];
        input2[0].rotation = SoaQuat::new(
            [0.0, 0.70710677, -0.70710677, -0.382683432],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -0.70710677, 0.0],
            [1.0, 0.70710677, 0.0, -0.9238795],
        );
        let mut layers = vec![
            BlendingLayer {
                transform: make_buf(input1),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
            BlendingLayer {
                transform: make_buf(input2),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
        ];

        {
            layers[0].weight = 0.2;
            layers[0].transform.mut_buf().unwrap()[0].translation =
                SoaVec3::new([2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]);
            layers[1].weight = 0.3;
            layers[1].transform.mut_buf().unwrap()[0].translation =
                SoaVec3::new([3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]);

            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![SoaVec3::new(
                    [2.6, 3.6, 4.6, 5.6],
                    [6.6, 7.6, 8.6, 9.6],
                    [10.6, 11.6, 12.6, 13.6],
                )],
                vec![SoaQuat::new(
                    [0.30507791, 0.45761687, -0.58843851, 0.38268352],
                    [0.0, 0.0, 0.39229235, 0.0],
                    [0.0, 0.0, -0.58843851, 0.0],
                    [0.95224595, 0.88906217, 0.39229235, 0.92387962],
                )],
                vec![SoaVec3::splat_col([1.0, 1.0, 1.0])],
                "normalize - 1",
            );
        }

        layers[0].transform.mut_buf().unwrap()[0].translation = SoaVec3::new(
            [5.0, 10.0, 15.0, 20.0],
            [25.0, 30.0, 35.0, 40.0],
            [45.0, 50.0, 55.0, 60.0],
        );
        layers[1].transform.mut_buf().unwrap()[0].translation = SoaVec3::new(
            [10.0, 15.0, 20.0, 25.0],
            [30.0, 35.0, 40.0, 45.0],
            [50.0, 55.0, 60.0, 65.0],
        );

        {
            layers[0].weight = 2.0;
            layers[1].weight = 3.0;

            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![SoaVec3::new(
                    [8.0, 13.0, 18.0, 23.0],
                    [28.0, 33.0, 38.0, 43.0],
                    [48.0, 53.0, 58.0, 63.0],
                )],
                vec![SoaQuat::new(
                    [0.30507791, 0.45761687, -0.58843851, 0.38268352],
                    [0.0, 0.0, 0.39229235, 0.0],
                    [0.0, 0.0, -0.58843851, 0.0],
                    [0.95224595, 0.88906217, 0.39229235, 0.92387962],
                )],
                vec![SoaVec3::splat_col([1.0, 1.0, 1.0])],
                "normalize - 1",
            );
        }

        {
            layers[1].joint_weights = vec![Vec4::new(1.0, -1.0, 2.0, 0.1)];

            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![SoaVec3::new(
                    [8.0, 10.0, 150.0 / 8.0, 47.5 / 2.30],
                    [28.0, 30.0, 310.0 / 8.0, 93.5 / 2.30],
                    [48.0, 50.0, 470.0 / 8.0, 139.5 / 2.30],
                )],
                vec![],
                vec![SoaVec3::splat_col([1.0, 1.0, 1.0])],
                "normalize - 3",
            );
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_threshold() {
        let skeleton = new_skeleton1();

        let mut input1 = vec![IDENTITY; 1];
        input1[0].translation = SoaVec3::new([2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]);
        let mut input2 = vec![IDENTITY; 1];
        input2[0].translation = SoaVec3::new([3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0]);
        let mut layers = vec![
            BlendingLayer {
                transform: make_buf(input1),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
            BlendingLayer {
                transform: make_buf(input2),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
        ];

        {
            layers[0].weight = 0.04;
            layers[1].weight = 0.06;
            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![SoaVec3::new(
                    [2.6, 3.6, 4.6, 5.6],
                    [6.6, 7.6, 8.6, 9.6],
                    [10.6, 11.6, 12.6, 13.6],
                )],
                vec![SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0])],
                vec![SoaVec3::splat_col([1.0, 1.0, 1.0])],
                "threshold - 1",
            );
        }

        {
            layers[0].weight = 1.0e-27;
            layers[1].weight = 0.0;
            execute_test(
                &skeleton,
                layers.clone(),
                Vec::new(),
                vec![SoaVec3::splat_col([0.0; 3])],
                vec![SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0])],
                vec![SoaVec3::new(
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                )],
                "threshold - 2",
            );
        }
    }

    // fn new_layers3() -> Vec<BlendingLayer<f32>> {
    //     let mut input1 = vec![SoaTransform::<f32>::default(); 4];
    //     input1[0].translation = Vec3::new(0.0, 4.0, 8.0);
    //     input1[1].translation = Vec3::new(1.0, 5.0, 9.0);
    //     input1[2].translation = Vec3::new(2.0, 6.0, 10.0);
    //     input1[3].translation = Vec3::new(3.0, 7.0, 11.0);
    //     input1[0].rotation = Quat::new(0.70710677, 0.70710677, 0.0, 0.0);
    //     input1[1].rotation = Quat::identity();
    //     input1[2].rotation = Quat::new(-0.70710677, 0.0, 0.70710677, 0.0);
    //     input1[3].rotation = Quat::new(0.9238795, 0.382683432, 0.0, 0.0);
    //     input1[0].scale = Vec3::new(12.0, 16.0, 20.0);
    //     input1[1].scale = Vec3::new(13.0, 17.0, 21.0);
    //     input1[2].scale = Vec3::new(14.0, 18.0, 22.0);
    //     input1[3].scale = Vec3::new(15.0, 19.0, 23.0);

    //     let input2 = input1
    //         .iter()
    //         .map(|x| SoaTransform {
    //             translation: x.translation.neg(),
    //             rotation: x.rotation.conjugate(),
    //             scale: x.scale.neg(),
    //         })
    //         .collect::<Vec<_>>();

    //     return vec![
    //         BlendingLayer {
    //             input: make_buf(input1),
    //             weight: 0.0,
    //             joint_weights: Vec::new(),
    //         },
    //         BlendingLayer {
    //             input: make_buf(input2),
    //             weight: 0.0,
    //             joint_weights: Vec::new(),
    //         },
    //     ];
    // }

    #[test]
    #[wasm_bindgen_test]
    fn test_additive_weight() {
        let skeleton = Rc::new(Skeleton::from_raw(
            vec![IDENTITY; 1],
            vec![0; 4],
            JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new()),
        ));

        let mut input1 = vec![IDENTITY; 1];
        input1[0].translation = SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]);
        input1[0].rotation = SoaQuat::new(
            [0.70710677, 0.0, 0.0, 0.382683432],
            [0.0, 0.0, 0.70710677, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.70710677, 1.0, -0.70710677, 0.9238795],
        );
        input1[0].scale = SoaVec3::new(
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0],
        );
        let mut input2 = vec![IDENTITY; 1];
        input2[0].translation = input1[0].translation.neg();
        input2[0].rotation = input1[0].rotation.conjugate();
        input2[0].scale = input1[0].scale.neg();

        let mut layers = vec![BlendingLayer {
            transform: make_buf(input1.clone()),
            weight: 0.0,
            joint_weights: Vec::new(),
        }];

        {
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::splat_col([0.0; 3]); 4],
                vec![SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]); 4],
                vec![SoaVec3::splat_col([1.0; 3]); 4],
                "additive weight - 1",
            );
        }

        {
            layers[0].weight = 0.5;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::new(
                    [0.0, 0.5, 1.0, 1.5],
                    [2.0, 2.5, 3.0, 3.5],
                    [4.0, 4.5, 5.0, 5.5],
                )],
                vec![SoaQuat::new(
                    [0.3826834, 0.0, 0.0, 0.19509032],
                    [0.0, 0.0, -0.3826834, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.9238795, 1.0, 0.9238795, 0.98078528],
                )],
                vec![SoaVec3::new(
                    [6.5, 7.0, 7.5, 8.0],
                    [8.5, 9.0, 9.5, 10.0],
                    [10.5, 11.0, 11.5, 12.0],
                )],
                "additive weight - 2",
            );
        }

        {
            layers[0].weight = 1.0;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::new(
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                )],
                vec![SoaQuat::new(
                    [0.70710677, 0.0, 0.0, 0.382683432],
                    [0.0, 0.0, -0.70710677, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.70710677, 1.0, 0.70710677, 0.9238795],
                )],
                vec![SoaVec3::new(
                    [12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0],
                )],
                "additive weight - 3",
            );
        }

        let mut layers = vec![
            BlendingLayer {
                transform: make_buf(input1.clone()),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
            BlendingLayer {
                transform: make_buf(input2),
                weight: 0.0,
                joint_weights: Vec::new(),
            },
        ];

        {
            layers[0].weight = 0.0;
            layers[1].weight = 1.0;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::new(
                    [-0.0, -1.0, -2.0, -3.0],
                    [-4.0, -5.0, -6.0, -7.0],
                    [-8.0, -9.0, -10.0, -11.0],
                )],
                vec![SoaQuat::new(
                    [-0.70710677, -0.0, -0.0, -0.382683432],
                    [-0.0, -0.0, 0.70710677, -0.0],
                    [-0.0, -0.0, -0.0, -0.0],
                    [0.70710677, 1.0, 0.70710677, 0.9238795],
                )],
                vec![SoaVec3::new(
                    [-12.0, -13.0, -14.0, -15.0],
                    [-16.0, -17.0, -18.0, -19.0],
                    [-20.0, -21.0, -22.0, -23.0],
                )],
                "additive weight - 4",
            );
        }

        {
            layers[0].weight = 1.0;
            layers[1].weight = 1.0;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::splat_col([0.0; 3]); 4],
                vec![SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]); 4],
                vec![SoaVec3::new(
                    [-144.0, -169.0, -196.0, -225.0],
                    [-256.0, -289.0, -324.0, -361.0],
                    [-400.0, -441.0, -484.0, -529.0],
                )],
                "additive weight - 5",
            );
        }

        {
            layers[0].weight = 0.5;
            layers[1].transform = make_buf(input1);
            layers[1].weight = -0.5;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::splat_col([0.0; 3]); 4],
                vec![SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]); 4],
                vec![SoaVec3::splat_col([1.0; 3]); 4],
                "additive weight - 6",
            );
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_additive_joint_weight() {
        let skeleton = Rc::new(Skeleton::from_raw(
            vec![IDENTITY; 1],
            vec![0; 4],
            JointHashMap::with_hashers(DeterministicState::new(), DeterministicState::new()),
        ));

        let mut input1 = vec![IDENTITY; 1];
        input1[0].translation = SoaVec3::new([0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]);
        input1[0].rotation = SoaQuat::new(
            [0.70710677, 0.0, 0.0, 0.382683432],
            [0.0, 0.0, 0.70710677, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.70710677, 1.0, -0.70710677, 0.9238795],
        );
        input1[0].scale = SoaVec3::new(
            [12.0, 13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0, 19.0],
            [20.0, 21.0, 22.0, 23.0],
        );
        let mut layers = vec![BlendingLayer {
            transform: make_buf(input1.clone()),
            weight: 0.0,
            joint_weights: vec![Vec4::new(1.0, 0.5, 0.0, -1.0)],
        }];

        {
            layers[0].weight = 0.0;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::splat_col([0.0; 3]); 4],
                vec![SoaQuat::splat_col([0.0, 0.0, 0.0, 1.0]); 4],
                vec![SoaVec3::splat_col([1.0; 3]); 4],
                "additive joint weight - 1",
            );
        }

        {
            layers[0].weight = 0.5;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::new(
                    [0.0, 0.25, 0.0, 0.0],
                    [2.0, 1.25, 0.0, 0.0],
                    [4.0, 2.25, 0.0, 0.0],
                )],
                vec![SoaQuat::new(
                    [0.3826834, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.9238795, 1.0, 1.0, 1.0],
                )],
                vec![SoaVec3::new(
                    [6.5, 4.0, 1.0, 1.0],
                    [8.5, 5.0, 1.0, 1.0],
                    [10.5, 6.0, 1.0, 1.0],
                )],
                "additive joint weight - 2",
            );
        }

        {
            layers[0].weight = 1.0;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::new(
                    [0.0, 0.5, 0.0, 0.0],
                    [4.0, 2.5, 0.0, 0.0],
                    [8.0, 4.5, 0.0, 0.0],
                )],
                vec![SoaQuat::new(
                    [0.70710677, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.70710677, 1.0, 1.0, 1.0],
                )],
                vec![SoaVec3::new(
                    [12.0, 7.0, 1.0, 1.0],
                    [16.0, 9.0, 1.0, 1.0],
                    [20.0, 11.0, 1.0, 1.0],
                )],
                "additive joint weight - 3",
            );
        }

        {
            layers[0].weight = -1.0;
            execute_test(
                &skeleton,
                vec![],
                layers.clone(),
                vec![SoaVec3::new(
                    [0.0, -0.5, 0.0, 0.0],
                    [-4.0, -2.5, 0.0, 0.0],
                    [-8.0, -4.5, 0.0, 0.0],
                )],
                vec![SoaQuat::new(
                    [-0.70710677, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.70710677, 1.0, 1.0, 1.0],
                )],
                vec![SoaVec3::new(
                    [1.0 / 12.0, 1.0 / 7.0, 1.0, 1.0],
                    [1.0 / 16.0, 1.0 / 9.0, 1.0, 1.0],
                    [1.0 / 20.0, 1.0 / 11.0, 1.0, 1.0],
                )],
                "additive joint weight - 4",
            )
        };
    }
}
