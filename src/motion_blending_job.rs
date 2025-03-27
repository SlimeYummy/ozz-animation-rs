//!
//! Motion Blending Job.
//!

use glam::{Quat, Vec3A, Vec4};
use glam_ext::Transform3A;

use crate::base::OzzError;

/// Defines a layer of blending input data and its weight.
#[derive(Debug, Default, Clone)]
pub struct MotionBlendingLayer {
    /// Blending weight of this layer.
    ///
    /// Negative values are considered as 0.
    /// Normalization is performed at the end of the blending stage, so weight can be in any range,
    /// even though range [0:1] is optimal.
    pub weight: f32,

    /// The motion delta transform to be blended.
    pub delta: Transform3A,
}

///
/// MotionBlendingJob is in charge of blending delta motions according to their respective weight.
///
/// MotionBlendingJob is usually done to blend the motion resulting from the motion extraction
/// process, in parallel to blending animations.
///
#[derive(Debug, Default)]
pub struct MotionBlendingJob {
    layers: Vec<MotionBlendingLayer>,
    output: Transform3A,
}

pub type MotionBlendingJobRef = MotionBlendingJob;
pub type MotionBlendingJobRc = MotionBlendingJob;
pub type MotionBlendingJobArc = MotionBlendingJob;

impl MotionBlendingJob {
    /// Gets layers of `MotionBlendingJob`.
    #[inline]
    pub fn layers(&self) -> &[MotionBlendingLayer] {
        &self.layers
    }

    /// Gets mutable layers of `MotionBlendingJob`.
    ///
    /// Job input layers, can be empty. The range of layers that must be blended.
    #[inline]
    pub fn layers_mut(&mut self) -> &mut Vec<MotionBlendingLayer> {
        &mut self.layers
    }

    /// Gets output of `MotionBlendingJob`.
    #[inline]
    pub fn output(&self) -> Transform3A {
        self.output
    }

    /// Sets output of `MotionBlendingJob`.
    ///
    /// The range of output transforms to be filled with blended layer transforms during job execution.
    #[inline]
    pub fn set_output(&mut self, output: Transform3A) {
        self.output = output;
    }

    /// Clears output of `MotionBlendingJob`.
    #[inline]
    pub fn clear_output(&mut self) {
        self.output = Default::default();
    }

    /// Validates `MotionBlendingJob` parameters.
    pub fn validate(&self) -> bool {
        true
    }

    /// Runs job's blending task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        self.output = Transform3A::IDENTITY;

        let mut acc_weight = 0.0; // Accumulates weights for normalization
        let mut dl = 0.0f32; // Weighted translation lengths
        let mut dt = Vec3A::ZERO; // Weighted translations directions
        let mut dr = Vec4::ZERO; // Weighted rotations

        for layer in &self.layers {
            let weight = layer.weight;
            if weight <= 0.0 {
                continue;
            }
            acc_weight += weight;

            // Decomposes translation into a normalized vector and a length, to limit
            // lerp error while interpolating vector length.
            let len = layer.delta.translation.length();
            dl += len * weight;
            let denom = if len == 0.0 { 0.0 } else { weight / len };
            dt += layer.delta.translation * denom;

            // Accumulate weighted rotation (NLerp)
            let rot_vec = Vec4::from(layer.delta.rotation);
            let dot = dr.dot(rot_vec);
            let signed_weight = weight.copysign(if dot >= 0.0 { 1.0 } else { -1.0 });
            dr += rot_vec * signed_weight;
        }

        // Normalizes translation and re-applies interpolated length.
        let denom = dt.length() * acc_weight;
        let norm = if denom == 0.0 { 0.0 } else { dl / denom };
        self.output.translation = dt * norm;

        if dr.length_squared() != 0.0 {
            self.output.rotation = Quat::from_vec4(dr).normalize();
        } else {
            self.output.rotation = Quat::IDENTITY;
        }

        self.output.scale = Vec3A::ONE;
        Ok(())
    }
}

#[cfg(test)]
mod motion_blending_tests {
    use glam::Vec3;
    use wasm_bindgen_test::*;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_empty() {
        let mut job = MotionBlendingJob::default();
        job.run().unwrap();
        assert_eq!(job.output(), Transform3A::IDENTITY);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_run() {
        let mut job = MotionBlendingJob::default();

        // No layer
        job.run().unwrap();
        assert_eq!(job.output(), Transform3A::IDENTITY);

        // With layers
        job.layers_mut().push(MotionBlendingLayer {
            weight: 0.0,
            delta: Transform3A::new(
                Vec3::new(2.0, 0.0, 0.0),
                Quat::from_xyzw(0.70710677, 0.0, 0.0, 0.70710677),
                Vec3::ONE,
            ),
        });
        job.layers_mut().push(MotionBlendingLayer {
            weight: 0.0,
            delta: Transform3A::new(
                Vec3::new(0.0, 0.0, 3.0),
                Quat::from_xyzw(0.0, -0.70710677, 0.0, -0.70710677),
                Vec3::ONE,
            ),
        });

        // 0 weights
        job.layers_mut()[0].weight = 0.0;
        job.layers_mut()[1].weight = 0.0;
        job.run().unwrap();
        assert_eq!(job.output(), Transform3A::IDENTITY);

        // One non 0 weights
        job.layers_mut()[0].weight = 0.8;
        job.layers_mut()[1].weight = 0.0;
        job.run().unwrap();
        assert!(job.output().abs_diff_eq(
            Transform3A::new(
                Vec3::new(2.0, 0.0, 0.0),
                Quat::from_xyzw(0.70710677, 0.0, 0.0, 0.70710677),
                Vec3::ONE,
            ),
            1e-6
        ));

        // one negative weights
        job.layers_mut()[0].weight = 0.8;
        job.layers_mut()[1].weight = -1.0;
        job.run().unwrap();
        assert!(job.output().abs_diff_eq(
            Transform3A::new(
                Vec3::new(2.0, 0.0, 0.0),
                Quat::from_xyzw(0.70710677, 0.0, 0.0, 0.70710677),
                Vec3::ONE,
            ),
            1e-6
        ));

        // Two non 0 weights
        job.layers_mut()[0].weight = 0.8;
        job.layers_mut()[1].weight = 0.2;
        job.run().unwrap();
        assert!(job.output().abs_diff_eq(
            Transform3A::new(
                Vec3::new(2.134313, 0.0, 0.533578),
                Quat::from_xyzw(0.6172133, 0.1543033, 0.0, 0.7715167),
                Vec3::ONE,
            ),
            1e-6
        ));

        // Non normalized weights
        job.layers_mut()[0].weight = 8.0;
        job.layers_mut()[1].weight = 2.0;
        job.run().unwrap();
        assert!(job.output().abs_diff_eq(
            Transform3A::new(
                Vec3::new(2.134313, 0.0, 0.533578),
                Quat::from_xyzw(0.6172133, 0.1543033, 0.0, 0.7715167),
                Vec3::ONE,
            ),
            1e-6
        ));

        // 0 length translation
        job.layers_mut()[0].delta.translation = Vec3A::ZERO;
        job.layers_mut()[1].delta.translation = Vec3A::new(0.0, 0.0, 2.0);
        job.layers_mut()[0].weight = 0.8;
        job.layers_mut()[1].weight = 0.2;
        job.run().unwrap();
        assert!(job.output().abs_diff_eq(
            Transform3A::new(
                Vec3::new(0.0, 0.0, 0.4),
                Quat::from_xyzw(0.6172133, 0.1543033, 0.0, 0.7715167),
                Vec3::ONE,
            ),
            1e-6
        ));

        // Opposed translations
        job.layers_mut()[0].delta.translation = Vec3A::new(0.0, 0.0, -2.0);
        job.layers_mut()[1].delta.translation = Vec3A::new(0.0, 0.0, 2.0);
        job.layers_mut()[0].weight = 1.0;
        job.layers_mut()[1].weight = 1.0;
        job.run().unwrap();
        assert!(job.output().abs_diff_eq(
            Transform3A::new(
                Vec3::new(0.0, 0.0, 0.0),
                Quat::from_xyzw(0.408248, 0.408248, 0.0, 0.816496),
                Vec3::ONE,
            ),
            1e-6
        ));
    }
}
