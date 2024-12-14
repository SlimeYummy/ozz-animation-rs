//!
//! Aim IK Job.
//!

use glam::{Mat4, Quat, Vec3A};
use wide::{f32x4, CmpEq, CmpGt, CmpNe};

use crate::base::OzzError;
use crate::math::*;

///
/// Rotates a joint so it aims at a target.
///
/// Joint aim direction and up vectors can be different from joint axis. The job computes the
/// transformation (rotation) that needs to be applied to the joints such that a provided forward
/// vector (in joint local-space) aims at the target position (in skeleton model-space). Up vector
/// (in joint local-space) is also used to keep the joint oriented in the same direction as the
/// pole vector. The job also exposes an offset (in joint local-space) from where the forward
/// vector should aim the target.
///
/// Result is unstable if joint-to-target direction is parallel to pole vector, or if target is too
/// close to joint position.
///
#[derive(Debug)]
pub struct IKAimJob {
    target: f32x4,
    forward: f32x4,
    offset: f32x4,
    up: f32x4,
    pole_vector: f32x4,
    twist_angle: f32,
    weight: f32,
    joint: AosMat4,

    joint_correction: f32x4,
    reached: bool,
}

impl Default for IKAimJob {
    fn default() -> Self {
        Self {
            target: ZERO,
            forward: X_AXIS,
            offset: ZERO,
            up: Y_AXIS,
            pole_vector: Y_AXIS,
            twist_angle: 0.0,
            weight: 1.0,
            joint: AosMat4::identity(),

            joint_correction: QUAT_UNIT,
            reached: false,
        }
    }
}

impl IKAimJob {
    /// Gets target of `IKAimJob`.
    #[inline]
    pub fn target(&self) -> Vec3A {
        fx4_to_vec3a(self.target)
    }

    /// Sets target of `IKAimJob`.
    ///
    /// Target position to aim at, in model-space
    #[inline]
    pub fn set_target(&mut self, target: Vec3A) {
        self.target = fx4_from_vec3a(target);
    }

    /// Gets forward of `IKAimJob`.
    #[inline]
    pub fn forward(&self) -> Vec3A {
        fx4_to_vec3a(self.forward)
    }

    /// Sets forward of `IKAimJob`.
    ///
    /// Joint forward axis, in joint local-space, to be aimed at target position. Default is x axis.
    ///
    /// This vector shall be normalized, otherwise validation will fail.
    #[inline]
    pub fn set_forward(&mut self, forward: Vec3A) {
        self.forward = fx4_from_vec3a(forward);
    }

    /// Gets offset of `IKAimJob`.
    #[inline]
    pub fn offset(&self) -> Vec3A {
        fx4_to_vec3a(self.offset)
    }

    /// Sets offset of `IKAimJob`.
    ///
    /// Offset position from the joint in local-space, that will aim at target.
    #[inline]
    pub fn set_offset(&mut self, offset: Vec3A) {
        self.offset = fx4_from_vec3a(offset);
    }

    /// Gets up of `IKAimJob`.
    #[inline]
    pub fn up(&self) -> Vec3A {
        fx4_to_vec3a(self.up)
    }

    /// Sets up of `IKAimJob`.
    ///
    /// Joint up axis, in joint local-space, used to keep the joint oriented in the same direction as
    /// the pole vector. Default is y axis.
    #[inline]
    pub fn set_up(&mut self, up: Vec3A) {
        self.up = fx4_from_vec3a(up);
    }

    /// Gets pole vector of `IKAimJob`.
    #[inline]
    pub fn pole_vector(&self) -> Vec3A {
        fx4_to_vec3a(self.pole_vector)
    }

    /// Sets pole vector of `IKAimJob`.
    ///
    /// Pole vector, in model-space.
    /// The pole vector defines the direction the up should point to.
    ///
    /// Note that IK chain orientation will flip when target vector and the pole vector are aligned/crossing
    /// each other. It's caller responsibility to ensure that this doesn't happen.
    #[inline]
    pub fn set_pole_vector(&mut self, pole_vector: Vec3A) {
        self.pole_vector = fx4_from_vec3a(pole_vector);
    }

    /// Gets twist angle of `IKAimJob`.
    #[inline]
    pub fn twist_angle(&self) -> f32 {
        self.twist_angle
    }

    /// Sets twist angle of `IKAimJob`.
    ///
    /// Twist_angle rotates joint around the target vector. Default is 0.
    #[inline]
    pub fn set_twist_angle(&mut self, twist_angle: f32) {
        self.twist_angle = twist_angle;
    }

    /// Gets weight of `IKAimJob`.
    #[inline]
    pub fn weight(&self) -> f32 {
        self.weight
    }

    /// Sets weight of `IKAimJob`.
    ///
    /// Weight given to the IK correction clamped in range 0.0-1.0.
    /// This allows to blend / interpolate from no IK applied (0 weight) to full IK (1).
    #[inline]
    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }

    /// Gets joint of `IKAimJob`.
    #[inline]
    pub fn joint(&self) -> Mat4 {
        self.joint.into()
    }

    /// Sets joint of `IKAimJob`.
    ///
    /// Joint model-space matrix.
    #[inline]
    pub fn set_joint(&mut self, joint: Mat4) {
        self.joint = joint.into();
    }

    /// Gets **output** joint correction of `IKAimJob`.
    ///
    /// Output local-space joint correction quaternion.
    /// It needs to be multiplied with joint local-space quaternion.
    #[inline]
    pub fn joint_correction(&self) -> Quat {
        fx4_to_quat(self.joint_correction)
    }

    /// Gets reached of `IKAimJob`.
    #[inline]
    pub fn clear_joint_correction(&mut self) {
        self.joint_correction = QUAT_UNIT;
    }

    /// Gets **output** reached of `IKAimJob`.
    ///
    /// True if target can be reached with IK computations.
    ///
    /// Target is considered not reachable when target is between joint and offset position.
    #[inline]
    pub fn reached(&self) -> bool {
        self.reached
    }

    /// Gets reached of `IKAimJob`.
    #[inline]
    pub fn clear_reached(&mut self) {
        self.reached = false;
    }

    /// Clears all outputs of `IKAimJob`.
    #[inline]
    pub fn clear_outs(&mut self) {
        self.clear_joint_correction();
        self.clear_reached();
    }

    /// Validates `IKAimJob` parameters.
    #[inline]
    fn validate(&self) -> bool {
        vec3_is_normalized(self.forward)
    }

    /// Runs aim IK job's task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        if !self.validate() {
            return Err(OzzError::InvalidJob);
        }

        let inv_joint = self.joint.invert();

        let joint_to_target_js = inv_joint.transform_point(self.target);
        let joint_to_target_js_len2 = vec3_length2_s(joint_to_target_js);

        let offsetted_forward = Self::compute_offsetted_forward(self.forward, self.offset, joint_to_target_js);
        self.reached = offsetted_forward.is_some();
        if !self.reached || (joint_to_target_js_len2.cmp_eq(ZERO).to_bitmask() & 0x1 == 0x1) {
            self.joint_correction = QUAT_UNIT;
            return Ok(());
        }

        let offsetted_forward = offsetted_forward.unwrap();

        let joint_to_target_rot_js = quat_from_vectors(offsetted_forward, joint_to_target_js);
        let corrected_up_js = quat_transform_vector(joint_to_target_rot_js, self.up);

        let pole_vector_js = inv_joint.transform_vector(self.pole_vector);
        let ref_joint_normal_js = vec3_cross(pole_vector_js, joint_to_target_js);
        let joint_normal_js = vec3_cross(corrected_up_js, joint_to_target_js);
        let ref_joint_normal_js_len2 = vec3_length2_s(ref_joint_normal_js);
        let joint_normal_js_len2 = vec3_length2_s(joint_normal_js);
        let denoms = fx4_set_z(
            fx4_set_y(joint_to_target_js_len2, joint_normal_js_len2),
            ref_joint_normal_js_len2,
        );

        let rotate_plane_axis_js;
        let rotate_plane_js;
        if denoms.cmp_ne(ZERO).to_bitmask() & 0x7 == 0x7 {
            let rsqrts = denoms.sqrt().recip();
            rotate_plane_axis_js = joint_to_target_js * fx4_splat_x(rsqrts);

            let rotate_plane_cos_angle = vec3_dot_s(
                joint_normal_js * fx4_splat_y(rsqrts),
                ref_joint_normal_js * fx4_splat_z(rsqrts),
            );
            let axis_flip = fx4_sign(fx4_splat_x(vec3_dot_s(ref_joint_normal_js, corrected_up_js)));
            let rotate_plane_axis_flipped_js = fx4_xor(rotate_plane_axis_js, axis_flip);
            rotate_plane_js = quat_from_cos_angle(
                rotate_plane_axis_flipped_js,
                rotate_plane_cos_angle.fast_max(NEG_ONE).fast_min(ONE), // clamp elements between -1.0 and 1.0
            );
        } else {
            rotate_plane_axis_js = joint_to_target_js * fx4_splat_x(denoms.sqrt().recip());
            rotate_plane_js = QUAT_UNIT;
        }

        let twisted = if self.twist_angle != 0.0 {
            let twist_ss = quat_from_axis_angle(rotate_plane_axis_js, f32x4::splat(self.twist_angle));
            quat_mul(quat_mul(twist_ss, rotate_plane_js), joint_to_target_rot_js)
        } else {
            quat_mul(rotate_plane_js, joint_to_target_rot_js)
        };

        let twisted_fu = quat_positive_w(twisted);
        if self.weight < 1.0 {
            let simd_weight = f32x4::splat(self.weight).fast_max(ZERO);
            self.joint_correction = quat_normalize(fx4_lerp(QUAT_UNIT, twisted_fu, simd_weight));
        } else {
            self.joint_correction = twisted_fu;
        }
        Ok(())
    }

    fn compute_offsetted_forward(forward: f32x4, offset: f32x4, target: f32x4) -> Option<f32x4> {
        let ao_l = vec3_dot_s(forward, offset);
        let ac_l2 = vec3_length2_s(offset) - ao_l * ao_l;
        let r2 = vec3_length2_s(target);
        if ac_l2.cmp_gt(r2).to_bitmask() & 0x1 == 0x1 {
            return None;
        }
        let ai_l = (r2 - ac_l2).sqrt();
        let offsetted_forward = offset + forward * fx4_splat_x(ai_l - ao_l);
        Some(offsetted_forward)
    }
}

#[cfg(test)]
mod ik_aim_job_tests {
    use core::f32::consts;
    use glam::Vec3;
    use wasm_bindgen_test::*;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_validity() {
        let mut job = IKAimJob::default();
        job.set_forward(Vec3A::new(0.5, 0.0, 0.0));
        assert!(!job.validate());

        let mut job = IKAimJob::default();
        job.set_forward(Vec3A::Z);
        assert!(job.validate());
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_correction() {
        let parents = [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::Y),
            Mat4::from_rotation_x(consts::FRAC_PI_3),
            Mat4::from_scale(Vec3::new(2.0, 2.0, 2.0)),
            Mat4::from_scale(Vec3::new(1.0, 2.0, 1.0)),
            Mat4::from_scale(Vec3::new(-3.0, -3.0, -3.0)),
        ];

        for parent in parents {
            let mut job = IKAimJob::default();
            job.set_joint(parent);
            job.set_forward(Vec3A::X);
            job.set_up(Vec3A::Y);
            job.set_pole_vector(parent.transform_vector3a(Vec3A::Y));

            {
                // x
                job.set_target(parent.transform_point3a(Vec3A::X));
                job.run().unwrap();
                assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            }

            {
                // -x
                job.set_target(parent.transform_point3a(-Vec3A::X));
                job.run().unwrap();
                assert!(job
                    .joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, consts::PI), 2e-3));
            }

            {
                // z
                job.set_target(parent.transform_point3a(Vec3A::Z));
                job.run().unwrap();
                assert!(job
                    .joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, -consts::FRAC_PI_2), 2e-3));
            }

            {
                // -z
                job.set_target(parent.transform_point3a(-Vec3A::Z));
                job.run().unwrap();
                assert!(job
                    .joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, consts::FRAC_PI_2), 2e-3));
            }

            {
                // 45 up y
                job.set_target(parent.transform_point3a(Vec3A::new(1.0, 1.0, 0.0)));
                job.run().unwrap();
                assert!(job
                    .joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_4), 2e-3));
            }

            {
                // 45 up y
                job.set_target(parent.transform_point3a(Vec3A::new(2.0, 2.0, 0.0)));
                job.run().unwrap();
                assert!(job
                    .joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_4), 2e-3));
            }
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_forward() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_target(Vec3A::X);
        job.set_up(Vec3A::Y);
        job.set_pole_vector(Vec3A::Y);

        {
            job.set_forward(Vec3A::X);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_forward(-Vec3A::X);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, -consts::PI), 2e-3));
        }

        {
            job.set_forward(Vec3A::Z);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_up() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_target(Vec3A::X);
        job.set_pole_vector(Vec3A::Y);

        {
            job.set_up(Vec3A::Y);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_up(-Vec3A::Y);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, consts::PI), 2e-3));
        }

        {
            // up z
            job.set_up(Vec3A::Z);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, -consts::FRAC_PI_2), 2e-3));
        }

        {
            // up 2z
            job.set_up(Vec3A::Z * 2.0);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, -consts::FRAC_PI_2), 2e-3));
        }

        {
            // up small z
            job.set_up(Vec3A::Z * 1e-9);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, -consts::FRAC_PI_2), 2e-3));
        }

        {
            // up 0z
            job.set_up(Vec3A::ZERO);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_pole() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_target(Vec3A::X);
        job.set_forward(Vec3A::X);
        job.set_up(Vec3A::Y);

        {
            job.set_pole_vector(Vec3A::Y);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_pole_vector(-Vec3A::Y);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, consts::PI), 2e-3));
        }

        {
            // pole z
            job.set_pole_vector(Vec3A::Z);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, consts::FRAC_PI_2), 2e-3));
        }

        {
            // pole 2z
            job.set_pole_vector(Vec3A::Z * 2.0);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, consts::FRAC_PI_2), 2e-3));
        }

        {
            // pole small z
            job.set_pole_vector(Vec3A::Z * 1e-9);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_offset() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_target(Vec3A::X);
        job.set_forward(Vec3A::X);
        job.set_up(Vec3A::Y);
        job.set_pole_vector(Vec3A::Y);

        {
            // no offset
            job.set_offset(Vec3A::ZERO);
            job.run().unwrap();
            assert!(job.reached());
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3))
        }

        {
            // inside target sphere
            job.set_offset(Vec3A::new(0.0, consts::FRAC_1_SQRT_2, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_4), 2e-3));
        }

        {
            // inside target sphere
            job.set_offset(Vec3A::new(0.5, 0.5, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_6), 2e-3));
        }

        {
            // inside target sphere
            job.set_offset(Vec3A::new(-0.5, 0.5, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_6), 2e-3));
        }

        {
            // inside target sphere
            job.set_offset(Vec3A::new(0.5, 0.0, 0.5));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, consts::FRAC_PI_6), 2e-3));
        }

        {
            // on target sphere
            job.set_offset(Vec3A::new(0.0, 1.0, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
        }

        {
            // outside target sphere
            job.set_offset(Vec3A::new(0.0, 2.0, 0.0));
            job.run().unwrap();
            assert!(!job.reached());
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // inside target sphere
            job.set_offset(Vec3A::new(0.0, 1.0, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_twist() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_target(Vec3A::X);
        job.set_forward(Vec3A::X);
        job.set_up(Vec3A::Y);

        {
            job.set_pole_vector(Vec3A::Y);
            job.set_twist_angle(0.0);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_pole_vector(Vec3A::Y);
            job.set_twist_angle(consts::FRAC_PI_2);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, consts::FRAC_PI_2), 2e-3));
        }

        {
            job.set_pole_vector(Vec3A::Y);
            job.set_twist_angle(-consts::FRAC_PI_2);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, -consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_aligned_target_up() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_forward(Vec3A::X);
        job.set_pole_vector(Vec3A::Y);

        {
            // no alinged
            job.set_target(Vec3A::X);
            job.set_up(Vec3A::Y);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // alinged Y
            job.set_target(Vec3A::Y);
            job.set_up(Vec3A::Y);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
        }

        {
            // alinged 2Y
            job.set_target(Vec3A::Y);
            job.set_up(Vec3A::Y * 2.0);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
        }

        {
            // alinged -2Y
            job.set_target(-Vec3A::Y * 2.0);
            job.set_up(Vec3A::Y);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_aligned_target_pole() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_forward(Vec3A::X);
        job.set_up(Vec3A::Y);

        {
            job.set_target(Vec3A::X);
            job.set_pole_vector(Vec3A::Y);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_target(Vec3A::Y);
            job.set_pole_vector(Vec3A::Y);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_target_too_close() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_target(Vec3A::ZERO);
        job.set_forward(Vec3A::X);
        job.set_up(Vec3A::Y);
        job.set_pole_vector(Vec3A::Y);
        job.run().unwrap();
        assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_weight() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::IDENTITY);
        job.set_target(Vec3A::Z);
        job.set_forward(Vec3A::X);
        job.set_up(Vec3A::Y);
        job.set_pole_vector(Vec3A::Y);

        {
            // 1 weight
            job.set_weight(1.0);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, -consts::FRAC_PI_2), 2e-3));
        }

        {
            // weight > 1
            job.set_weight(2.0);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, -consts::FRAC_PI_2), 2e-3));
        }

        {
            // half weight
            job.set_weight(0.5);
            job.run().unwrap();
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, -consts::FRAC_PI_4), 2e-3));
        }

        {
            // 0 weight
            job.set_weight(0.0);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // weight < 0
            job.set_weight(0.0);
            job.run().unwrap();
            assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_zero_scale() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::ZERO);
        job.run().unwrap();
        assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
    }
}
