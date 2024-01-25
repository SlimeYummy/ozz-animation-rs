use glam::{Mat4, Quat, Vec3A};
use std::simd::prelude::*;
use std::simd::StdFloat;

use crate::base::OzzError;
use crate::math::*;

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
    pub fn target(&self) -> Vec3A {
        return f32x4_to_vec3a(self.target);
    }

    pub fn set_target(&mut self, target: Vec3A) {
        self.target = f32x4_from_vec3a(target);
    }

    pub fn forward(&self) -> Vec3A {
        return f32x4_to_vec3a(self.forward);
    }

    pub fn set_forward(&mut self, forward: Vec3A) {
        self.forward = f32x4_from_vec3a(forward);
    }

    pub fn offset(&self) -> Vec3A {
        return f32x4_to_vec3a(self.offset);
    }

    pub fn set_offset(&mut self, offset: Vec3A) {
        self.offset = f32x4_from_vec3a(offset);
    }

    pub fn up(&self) -> Vec3A {
        return f32x4_to_vec3a(self.up);
    }

    pub fn set_up(&mut self, up: Vec3A) {
        self.up = f32x4_from_vec3a(up);
    }

    pub fn pole_vector(&self) -> Vec3A {
        return f32x4_to_vec3a(self.pole_vector);
    }

    pub fn set_pole_vector(&mut self, pole_vector: Vec3A) {
        self.pole_vector = f32x4_from_vec3a(pole_vector);
    }

    pub fn twist_angle(&self) -> f32 {
        return self.twist_angle;
    }

    pub fn set_twist_angle(&mut self, twist_angle: f32) {
        self.twist_angle = twist_angle;
    }

    pub fn weight(&self) -> f32 {
        return self.weight;
    }

    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }

    pub fn joint(&self) -> Mat4 {
        return self.joint.into();
    }

    pub fn set_joint(&mut self, joint: Mat4) {
        self.joint = joint.into();
    }

    pub fn joint_correction(&self) -> Quat {
        return f32x4_to_quat(self.joint_correction);
    }

    pub fn clear_joint_correction(&mut self) {
        self.joint_correction = QUAT_UNIT;
    }

    pub fn reached(&self) -> bool {
        return self.reached;
    }

    pub fn clear_reached(&mut self) {
        self.reached = false;
    }

    pub fn clear_outs(&mut self) {
        self.clear_joint_correction();
        self.clear_reached();
    }

    pub fn run(&mut self) -> Result<(), OzzError> {
        if !self.validate() {
            return Err(OzzError::InvalidJob);
        }

        let inv_joint = self.joint.invert();

        let joint_to_target_js = inv_joint.transform_point(self.target);
        let joint_to_target_js_len2 = vec3_length2_s(joint_to_target_js);

        let offsetted_forward = Self::compute_offsetted_forward(self.forward, self.offset, joint_to_target_js);
        self.reached = offsetted_forward.is_some();
        if !self.reached || (joint_to_target_js_len2.simd_eq(ZERO).to_bitmask() & 0x1 == 0x1) {
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
        let denoms = f32x4_set_z(
            f32x4_set_y(joint_to_target_js_len2, joint_normal_js_len2),
            ref_joint_normal_js_len2,
        );

        let rotate_plane_axis_js;
        let rotate_plane_js;
        if denoms.simd_ne(ZERO).to_bitmask() & 0x7 == 0x7 {
            let rsqrts = denoms.recip().sqrt();
            rotate_plane_axis_js = joint_to_target_js * f32x4_splat_x(rsqrts);

            let rotate_plane_cos_angle = vec3_dot_s(
                joint_normal_js * f32x4_splat_y(rsqrts),
                ref_joint_normal_js * f32x4_splat_z(rsqrts),
            );
            let axis_flip = as_i32x4(f32x4_splat_x(vec3_dot_s(ref_joint_normal_js, corrected_up_js))) & SIGN;
            let rotate_plane_axis_flipped_js = as_f32x4(as_i32x4(rotate_plane_axis_js) ^ axis_flip);
            rotate_plane_js = quat_from_cos_angle(
                rotate_plane_axis_flipped_js,
                rotate_plane_cos_angle.simd_clamp(-ONE, ONE),
            );
        } else {
            rotate_plane_axis_js = joint_to_target_js * f32x4_splat_x(denoms.sqrt());
            rotate_plane_js = QUAT_UNIT;
        }

        let twisted;
        if self.twist_angle != 0.0 {
            let twist_ss = quat_from_axis_angle(rotate_plane_axis_js, f32x4::splat(self.twist_angle));
            twisted = quat_mul(quat_mul(twist_ss, rotate_plane_js), joint_to_target_rot_js);
        } else {
            twisted = quat_mul(rotate_plane_js, joint_to_target_rot_js);
        }

        let twisted_fu = as_f32x4(as_i32x4(twisted) ^ (SIGN & f32x4_splat_w(twisted).simd_lt(ZERO).to_int()));
        if self.weight < 1.0 {
            let simd_weight = f32x4::splat(self.weight).simd_max(ZERO);
            self.joint_correction = quat_normalize(f32x4_lerp(QUAT_UNIT, twisted_fu, simd_weight));
        } else {
            self.joint_correction = twisted_fu;
        }
        return Ok(());
    }

    fn validate(&self) -> bool {
        return vec3_is_normalized(self.forward);
    }

    fn compute_offsetted_forward(forward: f32x4, offset: f32x4, target: f32x4) -> Option<f32x4> {
        let ao_l = vec3_dot_s(forward, offset);
        let ac_l2 = vec3_length2_s(offset) - ao_l * ao_l;
        let r2 = vec3_length2_s(target);
        if ac_l2.simd_gt(r2).to_bitmask() & 0x1 == 0x1 {
            return None;
        }
        let ai_l = (r2 - ac_l2).sqrt();
        let offsetted_forward = offset + forward * f32x4_splat_x(ai_l - ao_l);
        return Some(offsetted_forward);
    }
}

#[cfg(test)]
mod ik_aim_job_tests {
    use core::f32::consts;
    use glam::Vec3;

    use super::*;

    #[test]
    fn test_validity() {
        let mut job = IKAimJob::default();
        job.set_forward(Vec3A::new(0.5, 0.0, 0.0));
        assert!(!job.validate());

        let mut job = IKAimJob::default();
        job.set_forward(Vec3A::Z);
        assert!(job.validate());
    }

    #[test]
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
            println!(
                "{:?} {:?}",
                job.joint_correction(),
                Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_4)
            );
            assert!(job
                .joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, -consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
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
    fn test_zero_scale() {
        let mut job = IKAimJob::default();
        job.set_joint(Mat4::ZERO);
        job.run().unwrap();
        assert!(job.joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
    }
}
