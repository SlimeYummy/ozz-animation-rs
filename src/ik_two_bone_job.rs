use glam::{Mat4, Quat, Vec3A};
use std::simd::prelude::*;
use std::simd::StdFloat;

use crate::math::*;
use crate::{Float4x4, OzzError};

#[derive(Debug)]
struct IKConstantSetup {
    inv_start_joint: Float4x4,
    start_mid_ms: f32x4,
    mid_end_ms: f32x4,
    start_mid_ss: f32x4,
    start_mid_ss_len2: f32x4,
    mid_end_ss_len2: f32x4,
    start_end_ss_len2: f32x4,
}

impl IKConstantSetup {
    fn new(job: &IKTwoBoneJob) -> IKConstantSetup {
        let inv_start_joint = job.start_joint.invert();
        let inv_mid_joint = job.mid_joint.invert();

        let start_ms: f32x4 = inv_mid_joint.transform_point(job.start_joint.cols[3].into());
        let end_ms: f32x4 = inv_mid_joint.transform_point(job.end_joint.cols[3].into());

        let mid_ss: f32x4 = inv_start_joint.transform_point(job.mid_joint.cols[3].into());
        let end_ss: f32x4 = inv_start_joint.transform_point(job.end_joint.cols[3].into());

        let mid_end_ss = end_ss - mid_ss;
        let start_end_ss = end_ss;
        let start_mid_ss = mid_ss;

        return IKConstantSetup {
            inv_start_joint,
            start_mid_ms: -start_ms,
            mid_end_ms: end_ms,
            start_mid_ss,
            start_mid_ss_len2: vec3_length2_s(start_mid_ss), // first
            mid_end_ss_len2: vec3_length2_s(mid_end_ss),     // first
            start_end_ss_len2: vec3_length2_s(start_end_ss), // first
        };
    }
}

#[derive(Debug)]
pub struct IKTwoBoneJob {
    target: f32x4,
    mid_axis: f32x4,
    pole_vector: f32x4,
    twist_angle: f32,
    soften: f32,
    weight: f32,
    start_joint: Float4x4,
    mid_joint: Float4x4,
    end_joint: Float4x4,

    start_joint_correction: f32x4,
    mid_joint_correction: f32x4,
    reached: bool,
}

impl Default for IKTwoBoneJob {
    fn default() -> Self {
        Self {
            target: ZERO,
            mid_axis: Z_AXIS,
            pole_vector: Y_AXIS,
            twist_angle: 0.0,
            soften: 1.0,
            weight: 1.0,
            start_joint: Float4x4::identity(),
            mid_joint: Float4x4::identity(),
            end_joint: Float4x4::identity(),
            start_joint_correction: QUAT_UNIT,
            mid_joint_correction: QUAT_UNIT,
            reached: false,
        }
    }
}

impl IKTwoBoneJob {
    pub fn target(&self) -> Vec3A {
        return f32x4_to_vec3a(self.target);
    }

    pub fn set_target(&mut self, target: Vec3A) {
        self.target = f32x4_from_vec3a(target);
    }

    pub fn mid_axis(&self) -> Vec3A {
        return f32x4_to_vec3a(self.mid_axis);
    }

    pub fn set_mid_axis(&mut self, mid_axis: Vec3A) {
        self.mid_axis = f32x4_from_vec3a(mid_axis);
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

    pub fn soften(&self) -> f32 {
        return self.soften;
    }

    pub fn set_soften(&mut self, soften: f32) {
        self.soften = soften;
    }

    pub fn weight(&self) -> f32 {
        return self.weight;
    }

    pub fn set_weight(&mut self, weight: f32) {
        self.weight = weight;
    }

    pub fn start_joint(&self) -> Mat4 {
        return self.start_joint.into();
    }

    pub fn set_start_joint(&mut self, start_joint: Mat4) {
        self.start_joint = start_joint.into();
    }

    pub fn mid_joint(&self) -> Mat4 {
        return self.mid_joint.into();
    }

    pub fn set_mid_joint(&mut self, mid_joint: Mat4) {
        self.mid_joint = mid_joint.into();
    }

    pub fn end_joint(&self) -> Mat4 {
        return self.end_joint.into();
    }

    pub fn set_end_joint(&mut self, end_joint: Mat4) {
        self.end_joint = end_joint.into();
    }

    pub fn start_joint_correction(&self) -> Quat {
        return f32x4_to_quat(self.start_joint_correction);
    }

    pub fn clear_start_joint_correction(&mut self) {
        self.start_joint_correction = QUAT_UNIT;
    }

    pub fn mid_joint_correction(&self) -> Quat {
        return f32x4_to_quat(self.mid_joint_correction);
    }

    pub fn clear_mid_joint_correction(&mut self) {
        self.mid_joint_correction = QUAT_UNIT;
    }

    pub fn reached(&self) -> bool {
        return self.reached;
    }

    pub fn clear_reached(&mut self) {
        self.reached = false;
    }

    pub fn clear_outs(&mut self) {
        self.clear_start_joint_correction();
        self.clear_mid_joint_correction();
        self.clear_reached();
    }

    pub fn run(&mut self) -> Result<(), OzzError> {
        if !self.validate() {
            return Err(OzzError::InvalidJob);
        }

        if self.weight <= 0.0 {
            self.start_joint_correction = QUAT_UNIT;
            self.mid_joint_correction = QUAT_UNIT;
            self.reached = false;
            return Ok(());
        }

        let setup = IKConstantSetup::new(self);
        let (lreached, start_target_ss, start_target_ss_len2) = self.soften_target(&setup);
        self.reached = lreached && self.weight >= 1.0;

        let mid_rot_ms = self.compute_mid_joint(&setup, start_target_ss_len2);
        let start_rot_ss = self.compute_start_joint(&setup, mid_rot_ms, start_target_ss, start_target_ss_len2);
        self.weight_output(start_rot_ss, mid_rot_ms);

        return Ok(());
    }

    fn validate(&self) -> bool {
        return vec3_is_normalized(self.mid_axis);
    }

    fn soften_target(&self, setup: &IKConstantSetup) -> (bool, f32x4, f32x4) {
        let start_target_original_ss = setup.inv_start_joint.transform_point(self.target);
        let start_target_original_ss_len2 = vec3_length2_s(start_target_original_ss); // first
        let lengths = f32x4_set_z(
            f32x4_set_y(setup.start_mid_ss_len2, setup.mid_end_ss_len2),
            start_target_original_ss_len2,
        )
        .sqrt();
        let start_mid_ss_len = lengths; // first
        let mid_end_ss_len = simd_swizzle!(lengths, [1; 4]); // first
        let start_target_original_ss_len = simd_swizzle!(lengths, [2; 4]); // first
        let bone_len_diff_abs = (start_mid_ss_len - mid_end_ss_len).abs(); // first
        let bones_chain_len = start_mid_ss_len + mid_end_ss_len; // first
        let da = bones_chain_len * f32x4::from_array([self.soften, 0.0, 0.0, 0.0]).simd_clamp(ZERO, ONE); // da.yzw needs to be 0
        let ds = bones_chain_len - da;

        let left = f32x4_set_w(start_target_original_ss_len, ds);
        let right = f32x4_set_z(da, bone_len_diff_abs);
        let comp_mask = left.simd_gt(right).to_bitmask();

        let start_target_ss;
        let start_target_ss_len2;

        // xyw all 1, z is untested.
        if (comp_mask & 0xb) == 0xb {
            let alpha = (start_target_original_ss_len - da) * ds.recip();

            let op = f32x4_set_y(THREE, alpha + THREE);
            let op2 = op * op;
            let op4 = op2 * op2;
            let ratio = op4 * simd_swizzle!(op4, [1; 4]).recip(); // first

            let start_target_ss_len = da + ds - ds * ratio; // first
            start_target_ss_len2 = start_target_ss_len * start_target_ss_len; // first
            start_target_ss =
                start_target_original_ss * f32x4_splat_x(start_target_ss_len * start_target_original_ss_len.recip());
        // first
        } else {
            start_target_ss = start_target_original_ss; // first
            start_target_ss_len2 = start_target_original_ss_len2; // first
        }

        return ((comp_mask & 0x5) == 0x4, start_target_ss, start_target_ss_len2);
    }

    fn compute_mid_joint(&self, setup: &IKConstantSetup, start_target_ss_len2: f32x4) -> f32x4 {
        let start_mid_end_sum_ss_len2 = setup.start_mid_ss_len2 + setup.mid_end_ss_len2;
        let start_mid_end_ss_half_rlen =
            f32x4_splat_x(FRAC_1_2 * (setup.start_mid_ss_len2 * setup.mid_end_ss_len2).sqrt()); // first

        let mid_cos_angles_unclamped = (f32x4_splat_x(start_mid_end_sum_ss_len2)
            - f32x4_set_y(start_target_ss_len2, setup.start_end_ss_len2))
            * start_mid_end_ss_half_rlen; // first
        let mid_cos_angles = mid_cos_angles_unclamped.simd_clamp(NEG_ONE, ONE); // first

        let mid_corrected_angle = f32x4_acos(mid_cos_angles); // first

        let bent_side_ref = vec3_cross(setup.start_mid_ms, self.mid_axis); // first
        let bent_side_flip = simd_swizzle!(
            vec3_dot_s(bent_side_ref, setup.mid_end_ms).simd_lt(ZERO).to_int(),
            [0; 4]
        ); // first
        let mid_initial_angle = as_f32x4(as_i32x4(f32x4_acos(f32x4_splat_y(mid_cos_angles))) ^ (bent_side_flip & SIGN)); // first

        let mid_angles_diff = mid_corrected_angle - mid_initial_angle; // first

        return quat_from_axis_angle(self.mid_axis, mid_angles_diff);
    }

    fn compute_start_joint(
        &self,
        setup: &IKConstantSetup,
        mid_rot_ms: f32x4,
        start_target_ss: f32x4,
        start_target_ss_len2: f32x4,
    ) -> f32x4 {
        let pole_ss = setup.inv_start_joint.transform_vector(self.pole_vector);

        let mid_end_ss_final = setup.inv_start_joint.transform_vector(
            self.mid_joint
                .transform_vector(quat_transform_vector(mid_rot_ms, setup.mid_end_ms)),
        );
        let start_end_ss_final = setup.start_mid_ss + mid_end_ss_final;

        let end_to_target_rot_ss = quat_from_vectors(start_end_ss_final, start_target_ss);

        let mut start_rot_ss = end_to_target_rot_ss;

        if start_target_ss_len2.simd_gt(ZERO).to_bitmask() & 0x1 == 0x1 {
            let ref_plane_normal_ss = vec3_cross(start_target_ss, pole_ss);
            let ref_plane_normal_ss_len2 = vec3_length2_s(ref_plane_normal_ss); // first

            let mid_axis_ss = setup
                .inv_start_joint
                .transform_vector(self.mid_joint.transform_vector(self.mid_axis));
            let joint_plane_normal_ss = quat_transform_vector(end_to_target_rot_ss, mid_axis_ss);
            let joint_plane_normal_ss_len2 = vec3_length2_s(joint_plane_normal_ss); // first

            let rsqrts = f32x4_set_z(
                f32x4_set_y(start_target_ss_len2, ref_plane_normal_ss_len2),
                joint_plane_normal_ss_len2,
            )
            .recip()
            .sqrt();

            let rotate_plane_cos_angle = vec3_dot_s(
                ref_plane_normal_ss * f32x4_splat_y(rsqrts),
                joint_plane_normal_ss * f32x4_splat_z(rsqrts),
            ); // first

            let rotate_plane_axis_ss = start_target_ss * f32x4_splat_x(rsqrts);
            let start_axis_flip = as_i32x4(f32x4_splat_x(vec3_dot_s(joint_plane_normal_ss, pole_ss))) & SIGN;
            let rotate_plane_axis_flipped_ss = as_f32x4(as_i32x4(rotate_plane_axis_ss) ^ start_axis_flip);

            let rotate_plane_ss = quat_from_cos_angle(
                rotate_plane_axis_flipped_ss,
                rotate_plane_cos_angle.simd_clamp(NEG_ONE, ONE),
            );

            if self.twist_angle != 0.0 {
                let twist_ss = quat_from_axis_angle(rotate_plane_axis_ss, f32x4::splat(self.twist_angle));
                start_rot_ss = quat_mul(quat_mul(twist_ss, rotate_plane_ss), end_to_target_rot_ss);
            } else {
                start_rot_ss = quat_mul(rotate_plane_ss, end_to_target_rot_ss);
            }
        }

        return start_rot_ss;
    }

    fn weight_output(&mut self, start_rot: f32x4, mid_rot: f32x4) {
        let start_rot_fu = as_f32x4(as_i32x4(start_rot) ^ (SIGN & f32x4_splat_w(start_rot).simd_lt(ZERO).to_int()));
        let mid_rot_fu = as_f32x4(as_i32x4(mid_rot) ^ (SIGN & f32x4_splat_w(mid_rot).simd_lt(ZERO).to_int()));

        if self.weight < 1.0 {
            let simd_weight = f32x4::splat(self.weight).simd_max(ZERO);

            let start_lerp = f32x4_lerp(QUAT_UNIT, start_rot_fu, simd_weight);
            let mid_lerp = f32x4_lerp(QUAT_UNIT, mid_rot_fu, simd_weight);

            let rsqrts = f32x4::from_array([
                (start_lerp * start_lerp).reduce_sum(),
                (mid_lerp * mid_lerp).reduce_sum(),
                0.0,
                0.0,
            ])
            .recip()
            .sqrt();

            self.start_joint_correction = start_lerp * f32x4_splat_x(rsqrts);
            self.mid_joint_correction = mid_lerp * f32x4_splat_y(rsqrts);
        } else {
            self.start_joint_correction = start_rot_fu;
            self.mid_joint_correction = mid_rot_fu;
        }
    }
}

#[cfg(test)]
mod ik_two_bone_tests {
    use core::f32::consts;
    use glam::Vec3;

    use super::*;

    #[test]
    fn test_validity() {
        let mut job = IKTwoBoneJob::default();
        job.set_mid_axis(Vec3A::new(1.0, 2.0, 3.0));
        assert!(!job.validate());

        let mut job = IKTwoBoneJob::default();
        job.set_mid_axis(Vec3A::Z);
        assert!(job.validate());
    }

    #[test]
    fn test_start_joint_correction() {
        let base_start = Mat4::IDENTITY;
        let base_mid =
            Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::Z, core::f32::consts::FRAC_PI_2), Vec3::Y);
        let base_end = Mat4::from_translation(Vec3::X + Vec3::Y);
        let mid_axis = Vec3A::cross(
            Vec3A::from(base_start.col(3)) - Vec3A::from(base_mid.col(3)),
            Vec3A::from(base_end.col(3)) - Vec3A::from(base_mid.col(3)),
        );

        let parents = [
            Mat4::IDENTITY,
            Mat4::from_translation(Vec3::Y),
            Mat4::from_rotation_x(core::f32::consts::FRAC_PI_3),
            Mat4::from_scale(Vec3::splat(2.0)),
            Mat4::from_scale(Vec3::new(1.0, 2.0, 1.0)),
            Mat4::from_scale(Vec3::new(-3.0, -3.0, -3.0)),
        ];

        for parent in parents {
            let start = parent * base_start;
            let mid = parent * base_mid;
            let end = parent * base_end;

            let mut job = IKTwoBoneJob::default();
            job.set_pole_vector(parent.transform_vector3a(Vec3A::Y));
            job.set_mid_axis(mid_axis);
            job.set_start_joint(start);
            job.set_mid_joint(mid);
            job.set_end_joint(end);

            {
                // 0 degree
                job.set_target(parent.transform_point3a(Vec3A::new(1.0, 1.0, 0.0)));
                job.run().unwrap();
                assert_eq!(job.reached, true);
                assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
                assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            }

            {
                // 90 degree
                job.set_target(parent.transform_point3a(Vec3A::new(0.0, 1.0, 1.0)));
                job.run().unwrap();
                assert_eq!(job.reached, true);
                assert!(job
                    .start_joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, -consts::FRAC_PI_2), 2e-3));
                assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            }

            {
                // 180 degree
                job.set_target(parent.transform_point3a(Vec3A::new(-1.0, 1.0, 0.0)));
                job.run().unwrap();
                assert_eq!(job.reached, true);
                assert!(job
                    .start_joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, consts::PI), 2e-3));
                assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            }

            {
                // 270 degree
                job.set_target(parent.transform_point3a(Vec3A::new(0.0, 1.0, -1.0)));
                job.run().unwrap();
                assert_eq!(job.reached, true);
                assert!(job
                    .start_joint_correction()
                    .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, consts::FRAC_PI_2), 2e-3));
                assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            }
        }
    }

    fn new_ik_two_bone_job() -> IKTwoBoneJob {
        let start = Mat4::IDENTITY;
        let mid = Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), Vec3::Y);
        let end = Mat4::from_translation(Vec3::X + Vec3::Y);
        let mid_axis = Vec3A::cross(
            Vec3A::from(start.col(3)) - Vec3A::from(mid.col(3)),
            Vec3A::from(end.col(3)) - Vec3A::from(mid.col(3)),
        );

        let mut job = IKTwoBoneJob::default();
        job.set_start_joint(start);
        job.set_mid_joint(mid);
        job.set_end_joint(end);
        job.set_mid_axis(mid_axis);
        return job;
    }

    #[test]
    fn test_pole() {
        let mut job = new_ik_two_bone_job();

        {
            // pole Y
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(1.0, 1.0, 0.0));
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // pole Z
            job.set_pole_vector(Vec3A::Z);
            job.set_target(Vec3A::new(1.0, 0.0, 1.0));
            job.run().unwrap();
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, consts::FRAC_PI_2), 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // pole -Z
            job.set_pole_vector(-Vec3A::Z);
            job.set_target(Vec3A::new(1.0, 0.0, -1.0));
            job.run().unwrap();
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::X, -consts::FRAC_PI_2), 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // pole X
            job.set_pole_vector(Vec3A::X);
            job.set_target(Vec3A::new(1.0, -1.0, 0.0));
            job.run().unwrap();
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // pole -X
            job.set_pole_vector(-Vec3A::X);
            job.set_target(Vec3A::new(-1.0, 1.0, 0.0));
            job.run().unwrap();
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }
    }

    #[test]
    fn test_zero_scale() {
        let mut job = IKTwoBoneJob::default();
        job.set_start_joint(Mat4::ZERO);
        job.set_mid_joint(Mat4::ZERO);
        job.set_end_joint(Mat4::ZERO);

        job.run().unwrap();
        assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
    }

    #[test]
    fn test_soften() {
        let mut job = new_ik_two_bone_job();

        {
            // reachable
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.set_soften(1.0);

            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
        }

        {
            // reachable, softened
            job.set_pole_vector(Vec3A::Y);
            job.set_soften(0.5);

            job.set_target(Vec3A::new(2.0 * 0.5, 0.0, 0.0));
            job.run().unwrap();
            assert!(job.reached());

            job.set_target(Vec3A::new(2.0 * 0.4, 0.0, 0.0));
            job.run().unwrap();
            assert!(job.reached());
        }

        {
            // not reachable, softened
            job.set_pole_vector(Vec3A::Y);

            job.set_soften(0.5);
            job.set_target(Vec3A::new(2.0 * 0.6, 0.0, 0.0));
            job.run().unwrap();
            assert!(!job.reached());
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_xyzw(0.0, 0.0, -0.324383080, 0.945925772), 2e-3));
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_xyzw(0.0, 0.0, -0.124356493, 0.992237628), 2e-3));

            job.set_soften(0.0);
            job.set_target(Vec3A::new(0.0, 0.0, 0.0));
            job.run().unwrap();
            assert!(!job.reached());

            job.set_soften(0.5);
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.run().unwrap();
            assert!(!job.reached());

            job.set_soften(1.0);
            job.set_target(Vec3A::new(3.0, 0.0, 0.0));
            job.run().unwrap();
            assert!(!job.reached());
        }
    }

    #[test]
    fn test_twist() {
        let mut job = new_ik_two_bone_job();
        job.set_pole_vector(Vec3A::Y);
        job.set_target(Vec3A::new(1.0, 1.0, 0.0));

        {
            // 0 degree
            job.set_twist_angle(0.0);
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // 90 degree
            job.set_twist_angle(consts::FRAC_PI_2);
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(
                Quat::from_axis_angle(
                    Vec3::new(consts::FRAC_1_SQRT_2, consts::FRAC_1_SQRT_2, 0.0),
                    consts::FRAC_PI_2
                ),
                2e-3
            ));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // 180 degree twist
            job.set_twist_angle(consts::PI);
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(
                Quat::from_axis_angle(
                    Vec3::new(consts::FRAC_1_SQRT_2, consts::FRAC_1_SQRT_2, 0.0),
                    -consts::PI
                ),
                2e-3
            ));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // 270 degree twist
            job.set_twist_angle(2.0 * consts::PI);
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }
    }

    #[test]
    fn test_weight() {
        let mut job = new_ik_two_bone_job();

        {
            // weight 1.0
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.set_weight(1.0);
            job.run().unwrap();
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
        }

        {
            // weight > 1.0
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.set_weight(1.1);
            job.run().unwrap();
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
        }

        {
            // weight 0.0
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.set_weight(0.0);
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // weight < 0.0
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.set_weight(-0.1);
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // weight 0.5
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.set_weight(0.5);
            job.run().unwrap();
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_4), 2e-3));
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_4), 2e-3));
        }
    }

    #[test]
    fn test_pole_target_alignment() {
        let mut job = new_ik_two_bone_job();

        {
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(0.0, consts::SQRT_2, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job.start_joint_correction().is_nan());
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(0.001, consts::SQRT_2, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_4), 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_pole_vector(Vec3A::Y);
            job.set_target(Vec3A::new(0.0, 3.0, 0.0));
            job.run().unwrap();
            assert!(!job.reached());
            assert!(job.start_joint_correction().is_nan());
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    fn test_mid_axis() {
        let mut job = new_ik_two_bone_job();
        let mid_axis = job.mid_axis();

        {
            job.set_mid_axis(mid_axis);
            job.set_target(Vec3A::new(1.0, 1.0, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            job.set_mid_axis(-mid_axis);
            job.set_target(Vec3A::new(1.0, 1.0, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job
                .start_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Y, consts::PI), 2e-3));
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, consts::PI), 2e-3));
        }

        {
            job.set_end_joint(Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)));
            job.set_mid_axis(mid_axis);
            job.set_target(Vec3A::new(1.0, 1.0, 0.0));
            job.run().unwrap();
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job
                .mid_joint_correction()
                .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
        }
    }

    #[test]
    fn test_aligned_joints_and_target() {
        let mut job = IKTwoBoneJob::default();
        job.set_start_joint(Mat4::IDENTITY);
        job.set_mid_joint(Mat4::from_translation(Vec3::X));
        job.set_end_joint(Mat4::from_translation(Vec3::X).mul_scalar(2.0));
        job.set_mid_axis(Vec3A::Z);
        job.set_pole_vector(Vec3A::Y);

        {
            // reachable
            job.set_target(Vec3A::new(2.0, 0.0, 0.0));
            job.run().unwrap();
            assert!(job.reached());
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }

        {
            // unreachable
            job.set_target(Vec3A::new(3.0, 0.0, 0.0));
            job.run().unwrap();
            assert!(!job.reached());
            assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
            assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        }
    }

    #[test]
    fn test_zero_length_start_target() {
        let start = Mat4::IDENTITY;
        let mid = Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::Z, consts::FRAC_PI_2), Vec3::Y);
        let end = Mat4::from_translation(Vec3::X + Vec3::Y);

        let mut job = IKTwoBoneJob::default();
        job.set_target(Vec3A::from(start.col(3)));
        job.set_start_joint(start);
        job.set_mid_joint(mid);
        job.set_end_joint(end);

        job.run().unwrap();
        assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        assert!(job
            .mid_joint_correction()
            .abs_diff_eq(Quat::from_axis_angle(Vec3::Z, -consts::FRAC_PI_2), 2e-3));
    }

    #[test]
    fn test_zero_length_bone_chain() {
        let mut job = IKTwoBoneJob::default();
        job.set_pole_vector(Vec3A::Y);
        job.set_target(Vec3A::X);
        job.set_start_joint(Mat4::IDENTITY);
        job.set_mid_joint(Mat4::IDENTITY);
        job.set_end_joint(Mat4::IDENTITY);

        job.run().unwrap();
        assert!(!job.reached());
        assert!(job.start_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
        assert!(job.mid_joint_correction().abs_diff_eq(Quat::IDENTITY, 2e-3));
    }
}