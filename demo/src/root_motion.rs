use bevy::prelude::*;
use bevy::tasks::futures_lite::future::try_zip;
use ozz_animation_rs::*;
use std::f32::consts::PI;
use std::sync::{Arc, RwLock};

use crate::base::*;

pub struct OzzRootMotion {
    skeleton: Arc<Skeleton>,
    sample_job: SamplingJobArc,
    l2m_job: LocalToModelJobArc,
    models: Arc<RwLock<Vec<Mat4>>>,
    bone_trans: Vec<OzzTransform>,
    spine_trans: Vec<OzzTransform>,
    motion_pos: Arc<Track<Vec3>>,
    motion_pos_job: TrackSamplingJobArc<Vec3>,
    motion_rot: Arc<Track<Quat>>,
    motion_rot_job: TrackSamplingJobArc<Quat>,

    prev_ratio: f32,
    prev_track_pos: Vec3,
    prev_track_rot: Quat,
    character_pos: Vec3,
    character_rot: Quat,
}

impl OzzRootMotion {
    pub async fn new() -> Box<dyn OzzExample> {
        let (mut ar_skeleton, mut ar_animation) = try_zip(
            load_archive("/motion/skeleton.ozz"),
            load_archive("/motion/animation.ozz"),
        )
        .await
        .unwrap();

        let skeleton = Arc::new(Skeleton::from_archive(&mut ar_skeleton).unwrap());
        let animation = Arc::new(Animation::from_archive(&mut ar_animation).unwrap());

        let mut motion = load_archive("/motion/motion.ozz").await.unwrap();
        let motion_pos = Arc::new(Track::<Vec3>::from_archive(&mut motion).unwrap());
        let motion_rot = Arc::new(Track::<Quat>::from_archive(&mut motion).unwrap());

        let mut o = OzzRootMotion {
            skeleton: skeleton.clone(),
            sample_job: SamplingJob::default(),
            l2m_job: LocalToModelJob::default(),
            models: Arc::new(RwLock::new(vec![Mat4::default(); skeleton.num_joints()])),
            bone_trans: Vec::new(),
            spine_trans: Vec::new(),
            motion_pos,
            motion_pos_job: TrackSamplingJob::default(),
            motion_rot,
            motion_rot_job: TrackSamplingJob::default(),

            prev_ratio: 0.0,
            prev_track_pos: Vec3::ZERO,
            prev_track_rot: Quat::IDENTITY,
            character_pos: Vec3::ZERO,
            character_rot: Quat::IDENTITY,
        };

        o.sample_job.set_animation(animation.clone());
        o.sample_job.set_context(SamplingContext::new(animation.num_tracks()));
        let sample_out = Arc::new(RwLock::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
        o.sample_job.set_output(sample_out.clone());

        o.l2m_job.set_skeleton(skeleton.clone());
        o.l2m_job.set_input(sample_out.clone());
        o.l2m_job.set_output(o.models.clone());

        let mut bone_count = 0;
        let mut spine_count = 0;
        for i in 0..skeleton.num_joints() {
            let parent_id = skeleton.joint_parent(i);
            if parent_id as i32 == SKELETON_NO_PARENT {
                continue;
            }
            bone_count += 1;
            spine_count += 1;
            if skeleton.is_leaf(i as i16) {
                spine_count += 1;
            }
        }

        o.bone_trans.reserve(bone_count);
        o.spine_trans.reserve(spine_count);

        o.motion_pos_job.set_track(o.motion_pos.clone());
        o.motion_rot_job.set_track(o.motion_rot.clone());
        Box::new(o)
    }
}

impl OzzExample for OzzRootMotion {
    fn root(&self) -> Mat4 {
        self.models.buf().unwrap()[0]
    }

    fn bone_trans(&self) -> &[OzzTransform] {
        &self.bone_trans
    }

    fn spine_trans(&self) -> &[OzzTransform] {
        &self.spine_trans
    }

    fn update(&mut self, time: Time) {
        let duration = self.sample_job.animation().unwrap().duration();
        let ratio = (time.elapsed_secs() % duration) / duration;

        let mut pos_diff;
        let mut rot_diff;
        if ratio >= self.prev_ratio {
            self.motion_pos_job.set_ratio(ratio);
            self.motion_pos_job.run().unwrap();
            pos_diff = self.motion_pos_job.result() - self.prev_track_pos;

            self.motion_rot_job.set_ratio(ratio);
            self.motion_rot_job.run().unwrap();
            rot_diff = self.motion_rot_job.result() * self.prev_track_rot.inverse();
        } else {
            self.motion_pos_job.set_ratio(1.0);
            self.motion_pos_job.run().unwrap();
            pos_diff = self.motion_pos_job.result() - self.prev_track_pos;
            self.motion_pos_job.set_ratio(ratio);
            self.motion_pos_job.run().unwrap();
            pos_diff += self.motion_pos_job.result();

            self.motion_rot_job.set_ratio(1.0);
            self.motion_rot_job.run().unwrap();
            rot_diff = self.motion_rot_job.result() * self.prev_track_rot.inverse();
            self.motion_rot_job.set_ratio(ratio);
            self.motion_rot_job.run().unwrap();
            rot_diff *= self.motion_rot_job.result();
        }

        let angular_vel = Quat::from_rotation_y(-0.5 * PI * time.delta_secs());
        self.character_rot = self.character_rot * rot_diff * angular_vel;
        self.character_pos += self.character_rot * pos_diff;

        let root = Mat4::from_rotation_translation(self.character_rot, self.character_pos);
        self.prev_ratio = ratio;
        self.prev_track_rot = self.motion_rot_job.result();
        self.prev_track_pos = self.motion_pos_job.result();

        self.sample_job.set_ratio(ratio);
        self.sample_job.run().unwrap();
        self.l2m_job.set_root(&root);
        self.l2m_job.run().unwrap();

        self.bone_trans.clear();
        self.spine_trans.clear();

        let modals = self.models.buf().unwrap();
        for (i, current) in modals.iter().enumerate() {
            let parent_id = self.skeleton.joint_parent(i);
            if parent_id as i32 == SKELETON_NO_PARENT {
                continue;
            }
            let parent = &modals[parent_id as usize];

            let current_pos = current.w_axis.xyz();
            let parent_pos = parent.w_axis.xyz();
            let scale: f32 = (current_pos - parent_pos).length();

            let bone_dir = (current_pos - parent_pos).normalize();
            let dot1 = Vec3::dot(bone_dir, parent.x_axis.xyz());
            let dot2 = Vec3::dot(bone_dir, parent.z_axis.xyz());
            let binormal = if dot1.abs() < dot2.abs() {
                parent.x_axis.xyz()
            } else {
                parent.z_axis.xyz()
            };

            let bone_rot_y = Vec3::cross(binormal, bone_dir).normalize();
            let bone_rot_z = Vec3::cross(bone_dir, bone_rot_y).normalize();
            let bone_rot = Quat::from_mat3(&Mat3::from_cols(bone_dir, bone_rot_y, bone_rot_z));

            self.bone_trans.push(OzzTransform {
                scale,
                rotation: bone_rot,
                position: parent_pos,
            });

            let parent_rot = Quat::from_mat4(parent);
            self.spine_trans.push(OzzTransform {
                scale,
                rotation: parent_rot,
                position: parent_pos,
            });

            if self.skeleton.is_leaf(i as i16) {
                let current_rot = Quat::from_mat4(current);
                self.spine_trans.push(OzzTransform {
                    scale,
                    rotation: current_rot,
                    position: current_pos,
                });
            }
        }
    }
}
