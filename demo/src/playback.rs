use bevy::prelude::*;
use bevy::tasks::futures_lite::future::try_zip;
use ozz_animation_rs::math::*;
use ozz_animation_rs::*;
use std::sync::{Arc, RwLock};

use crate::base::*;

pub struct OzzPlayback {
    skeleton: Arc<Skeleton>,
    sample_job: ASamplingJob,
    l2m_job: ALocalToModelJob,
    models: Arc<RwLock<Vec<Mat4>>>,
    bone_trans: Vec<OzzTransform>,
    spine_trans: Vec<OzzTransform>,
}

impl OzzPlayback {
    pub async fn new() -> Box<dyn OzzExample> {
        let (mut ar_skeleton, mut ar_animation) = try_zip(
            load_archive("/playback/skeleton.ozz"),
            load_archive("/playback/animation.ozz"),
        )
        .await
        .unwrap();

        let skeleton = Arc::new(Skeleton::from_archive(&mut ar_skeleton).unwrap());
        let animation = Arc::new(Animation::from_archive(&mut ar_animation).unwrap());

        let mut oc = OzzPlayback {
            skeleton: skeleton.clone(),
            sample_job: SamplingJob::default(),
            l2m_job: LocalToModelJob::default(),
            models: Arc::new(RwLock::new(vec![Mat4::default(); skeleton.num_joints()])),
            bone_trans: Vec::new(),
            spine_trans: Vec::new(),
        };

        oc.sample_job.set_animation(animation.clone());
        oc.sample_job.set_context(SamplingContext::new(animation.num_tracks()));
        let sample_out = ozz_abuf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
        oc.sample_job.set_output(sample_out.clone());

        oc.l2m_job.set_skeleton(skeleton.clone());
        oc.l2m_job.set_input(sample_out.clone());
        oc.l2m_job.set_output(oc.models.clone());

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

        oc.bone_trans.reserve(bone_count);
        oc.spine_trans.reserve(spine_count);
        return Box::new(oc);
    }
}

impl OzzExample for OzzPlayback {
    fn root(&self) -> Mat4 {
        return self.models.vec().unwrap()[0];
    }

    fn bone_trans(&self) -> &[OzzTransform] {
        return &self.bone_trans;
    }

    fn spine_trans(&self) -> &[OzzTransform] {
        return &self.spine_trans;
    }

    fn update(&mut self, time: Time) {
        let duration = self.sample_job.animation().unwrap().duration();
        let ratio = (time.elapsed_seconds() % duration) / duration;
        self.sample_job.set_ratio(ratio);
        self.sample_job.run().unwrap();
        self.l2m_job.run().unwrap();

        self.bone_trans.clear();
        self.spine_trans.clear();

        let modals = self.models.vec().unwrap();
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
                scale: scale,
                rotation: bone_rot,
                position: parent_pos,
            });

            let parent_rot = Quat::from_mat4(parent);
            self.spine_trans.push(OzzTransform {
                scale: scale,
                rotation: parent_rot,
                position: parent_pos,
            });

            if self.skeleton.is_leaf(i as i16) {
                let current_rot = Quat::from_mat4(current);
                self.spine_trans.push(OzzTransform {
                    scale: scale,
                    rotation: current_rot,
                    position: current_pos,
                });
            }
        }
    }
}
