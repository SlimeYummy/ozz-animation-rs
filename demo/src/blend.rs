use bevy::prelude::*;
use bevy::tasks::futures_lite::future::try_zip;
use ozz_animation_rs::math::*;
use ozz_animation_rs::*;
use std::sync::{Arc, RwLock};

use crate::base::*;

pub struct OzzBlend {
    skeleton: Arc<Skeleton>,
    sample_job1: ASamplingJob,
    sample_job2: ASamplingJob,
    sample_job3: ASamplingJob,
    blending_job: ABlendingJob,
    l2m_job: ALocalToModelJob,
    models: Arc<RwLock<Vec<Mat4>>>,
    bone_trans: Vec<OzzTransform>,
    spine_trans: Vec<OzzTransform>,
}

impl OzzBlend {
    pub async fn new() -> Box<dyn OzzExample> {
        let ((mut ar_skeleton, mut ar_animation1), (mut ar_animation2, mut ar_animation3)) = try_zip(
            try_zip(
                load_archive("/blend/skeleton.ozz"),
                load_archive("/blend/animation1.ozz"),
            ),
            try_zip(
                load_archive("/blend/animation2.ozz"),
                load_archive("/blend/animation3.ozz"),
            ),
        )
        .await
        .unwrap();

        let skeleton = Arc::new(Skeleton::from_archive(&mut ar_skeleton).unwrap());
        let animation1 = Arc::new(Animation::from_archive(&mut ar_animation1).unwrap());
        let animation2 = Arc::new(Animation::from_archive(&mut ar_animation2).unwrap());
        let animation3 = Arc::new(Animation::from_archive(&mut ar_animation3).unwrap());

        let mut ob = OzzBlend {
            skeleton: skeleton.clone(),
            sample_job1: SamplingJob::default(),
            sample_job2: SamplingJob::default(),
            sample_job3: SamplingJob::default(),
            blending_job: BlendingJob::default(),
            l2m_job: LocalToModelJob::default(),
            models: Arc::new(RwLock::new(vec![Mat4::default(); skeleton.num_joints()])),
            bone_trans: Vec::new(),
            spine_trans: Vec::new(),
        };

        ob.sample_job1.set_animation(animation1.clone());
        ob.sample_job1
            .set_context(SamplingContext::new(animation1.num_tracks()));
        let sample_out1 = ozz_abuf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
        ob.sample_job1.set_output(sample_out1.clone());

        ob.sample_job2.set_animation(animation2.clone());
        ob.sample_job2
            .set_context(SamplingContext::new(animation2.num_tracks()));
        let sample_out2 = ozz_abuf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
        ob.sample_job2.set_output(sample_out2.clone());

        ob.sample_job3.set_animation(animation3.clone());
        ob.sample_job3
            .set_context(SamplingContext::new(animation3.num_tracks()));
        let sample_out3 = ozz_abuf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
        ob.sample_job3.set_output(sample_out3.clone());

        ob.blending_job.set_skeleton(skeleton.clone());
        let blending_out = ozz_abuf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
        ob.blending_job.set_output(blending_out.clone());
        ob.blending_job
            .layers_mut()
            .push(BlendingLayer::new(sample_out1.clone()));
        ob.blending_job
            .layers_mut()
            .push(BlendingLayer::new(sample_out2.clone()));
        ob.blending_job
            .layers_mut()
            .push(BlendingLayer::new(sample_out3.clone()));

        ob.l2m_job.set_skeleton(skeleton.clone());
        ob.l2m_job.set_input(blending_out.clone());
        ob.l2m_job.set_output(ob.models.clone());

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

        ob.bone_trans.reserve(bone_count);
        ob.spine_trans.reserve(spine_count);
        return Box::new(ob);
    }
}

impl OzzExample for OzzBlend {
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
        let duration = self.sample_job1.animation().unwrap().duration();
        let scaled_time = time.elapsed_seconds() * 0.5;
        let ratio = (scaled_time % duration) / duration;

        self.sample_job1.set_ratio(ratio);
        self.sample_job1.run().unwrap();
        self.sample_job2.set_ratio(ratio);
        self.sample_job2.run().unwrap();
        self.sample_job3.set_ratio(ratio);
        self.sample_job3.run().unwrap();

        let which = (scaled_time / duration).floor() as i32 % 3;
        if which == 0 {
            // animation1
            self.blending_job.layers_mut()[0].weight = 1.0;
            self.blending_job.layers_mut()[1].weight = 0.0;
            self.blending_job.layers_mut()[2].weight = 0.0;
        } else if which == 1 {
            // animation2
            self.blending_job.layers_mut()[0].weight = 0.0;
            self.blending_job.layers_mut()[1].weight = 1.0;
            self.blending_job.layers_mut()[2].weight = 1.0;
        } else {
            // blend
            self.blending_job.layers_mut()[0].weight = (1.0 - 2.0 * ratio).clamp(0.0, 1.0); // 0%=1.0, 50%=0.0, 100%=0.0
            self.blending_job.layers_mut()[2].weight = (2.0 * ratio - 1.0).clamp(0.0, 1.0); // 0%=0.0, 50%=0.0, 100%=1.0
            if ratio < 0.5 {
                // 0%=0.0, 50%=1.0, 100%=0.0
                self.blending_job.layers_mut()[1].weight = (2.0 * ratio).clamp(0.0, 1.0);
            } else {
                self.blending_job.layers_mut()[1].weight = (2.0 * (1.0 - ratio)).clamp(0.0, 1.0);
            }
        }

        self.blending_job.run().unwrap();
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
