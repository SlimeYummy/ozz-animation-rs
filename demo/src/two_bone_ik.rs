use bevy::prelude::*;
use ozz_animation_rs::*;
use std::sync::{Arc, RwLock};

use crate::base::*;

const TARGET_EXTENT: f32 = 0.5;
const TARGET_OFFSET: Vec3 = Vec3::new(0.0, 0.2, 0.1);

pub struct OzzTwoBoneIK {
    skeleton: Arc<Skeleton>,
    l2m_job1: LocalToModelJobArc,
    ik_job: IKTwoBoneJob,
    l2m_job2: LocalToModelJobArc,
    locals: Arc<RwLock<Vec<SoaTransform>>>,
    models1: Arc<RwLock<Vec<Mat4>>>,
    models2: Arc<RwLock<Vec<Mat4>>>,
    bone_trans: Vec<OzzTransform>,
    spine_trans: Vec<OzzTransform>,
}

unsafe impl Send for OzzTwoBoneIK {}
unsafe impl Sync for OzzTwoBoneIK {}

impl OzzTwoBoneIK {
    pub async fn new() -> Box<dyn OzzExample> {
        let mut ar_skeleton = load_archive("/two_bone_ik/skeleton.ozz").await.unwrap();

        let skeleton = Arc::new(Skeleton::from_archive(&mut ar_skeleton).unwrap());

        let mut oc = OzzTwoBoneIK {
            skeleton: skeleton.clone(),
            l2m_job1: LocalToModelJob::default(),
            ik_job: IKTwoBoneJob::default(),
            l2m_job2: LocalToModelJob::default(),
            locals: Arc::new(RwLock::new(vec![SoaTransform::default(); skeleton.num_soa_joints()])),
            models1: Arc::new(RwLock::new(vec![Mat4::default(); skeleton.num_joints()])),
            models2: Arc::new(RwLock::new(vec![Mat4::default(); skeleton.num_joints()])),
            bone_trans: Vec::new(),
            spine_trans: Vec::new(),
        };

        oc.l2m_job1.set_skeleton(skeleton.clone());
        oc.l2m_job1.set_input(oc.locals.clone());
        oc.l2m_job1.set_output(oc.models1.clone());

        oc.l2m_job2.set_skeleton(skeleton.clone());
        oc.l2m_job2.set_input(oc.locals.clone());
        oc.l2m_job2.set_output(oc.models2.clone());

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
        Box::new(oc)
    }
}

impl OzzExample for OzzTwoBoneIK {
    fn root(&self) -> Mat4 {
        self.models2.buf().unwrap()[0]
    }

    fn bone_trans(&self) -> &[OzzTransform] {
        &self.bone_trans
    }

    fn spine_trans(&self) -> &[OzzTransform] {
        &self.spine_trans
    }

    fn update(&mut self, time: Time) {
        let start_joint = self.skeleton.joint_by_name("shoulder").unwrap();
        let mid_joint = self.skeleton.joint_by_name("forearm").unwrap();
        let end_joint = self.skeleton.joint_by_name("wrist").unwrap();

        let ratio = time.elapsed_seconds() % 5.0;
        let anim_extent: f32 = (1.0 - ratio.cos()) * 0.5 * TARGET_EXTENT;
        let floor: usize = (ratio.abs() / (2.0 * core::f32::consts::PI)) as usize;

        let mut target = TARGET_OFFSET.to_array();
        target[floor % 3] += anim_extent;
        let target = Vec3::from_array(target);

        // local to modal

        self.locals
            .mut_buf()
            .unwrap()
            .clone_from_slice(self.skeleton.joint_rest_poses());
        self.l2m_job1.run().unwrap();

        // two bone ik

        self.ik_job.set_target(target.into());
        self.ik_job.set_mid_axis(Vec3::Z.into());
        self.ik_job.set_weight(1.0);
        self.ik_job.set_soften(0.97);
        self.ik_job.set_twist_angle(0.0);
        self.ik_job.set_pole_vector(Vec3::new(0.0, 1.0, 0.0).into());
        self.ik_job
            .set_start_joint(self.models1.buf().unwrap()[start_joint as usize]);
        self.ik_job
            .set_mid_joint(self.models1.buf().unwrap()[mid_joint as usize]);
        self.ik_job
            .set_end_joint(self.models1.buf().unwrap()[end_joint as usize]);

        self.ik_job.run().unwrap();

        // apply ik result, local to modal again

        self.models2
            .mut_buf()
            .unwrap()
            .clone_from_slice(self.models1.buf().unwrap().as_ref());
        {
            let mut locals_mut = self.locals.mut_buf().unwrap();

            let idx = start_joint as usize;
            let quat = locals_mut[idx / 4].rotation.col(idx & 3) * self.ik_job.start_joint_correction();
            locals_mut[idx / 4].rotation.set_col(idx & 3, quat);

            let idx = mid_joint as usize;
            let quat = locals_mut[idx / 4].rotation.col(idx & 3) * self.ik_job.mid_joint_correction();
            locals_mut[idx / 4].rotation.set_col(idx & 3, quat);
        }

        self.l2m_job2.set_from(start_joint as i32);
        self.l2m_job2.set_to(SKELETON_MAX_JOINTS);
        self.l2m_job2.run().unwrap();

        // outputs

        self.bone_trans.clear();
        self.spine_trans.clear();

        let modals = self.models2.buf().unwrap();
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
