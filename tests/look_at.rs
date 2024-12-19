use glam::{Mat4, Quat, Vec3A};
use ozz_animation_rs::math::{f32_cos, f32_sin};
use ozz_animation_rs::*;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen_test::*;

mod common;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct TestData {
    locals1: Vec<SoaTransform>,
    locals2: Vec<SoaTransform>,
    models1: Vec<Mat4>,
    models2: Vec<Mat4>,
    joint_corrections: [Quat; 4],
    reacheds: [bool; 4],
}

const JOINT_NAMES: &[&str] = &["Head", "Spine3", "Spine2", "Spine1"];

const TARGET_EXTENT: f32 = 1.0;
const TARGET_OFFSET: Vec3A = Vec3A::new(0.2, 1.5, -0.3);
const EYES_OFFSET: Vec3A = Vec3A::new(0.07, 0.1, 0.0);

#[test]
#[wasm_bindgen_test]
fn test_look_at() {
    run_look_at(1..=1, |_, data| {
        common::compare_with_cpp("look_at", "look_at", &data.models2, 1.5e-4).unwrap();
    });
}

#[cfg(feature = "rkyv")]
#[test]
#[wasm_bindgen_test]
fn test_look_at_deterministic() {
    run_look_at(0..=10, |idx, data| {
        common::compare_with_rkyv("look_at", &format!("look_at_{:02}", idx), data).unwrap();
    });
}

fn run_look_at<I, T>(range: I, tester: T)
where
    I: Iterator<Item = i32>,
    T: Fn(i32, &TestData),
{
    let skeleton = Rc::new(Skeleton::from_path("./resource/look_at/skeleton.ozz").unwrap());
    let animation = Rc::new(Animation::from_path("./resource/look_at/animation.ozz").unwrap());

    let joints_chain = [
        skeleton.joint_by_name(JOINT_NAMES[0]).unwrap(),
        skeleton.joint_by_name(JOINT_NAMES[1]).unwrap(),
        skeleton.joint_by_name(JOINT_NAMES[2]).unwrap(),
        skeleton.joint_by_name(JOINT_NAMES[3]).unwrap(),
    ];
    if !validate_joints_order(&skeleton, &joints_chain) {
        panic!("Invalid joints chain");
    }

    let locals1 = Rc::new(RefCell::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
    let locals2 = Rc::new(RefCell::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
    let models1 = Rc::new(RefCell::new(vec![Mat4::default(); skeleton.num_joints()]));
    let models2 = Rc::new(RefCell::new(vec![Mat4::default(); skeleton.num_joints()]));

    let mut sample_job: SamplingJob = SamplingJob::default();
    sample_job.set_animation(animation.clone());
    sample_job.set_context(SamplingContext::new(skeleton.num_joints()));
    sample_job.set_output(locals1.clone());

    let mut l2m_job1: LocalToModelJob = LocalToModelJob::default();
    l2m_job1.set_skeleton(skeleton.clone());
    l2m_job1.set_input(locals1.clone());
    l2m_job1.set_output(models1.clone());

    let mut ik_job = IKAimJob::default();

    let mut l2m_job2: LocalToModelJob = LocalToModelJob::default();
    l2m_job2.set_skeleton(skeleton.clone());
    l2m_job2.set_input(locals2.clone());
    l2m_job2.set_output(models2.clone());

    for idx in range {
        let time = idx as f32;
        let delta = (time / 5.0).fract();

        let animated_target = Vec3A::new(f32_sin(time * 0.5), f32_cos(time * 0.25), f32_cos(time) * 0.5 + 0.5);
        let target = animated_target * TARGET_EXTENT + TARGET_OFFSET;

        sample_job.set_ratio(delta);
        sample_job.run().unwrap();

        l2m_job1.run().unwrap();
        locals2.borrow_mut().clone_from_slice(locals1.borrow().as_slice());
        models2.borrow_mut().clone_from_slice(models1.borrow().as_slice());

        ik_job.set_pole_vector(Vec3A::Y);
        ik_job.set_target(target);

        let mut joint_corrections = [Quat::IDENTITY; 4];
        let mut reacheds = [false; 4];

        let mut previous_joint = SKELETON_NO_PARENT;
        for (idx, joint) in joints_chain.iter().enumerate() {
            ik_job.set_joint(models1.buf().unwrap()[*joint as usize]);
            ik_job.set_up(Vec3A::X);

            if idx == joints_chain.len() - 1 {
                ik_job.set_weight(1.0);
            } else {
                ik_job.set_weight(0.5);
            }

            if idx == 0 {
                ik_job.set_offset(EYES_OFFSET);
                ik_job.set_forward(Vec3A::Y);
            } else {
                let transform: Mat4 = models1.buf().unwrap()[previous_joint as usize];
                let corrected_forward_ms =
                    transform.transform_vector3a(ik_job.joint_correction().mul_vec3a(ik_job.forward()));
                let corrected_offset_ms =
                    transform.transform_point3a(ik_job.joint_correction().mul_vec3a(ik_job.offset()));

                let transform: Mat4 = models1.buf().unwrap()[*joint as usize];
                let inv_transform = transform.inverse();
                ik_job.set_offset(inv_transform.transform_point3a(corrected_offset_ms));
                ik_job.set_forward(inv_transform.transform_vector3a(corrected_forward_ms));
            }

            ik_job.run().unwrap();
            joint_corrections[idx] = ik_job.joint_correction();
            reacheds[idx] = ik_job.reached();

            {
                let mut locals_mut = locals2.borrow_mut();
                let idx = *joint as usize;
                let quat = locals_mut[idx / 4].rotation.quat(idx & 3) * ik_job.joint_correction();
                locals_mut[idx / 4].rotation.set_quat(idx & 3, quat);
            }

            previous_joint = *joint as i32;
        }

        l2m_job2.set_from(previous_joint);
        l2m_job2.run().unwrap();

        tester(
            idx,
            &TestData {
                locals1: locals1.buf().unwrap().to_vec(),
                locals2: locals2.buf().unwrap().to_vec(),
                models1: models1.buf().unwrap().to_vec(),
                models2: models2.buf().unwrap().to_vec(),
                joint_corrections,
                reacheds,
            },
        );
    }
}

fn validate_joints_order(skeleton: &Skeleton, joints: &[i16]) -> bool {
    if joints.is_empty() {
        return true;
    }

    let mut i = 1;
    let mut joint = joints[0];
    let mut parent = skeleton.joint_parent(joint as usize);
    while i != joints.len() && (joint as i32) != SKELETON_NO_PARENT {
        if parent == joints[i] {
            i += 1;
        }
        joint = parent;
        parent = skeleton.joint_parent(joint as usize);
    }

    joints.len() == i
}
