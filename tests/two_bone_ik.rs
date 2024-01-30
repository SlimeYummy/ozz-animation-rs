use glam::{Mat4, Quat, Vec3A};
use ozz_animation_rs::*;
use std::rc::Rc;

#[derive(Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct TestData {
    locals: Vec<SoaTransform>,
    models1: Vec<Mat4>,
    models2: Vec<Mat4>,
    start_correction: Quat,
    mid_correction: Quat,
    reached: bool,
}

const TARGET_EXTENT: f32 = 0.5;
const TARGET_OFFSET: Vec3A = Vec3A::new(0.0, 0.2, 0.1);

#[test]
fn test_two_bone_ik() {
    run_two_bone_ik(1..=1, |_, data| {
        test_utils::compare_with_cpp("two_bone_ik", "two_bone_ik", &data.models2, 1e-6).unwrap();
    });
}

#[cfg(feature = "rkyv")]
#[test]
fn test_two_bone_ik_deterministic() {
    run_two_bone_ik(0..=10, |idx, data| {
        test_utils::compare_with_rkyv("two_bone_ik", &format!("two_bone_ik_{:02}", idx), data).unwrap();
    });
}

fn run_two_bone_ik<I, T>(range: I, tester: T)
where
    I: Iterator<Item = i32>,
    T: Fn(i32, &TestData),
{
    let skeleton = Rc::new(Skeleton::from_file("./resource/two_bone_ik/skeleton.ozz").unwrap());

    let start_joint = skeleton.joint_by_name("shoulder").unwrap();
    let mid_joint = skeleton.joint_by_name("forearm").unwrap();
    let end_joint = skeleton.joint_by_name("wrist").unwrap();

    let locals = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    let models1 = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);
    let models2 = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);

    let mut l2m_job1: LocalToModelJob = LocalToModelJob::default();
    l2m_job1.set_skeleton(skeleton.clone());
    l2m_job1.set_input(locals.clone());
    l2m_job1.set_output(models1.clone());

    let mut ik_job = IKTwoBoneJob::default();

    let mut l2m_job2: LocalToModelJob = LocalToModelJob::default();
    l2m_job2.set_skeleton(skeleton.clone());
    l2m_job2.set_input(locals.clone());
    l2m_job2.set_output(models2.clone());

    for idx in range {
        let time = idx as f32;
        let anim_extent: f32 = (1.0 - f32_cos(time)) * 0.5 * TARGET_EXTENT;
        let floor: usize = (time.abs() / (2.0 * core::f32::consts::PI)) as usize;

        let mut target = TARGET_OFFSET.to_array();
        target[floor % 3] += anim_extent;
        let target = Vec3A::from_array(target);

        // local to modal

        locals.borrow_mut().clone_from_slice(skeleton.joint_rest_poses());
        l2m_job1.run().unwrap();

        // two bone ik

        ik_job.set_target(target);
        ik_job.set_mid_axis(Vec3A::Z);
        ik_job.set_weight(1.0);
        ik_job.set_soften(0.97);
        ik_job.set_twist_angle(0.0);
        ik_job.set_pole_vector(Vec3A::new(0.0, 1.0, 0.0));
        ik_job.set_start_joint(models1.borrow()[start_joint as usize].into());
        ik_job.set_mid_joint(models1.borrow()[mid_joint as usize].into());
        ik_job.set_end_joint(models1.borrow()[end_joint as usize].into());

        ik_job.run().unwrap();

        // apply ik result, local to modal again

        models2.borrow_mut().clone_from_slice(models1.borrow().as_slice());
        {
            let mut locals_mut = locals.vec_mut().unwrap();

            let idx = start_joint as usize;
            let quat = locals_mut[idx / 4].rotation.col(idx & 3) * ik_job.start_joint_correction();
            locals_mut[idx / 4].rotation.set_col(idx & 3, quat);

            let idx = mid_joint as usize;
            let quat = locals_mut[idx / 4].rotation.col(idx & 3) * ik_job.mid_joint_correction();
            locals_mut[idx / 4].rotation.set_col(idx & 3, quat);
        }

        l2m_job2.set_from(start_joint as i32);
        l2m_job2.set_to(SKELETON_MAX_JOINTS);
        l2m_job2.run().unwrap();

        // compare and save results

        tester(
            idx,
            &TestData {
                locals: locals.vec().unwrap().clone(),
                models1: models1.vec().unwrap().clone(),
                models2: models2.vec().unwrap().clone(),
                start_correction: ik_job.start_joint_correction(),
                mid_correction: ik_job.mid_joint_correction(),
                reached: ik_job.reached(),
            },
        );
    }
}
