use glam::{Mat4, Vec4};
use ozz_animation_rs::*;
use std::rc::Rc;

#[derive(Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct TestData {
    ratio: f32,
    sample_out_lower: Vec<SoaTransform>,
    sample_ctx_lower: SamplingContext,
    sample_out_upper: Vec<SoaTransform>,
    sample_ctx_upper: SamplingContext,
    blending_out: Vec<SoaTransform>,
    l2m_out: Vec<Mat4>,
}

#[test]
fn test_partial_blend() {
    run_partial_blend(5..=5, |_, data| {
        test_utils::compare_with_cpp("partial_blend", "partial_blend", &data.l2m_out, 1e-5).unwrap();
    });
}

#[cfg(feature = "rkyv")]
#[test]
fn test_partial_blend_deterministic() {
    run_partial_blend(-1..=11, |ratio, data| {
        test_utils::compare_with_rkyv("partial_blend", &format!("partial_blend{:+.2}", ratio), data).unwrap();
    });
}

fn run_partial_blend<I, T>(range: I, tester: T)
where
    I: Iterator<Item = i32>,
    T: Fn(f32, &TestData),
{
    let skeleton = Rc::new(Skeleton::from_file("./resource/partial_blend/skeleton.ozz").unwrap());
    let animation_lower = Rc::new(Animation::from_file("./resource/partial_blend/animation_base.ozz").unwrap());
    let animation_upper = Rc::new(Animation::from_file("./resource/partial_blend/animation_partial.ozz").unwrap());

    let mut sample_job_lower: SamplingJob = SamplingJob::default();
    sample_job_lower.set_animation(animation_lower.clone());
    sample_job_lower.set_context(SamplingContext::new(animation_lower.num_tracks()));
    let sample_out_lower = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job_lower.set_output(sample_out_lower.clone());

    let mut sample_job_upper: SamplingJob = SamplingJob::default();
    sample_job_upper.set_animation(animation_upper.clone());
    sample_job_upper.set_context(SamplingContext::new(animation_upper.num_tracks()));
    let sample_out_upper = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job_upper.set_output(sample_out_upper.clone());

    let mut blending_job = BlendingJob::default();
    blending_job.set_skeleton(skeleton.clone());
    let blending_out = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    blending_job.set_output(blending_out.clone());

    let mut layer_lower = BlendingLayer::new(sample_out_lower.clone());
    layer_lower.weight = 0.5;
    layer_lower.joint_weights = vec![Vec4::splat(1.0); skeleton.num_soa_joints()];

    let mut layer_upper = BlendingLayer::new(sample_out_upper.clone());
    layer_upper.weight = 0.5;
    layer_upper.joint_weights = vec![Vec4::splat(0.0); skeleton.num_soa_joints()];

    let upper_root = skeleton.joint_by_name("Spine1").unwrap();
    skeleton.iter_depth_first(upper_root, |joint, _| {
        let joint = joint as usize;
        layer_lower.joint_weights[joint / 4][joint % 4] = 0.5;
        layer_upper.joint_weights[joint / 4][joint % 4] = 0.5;
    });

    blending_job.layers_mut().push(layer_lower);
    blending_job.layers_mut().push(layer_upper);

    let mut l2m_job: LocalToModelJob = LocalToModelJob::default();
    l2m_job.set_skeleton(skeleton.clone());
    l2m_job.set_input(blending_out.clone());
    let l2m_out = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);
    l2m_job.set_output(l2m_out.clone());

    for i in range {
        let ratio = i as f32 / 10.0;

        sample_job_lower.set_ratio(ratio);
        sample_job_lower.run().unwrap();

        sample_job_upper.set_ratio(ratio);
        sample_job_upper.run().unwrap();

        blending_job.run().unwrap();

        l2m_job.run().unwrap();

        tester(
            ratio,
            &TestData {
                ratio,
                sample_out_lower: sample_out_lower.vec().unwrap().clone(),
                sample_ctx_lower: sample_job_lower.context().unwrap().clone_without_animation_id(),
                sample_out_upper: sample_out_upper.vec().unwrap().clone(),
                sample_ctx_upper: sample_job_upper.context().unwrap().clone_without_animation_id(),
                blending_out: blending_out.vec().unwrap().clone(),
                l2m_out: l2m_out.vec().unwrap().clone(),
            },
        );
    }
}
