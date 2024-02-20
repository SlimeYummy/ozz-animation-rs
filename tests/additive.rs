use glam::{Mat4, Vec4};
use ozz_animation_rs::*;
use std::rc::Rc;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct TestData {
    ratio: f32,
    sample_out_base: Vec<SoaTransform>,
    sample_ctx_base: SamplingContext,
    blending_out: Vec<SoaTransform>,
    l2m_out: Vec<Mat4>,
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct TestDataInit {
    sample_out_splay: Vec<SoaTransform>,
    sample_ctx_splay: SamplingContext,
    sample_out_curl: Vec<SoaTransform>,
    sample_ctx_curl: SamplingContext,
}

#[test]
fn test_additive() {
    run_additive(
        5..=5,
        |_| {},
        |_, data| test_utils::compare_with_cpp("additive", "additive", &data.l2m_out, 1e-6).unwrap(),
    );
}

#[cfg(feature = "rkyv")]
#[test]
fn test_additive_deterministic() {
    run_additive(
        -1..=11,
        |data| test_utils::compare_with_rkyv("additive", "additive_init", data).unwrap(),
        |ratio, data| test_utils::compare_with_rkyv("additive", &format!("additive{:+.2}", ratio), data).unwrap(),
    );
}

fn run_additive<I, T1, T2>(range: I, tester1: T1, tester2: T2)
where
    I: Iterator<Item = i32>,
    T1: Fn(&TestDataInit),
    T2: Fn(f32, &TestData),
{
    let skeleton = Rc::new(Skeleton::from_file("./resource/additive/skeleton.ozz").unwrap());
    let animation_base = Rc::new(Animation::from_file("./resource/additive/animation_base.ozz").unwrap());
    let animation_splay = Rc::new(Animation::from_file("./resource/additive/animation_splay_additive.ozz").unwrap());
    let animation_curl = Rc::new(Animation::from_file("./resource/additive/animation_curl_additive.ozz").unwrap());

    let mut sample_job_base: SamplingJob = SamplingJob::default();
    sample_job_base.set_animation(animation_base.clone());
    sample_job_base.set_context(SamplingContext::new(animation_base.num_tracks()));
    let sample_out_base = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job_base.set_output(sample_out_base.clone());

    let mut blending_job = BlendingJob::default();
    blending_job.set_skeleton(skeleton.clone());
    let blending_out = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    blending_job.set_output(blending_out.clone());

    let mut layer_base = BlendingLayer::new(sample_out_base.clone());
    layer_base.weight = 0.0;
    layer_base.joint_weights = vec![Vec4::splat(1.0); skeleton.num_soa_joints()];
    layer_base.transform = sample_out_base.clone();

    let left_hand = skeleton.joint_by_name("Lefthand").unwrap();
    skeleton.iter_depth_first(left_hand, |joint, _| {
        let joint = joint as usize;
        layer_base.joint_weights[joint / 4][joint % 4] = 0.0;
    });

    let right_hand = skeleton.joint_by_name("RightHand").unwrap();
    skeleton.iter_depth_first(right_hand, |joint, _| {
        let joint = joint as usize;
        layer_base.joint_weights[joint / 4][joint % 4] = 0.0;
    });

    blending_job.layers_mut().push(layer_base);

    let mut l2m_job: LocalToModelJob = LocalToModelJob::default();
    l2m_job.set_skeleton(skeleton.clone());
    l2m_job.set_input(blending_out.clone());
    let l2m_out = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);
    l2m_job.set_output(l2m_out.clone());

    let mut sample_job_splay: SamplingJob = SamplingJob::default();
    sample_job_splay.set_animation(animation_splay.clone());
    sample_job_splay.set_context(SamplingContext::new(animation_splay.num_tracks()));
    let sample_out_splay = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job_splay.set_output(sample_out_splay.clone());

    sample_job_splay.set_ratio(0.0); // Only needs the first frame pose
    sample_job_splay.run().unwrap();
    blending_job
        .additive_layers_mut()
        .push(BlendingLayer::new(sample_out_splay.clone()));

    let mut sample_job_curl: SamplingJob = SamplingJob::default();
    sample_job_curl.set_animation(animation_curl.clone());
    sample_job_curl.set_context(SamplingContext::new(animation_curl.num_tracks()));
    let sample_out_curl = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job_curl.set_output(sample_out_curl.clone());

    sample_job_curl.set_ratio(0.0); // Only needs the first frame pose
    sample_job_curl.run().unwrap();
    blending_job
        .additive_layers_mut()
        .push(BlendingLayer::new(sample_out_curl.clone()));

    tester1(&TestDataInit {
        sample_out_splay: sample_out_splay.vec().unwrap().clone(),
        sample_ctx_splay: sample_job_splay.context().unwrap().clone_without_animation_id(),
        sample_out_curl: sample_out_curl.vec().unwrap().clone(),
        sample_ctx_curl: sample_job_curl.context().unwrap().clone_without_animation_id(),
    });

    for i in range {
        let ratio = i as f32 / 10.0;

        sample_job_base.set_ratio(ratio);
        sample_job_base.run().unwrap();

        let t = ratio.clamp(0.0, 1.0);
        blending_job.additive_layers_mut()[0].weight = 0.5 + f32_cos(t * 1.7) * 0.5;
        blending_job.additive_layers_mut()[1].weight = 0.5 + f32_cos(t * 2.5) * 0.5;

        blending_job.run().unwrap();

        l2m_job.run().unwrap();

        tester2(
            ratio,
            &TestData {
                ratio,
                sample_out_base: sample_out_base.vec().unwrap().clone(),
                sample_ctx_base: sample_job_base.context().unwrap().clone_without_animation_id(),
                blending_out: blending_out.vec().unwrap().clone(),
                l2m_out: l2m_out.vec().unwrap().clone(),
            },
        );
    }
}