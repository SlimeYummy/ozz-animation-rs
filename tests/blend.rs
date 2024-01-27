#![cfg(feature = "rkyv")]

use glam::Mat4;
use ozz_animation_rs::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::rc::Rc;

#[derive(Debug, PartialEq, Archive, Serialize, Deserialize)]
struct TestData {
    ratio: f32,
    sample_out1: Vec<SoaTransform>,
    sample_ctx1: SamplingContext,
    sample_out2: Vec<SoaTransform>,
    sample_ctx2: SamplingContext,
    sample_out3: Vec<SoaTransform>,
    sample_ctx3: SamplingContext,
    blending_out: Vec<SoaTransform>,
    l2m_out: Vec<Mat4>,
}

#[test]
fn test_deterministic_blend() {
    let skeleton = Rc::new(Skeleton::from_file("./resource/blend/skeleton.ozz").unwrap());
    let animation1 = Rc::new(Animation::from_file("./resource/blend/animation1.ozz").unwrap());
    let animation2 = Rc::new(Animation::from_file("./resource/blend/animation2.ozz").unwrap());
    let animation3 = Rc::new(Animation::from_file("./resource/blend/animation3.ozz").unwrap());

    let mut sample_job1: SamplingJob = SamplingJob::default();
    sample_job1.set_animation(animation1.clone());
    sample_job1.set_context(SamplingContext::new(animation1.num_tracks()));
    let sample_out1 = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job1.set_output(sample_out1.clone());

    let mut sample_job2: SamplingJob = SamplingJob::default();
    sample_job2.set_animation(animation2.clone());
    sample_job2.set_context(SamplingContext::new(animation2.num_tracks()));
    let sample_out2 = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job2.set_output(sample_out2.clone());

    let mut sample_job3: SamplingJob = SamplingJob::default();
    sample_job3.set_animation(animation3.clone());
    sample_job3.set_context(SamplingContext::new(animation3.num_tracks()));
    let sample_out3 = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job3.set_output(sample_out3.clone());

    let mut blending_job = BlendingJob::default();
    blending_job.set_skeleton(skeleton.clone());
    let blending_out = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    blending_job.set_output(blending_out.clone());
    blending_job.layers_mut().push(BlendingLayer::new(sample_out1.clone()));
    blending_job.layers_mut().push(BlendingLayer::new(sample_out2.clone()));
    blending_job.layers_mut().push(BlendingLayer::new(sample_out3.clone()));

    let mut l2m_job: LocalToModelJob = LocalToModelJob::default();
    l2m_job.set_skeleton(skeleton.clone());
    l2m_job.set_input(blending_out.clone());
    let l2m_out = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);
    l2m_job.set_output(l2m_out.clone());

    for i in -1..=11 {
        let r = i as f32 / 10.0;

        sample_job1.set_ratio(r);
        sample_job1.run().unwrap();
        sample_job2.set_ratio(r);
        sample_job2.run().unwrap();
        sample_job3.set_ratio(r);
        sample_job3.run().unwrap();

        blending_job.layers_mut()[0].weight = (1.0 - 2.0 * r).clamp(0.0, 1.0); // 0%=1.0, 50%=0.0, 100%=0.0
        blending_job.layers_mut()[2].weight = (2.0 * r - 1.0).clamp(0.0, 1.0); // 0%=0.0, 50%=0.0, 100%=1.0
                                                                               // 0%=0.0, 50%=1.0, 100%=0.0
        if r < 0.5 {
            blending_job.layers_mut()[1].weight = (2.0 * r).clamp(0.0, 1.0);
        } else {
            blending_job.layers_mut()[1].weight = (2.0 * (1.0 - r)).clamp(0.0, 1.0);
        }
        blending_job.run().unwrap();

        l2m_job.run().unwrap();

        let data = TestData {
            ratio: r,
            sample_out1: sample_out1.vec().unwrap().clone(),
            sample_ctx1: sample_job1.context().unwrap().clone_without_animation_id(),
            sample_out2: sample_out2.vec().unwrap().clone(),
            sample_ctx2: sample_job2.context().unwrap().clone_without_animation_id(),
            sample_out3: sample_out3.vec().unwrap().clone(),
            sample_ctx3: sample_job3.context().unwrap().clone_without_animation_id(),
            blending_out: blending_out.vec().unwrap().clone(),
            l2m_out: l2m_out.vec().unwrap().clone(),
        };

        test_utils::compare_with_rkyv("blend", &format!("blend{:+.2}", r), &data).unwrap();
    }
}
