#![cfg(feature = "bincode")]

use bincode::{Decode, Encode};
use ozz_animation_rs::*;
use std::rc::Rc;

#[derive(Encode, Decode, PartialEq)]
struct TestData {
    ratio: f32,
    sample_out: Vec<SoaTransform>,
    sample_ctx: SamplingContext,
    l2m_out: Vec<Float4x4>,
}

#[test]
fn test_deterministic_playback() {
    let skeleton = Rc::new(Skeleton::from_file("./resource/playback/skeleton.ozz").unwrap());
    let animation = Rc::new(Animation::from_file("./resource/playback/animation.ozz").unwrap());

    if skeleton.num_joints() != animation.num_tracks() {
        panic!("skeleton.num_joints() != animation.num_tracks()");
    }

    let mut sample_job: SamplingJob = SamplingJob::default();
    sample_job.set_animation(animation.clone());
    sample_job.set_context(SamplingContext::new(animation.num_tracks()));
    let sample_out = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
    sample_job.set_output(sample_out.clone());

    let mut l2m_job: LocalToModelJob = LocalToModelJob::default();
    l2m_job.set_skeleton(skeleton.clone());
    l2m_job.set_input(sample_out.clone());
    let l2m_out = ozz_buf(vec![Float4x4::default(); skeleton.num_joints()]);
    l2m_job.set_output(l2m_out.clone());

    for i in -1..=11 {
        let t = i as f32 / 10.0;
        sample_job.set_ratio(t);
        sample_job.run().unwrap();
        l2m_job.run().unwrap();

        let data = TestData {
            ratio: t,
            sample_out: sample_out.vec().unwrap().clone(),
            sample_ctx: sample_job.context().unwrap().clone_without_animation_id(),
            l2m_out: l2m_out.vec().unwrap().clone(),
        };

        test_utils::save_to_file("playback", &format!("{:+.2}.bincode", t), &data).unwrap();

        test_utils::compare_with_file("playback", &format!("{:+.2}.bincode", t), &data).unwrap();
    }
}
