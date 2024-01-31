use glam::Mat4;
use ozz_animation_rs::*;
use std::rc::Rc;

#[derive(Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct TestData {
    ratio: f32,
    sample_out: Vec<SoaTransform>,
    sample_ctx: SamplingContext,
    l2m_out: Vec<Mat4>,
}

#[test]
fn test_playback() {
    run_playback(5..=5, |_, data| {
        test_utils::compare_with_cpp("playback", "playback", &data.l2m_out, 1e-5).unwrap()
    });
}

#[cfg(feature = "rkyv")]
#[test]
fn test_playback_deterministic() {
    run_playback(-1..=11, |ratio, data| {
        test_utils::compare_with_rkyv("playback", &format!("playback{:+.2}", ratio), data).unwrap()
    });
}

fn run_playback<I, T>(range: I, tester: T)
where
    I: Iterator<Item = i32>,
    T: Fn(f32, &TestData),
{
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
    let l2m_out = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);
    l2m_job.set_output(l2m_out.clone());

    for i in range {
        let ratio = i as f32 / 10.0;
        sample_job.set_ratio(ratio);
        sample_job.run().unwrap();
        l2m_job.run().unwrap();

        tester(
            ratio,
            &TestData {
                ratio,
                sample_out: sample_out.vec().unwrap().clone(),
                sample_ctx: sample_job.context().unwrap().clone_without_animation_id(),
                l2m_out: l2m_out.vec().unwrap().clone(),
            },
        );
    }
}
