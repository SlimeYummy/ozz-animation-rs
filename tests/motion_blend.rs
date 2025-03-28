use glam::{Mat4, Quat, Vec3};
use glam_ext::Transform3A;
use ozz_animation_rs::*;
use wasm_bindgen_test::*;

mod common;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct TestData {
    ratio: f32,
    track1_pos_out: Vec3,
    track1_rot_out: Quat,
    track2_pos_out: Vec3,
    track2_rot_out: Quat,
    blending_out: Mat4,
}

#[test]
#[wasm_bindgen_test]
fn test_motion_blend() {
    run_motion_blend(5..=5, |_, data| {
        common::compare_with_cpp("motion_blend", "motion_blend", &[data.blending_out], 1e-6).unwrap();
    });
}

#[cfg(feature = "rkyv")]
#[test]
#[wasm_bindgen_test]
fn test_motion_blend_deterministic() {
    run_motion_blend(-1..=11, |ratio, data| {
        common::compare_with_rkyv("motion_blend", &format!("motion_blend{:+.2}", ratio), data).unwrap();
    });
}

fn run_motion_blend<I, T>(range: I, tester: T)
where
    I: Iterator<Item = i32>,
    T: Fn(f32, &TestData),
{
    let mut archive = Archive::from_path("./resource/motion_blend/motion1.ozz").unwrap();
    let track1_pos = Track::<Vec3>::from_archive(&mut archive).unwrap();
    let track1_rot = Track::<Quat>::from_archive(&mut archive).unwrap();

    let mut archive = Archive::from_path("./resource/motion_blend/motion2.ozz").unwrap();
    let track2_pos = Track::<Vec3>::from_archive(&mut archive).unwrap();
    let track2_rot = Track::<Quat>::from_archive(&mut archive).unwrap();

    let mut track1_pos_job = TrackSamplingJobRef::<Vec3>::default();
    track1_pos_job.set_track(&track1_pos);
    let mut track1_rot_job = TrackSamplingJobRef::<Quat>::default();
    track1_rot_job.set_track(&track1_rot);

    let mut track2_pos_job = TrackSamplingJobRef::<Vec3>::default();
    track2_pos_job.set_track(&track2_pos);
    let mut track2_rot_job = TrackSamplingJobRef::<Quat>::default();
    track2_rot_job.set_track(&track2_rot);

    let mut blending_job = MotionBlendingJob::default();
    blending_job.layers_mut().push(MotionBlendingLayer::default());
    blending_job.layers_mut().push(MotionBlendingLayer::default());

    let mut last1 = Transform3A::default();
    let mut last2 = Transform3A::default();
    for i in range {
        let ratio = i as f32 / 10.0;

        track1_pos_job.set_ratio(ratio);
        track1_pos_job.run().unwrap();
        track1_rot_job.set_ratio(ratio);
        track1_rot_job.run().unwrap();
        track2_pos_job.set_ratio(1.0 - ratio);
        track2_pos_job.run().unwrap();
        track2_rot_job.set_ratio(1.0 - ratio);
        track2_rot_job.run().unwrap();

        blending_job.layers_mut()[0].weight = 0.4;
        blending_job.layers_mut()[0].delta = Transform3A::new(
            track1_pos_job.result() - Vec3::from(last1.translation),
            last1.rotation.conjugate() * track1_rot_job.result(),
            Vec3::ONE,
        );
        blending_job.layers_mut()[1].weight = 0.6;
        blending_job.layers_mut()[1].delta = Transform3A::new(
            track2_pos_job.result() - Vec3::from(last2.translation),
            last2.rotation.conjugate() * track2_rot_job.result(),
            Vec3::ONE,
        );
        blending_job.run().unwrap();

        last1 = Transform3A::new(track1_pos_job.result(), track1_rot_job.result(), Vec3::ONE);
        last2 = Transform3A::new(track2_pos_job.result(), track2_rot_job.result(), Vec3::ONE);

        tester(
            ratio,
            &TestData {
                ratio,
                track1_pos_out: track1_pos_job.result(),
                track1_rot_out: track1_rot_job.result(),
                track2_pos_out: track2_pos_job.result(),
                track2_rot_out: track2_rot_job.result(),
                blending_out: Mat4::from(blending_job.output()),
            },
        );
    }
}
