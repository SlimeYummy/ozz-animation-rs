use ozz_animation_rs::*;
use wasm_bindgen_test::*;

mod common;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct TestSamplingData {
    ratio: f32,
    result: f32,
}

#[cfg(feature = "rkyv")]
#[test]
#[wasm_bindgen_test]
fn test_track_sampling_deterministic() {
    let track = Track::<f32>::from_path("./resource/track/track.ozz").unwrap();
    let mut job: TrackSamplingJobRef<f32> = TrackSamplingJob::default();
    job.set_track(&track);

    let mut all_data = Vec::new();
    for i in -11..=21 {
        let ratio = i as f32 / 10.0;
        job.set_ratio(ratio);
        job.run().unwrap();

        all_data.push(TestSamplingData {
            ratio,
            result: job.result(),
        });
    }

    common::compare_with_rkyv("track", "track_sampling", &all_data).unwrap();
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct TestTriggeringData {
    from: f32,
    to: f32,
    results: Vec<Edge>,
}

#[cfg(feature = "rkyv")]
#[test]
#[wasm_bindgen_test]
fn test_track_triggering_deterministic() {
    let track = Track::<f32>::from_path("./resource/track/track.ozz").unwrap();
    let mut job: TrackTriggeringJobRef = TrackTriggeringJob::default();
    job.set_track(&track);

    let mut all_data = Vec::new();
    let mut j = -11;
    for i in -11..=21 {
        let from = j as f32 / 10.0;
        let to = i as f32 / 10.0;
        j = i;

        job.set_from(from);
        job.set_to(to);
        job.set_threshold(0.5);

        let results: Vec<_> = job.run().unwrap().map(|x| x).collect();
        all_data.push(TestTriggeringData { from, to, results });
    }

    common::compare_with_rkyv("track", "track_triggering", &all_data).unwrap();
}
