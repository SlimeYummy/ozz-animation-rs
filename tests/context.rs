use ozz_animation_rs::*;
use rand::{thread_rng, Rng};
use std::cell::RefCell;
use std::env;
use std::rc::Rc;
use wasm_bindgen_test::*;

mod common;

const N: i32 = 80;
const TIMES: usize = 400;

#[cfg(feature = "rkyv")]
#[test]
#[wasm_bindgen_test]
fn test_playback_contexts() {
    execute_contexts(
        "playback_contexts",
        "./resource/playback/skeleton.ozz",
        "./resource/playback/animation.ozz",
    );
}

#[cfg(feature = "rkyv")]
#[test]
#[wasm_bindgen_test]
fn test_blending_contexts() {
    execute_contexts(
        "blending_contexts",
        "./resource/blend/skeleton.ozz",
        "./resource/blend/animation1.ozz",
    );
}

#[cfg(feature = "rkyv")]
#[test]
#[wasm_bindgen_test]
fn test_additive_contexts() {
    execute_contexts(
        "additive_curl_contexts",
        "./resource/additive/skeleton.ozz",
        "./resource/additive/animation_curl_additive.ozz",
    );
    execute_contexts(
        "additive_splay_contexts",
        "./resource/additive/skeleton.ozz",
        "./resource/additive/animation_splay_additive.ozz",
    );
}

fn prepare_contexts(
    name: &str,
    skel_path: &str,
    anim_path: &str,
) -> (Rc<Skeleton>, Rc<Animation>, Vec<SamplingContext>) {
    let skeleton = Rc::new(Skeleton::from_path(skel_path).unwrap());
    let animation = Rc::new(Animation::from_path(anim_path).unwrap());

    if skeleton.num_joints() != animation.num_tracks() {
        panic!("skeleton.num_joints() != animation.num_tracks()");
    }

    let to_expected = env::var("SAVE_TO_EXPECTED").is_ok();
    if to_expected {
        let mut sample_job: SamplingJob = SamplingJob::default();
        sample_job.set_animation(animation.clone());
        sample_job.set_context(SamplingContext::new(animation.num_tracks()));
        let sample_out = Rc::new(RefCell::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
        sample_job.set_output(sample_out.clone());

        let mut contexts = Vec::new();
        for i in 0..=N {
            let ratio = (i as f32) / (N as f32);
            sample_job.set_ratio(ratio);
            sample_job.run().unwrap();
            contexts.push(sample_job.context().unwrap().clone_without_animation_id());
        }
        common::save_rkyv("context", name, &contexts, to_expected).unwrap();
        return (skeleton, animation, contexts);
    }

    let contexts: Vec<SamplingContext> = common::load_rkyv("context", name).unwrap();
    return (skeleton, animation, contexts);
}

#[derive(Debug)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct ContextPair {
    ratio1: f32,
    ctx1: SamplingContext,
    ratio2: f32,
    ctx2: SamplingContext,
}

fn execute_contexts(name: &str, skel_path: &str, anim_path: &str) {
    let (skeleton, animation, contexts) = prepare_contexts(name, skel_path, anim_path);

    let mut sample_job: SamplingJob = SamplingJob::default();
    sample_job.set_animation(animation.clone());
    sample_job.set_context(SamplingContext::new(animation.num_tracks()));
    let sample_out = Rc::new(RefCell::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
    sample_job.set_output(sample_out.clone());

    let mut rng = thread_rng();
    let rands = (0..TIMES)
        .map(|_| rng.gen_range(-10..=N + 10).clamp(0, N))
        .map(|n| (n as usize, (n as f32) / (N as f32)))
        .collect::<Vec<(usize, f32)>>();

    let mut prev_ratio = 0.0;
    sample_job.set_ratio(0.0);
    sample_job.run().unwrap();
    for (idx, ratio) in rands {
        println!("name: {} idx: {} ratio: [{}, {}]", name, idx, prev_ratio, ratio);
        sample_job.set_ratio(ratio);
        sample_job.run().unwrap();
        let context = sample_job.context().unwrap().clone_without_animation_id();
        if context != contexts[idx] {
            common::save_rkyv(
                "context",
                name,
                &ContextPair {
                    ratio1: prev_ratio,
                    ctx1: contexts[idx].clone(),
                    ratio2: ratio,
                    ctx2: context,
                },
                false,
            )
            .unwrap();
            panic!("name: {} idx: {} ratio: {}", name, idx, ratio);
        }
        prev_ratio = ratio;
    }
}
