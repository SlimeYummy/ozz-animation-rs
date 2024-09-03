//!
//! Ozz-animation-rs is a rust version skeletal animation library with cross-platform deterministic.
//!
//! Ozz-animation-rs is based on [ozz-animation](https://github.com/guillaumeblanc/ozz-animation) library,
//! an open source C++ 3d skeletal animation library and toolset. Ozz-animation-rs only implement ozz-animation's
//! runtime part. You should use this library with ozz-animation's toolset.
//!
//! In order to introduce cross-platform deterministic, ozz-animation-rs does not simply wrap ozz-animation's
//! runtime, but rewrite the full runtime library in rust. So it can be used in network game scenarios, such as
//! lock-step networking synchronize.
//!
//! ```no_run
//! use glam::Mat4;
//! use ozz_animation_rs::*;
//! use std::cell::RefCell;
//! use std::rc::Rc;
//!
//! // Load resources
//! let skeleton = Rc::new(Skeleton::from_path("./resource/playback/skeleton.ozz").unwrap());
//! let animation = Rc::new(Animation::from_path("./resource/playback/animation.ozz").unwrap());
//!
//! // Init sample job (Rc style)
//! let mut sample_job: SamplingJobRc = SamplingJob::default();
//! sample_job.set_animation(animation.clone());
//! sample_job.set_context(SamplingContext::new(animation.num_tracks()));
//! let sample_out = Rc::new(RefCell::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
//! sample_job.set_output(sample_out.clone());
//!
//! // Init local to model job (Ref style)
//! let mut l2m_job: LocalToModelJobRef = LocalToModelJob::default();
//! l2m_job.set_skeleton(&skeleton);
//! let sample_out_ref = sample_out.borrow();
//! l2m_job.set_input(sample_out_ref.as_ref());
//! let mut l2m_out = vec![Mat4::default(); skeleton.num_joints()];
//! l2m_job.set_output(&mut l2m_out);
//!
//! // Run the jobs
//! let ratio = 0.5;
//!
//! sample_job.set_ratio(ratio);
//! sample_job.run().unwrap();
//!
//! l2m_job.run().unwrap();
//! l2m_out.buf().unwrap(); // Outputs here, are model-space matrices
//! ```
//!

#![feature(portable_simd)]

pub mod animation;
pub mod archive;
pub mod base;
pub mod blending_job;
mod endian;
pub mod ik_aim_job;
pub mod ik_two_bone_job;
pub mod local_to_model_job;
pub mod math;
#[cfg(all(feature = "wasm", feature = "nodejs"))]
pub mod nodejs;
pub mod sampling_job;
pub mod skeleton;
pub mod skinning_job;
pub mod track;
pub mod track_sampling_job;
pub mod track_triggering_job;

pub use animation::Animation;
pub use archive::{Archive, ArchiveRead};
pub use base::{
    ozz_arc_buf, ozz_rc_buf, OzzArcBuf, OzzBuf, OzzError, OzzMutBuf, OzzObj, OzzRcBuf, SKELETON_MAX_JOINTS,
    SKELETON_MAX_SOA_JOINTS, SKELETON_NO_PARENT,
};
pub use blending_job::{BlendingContext, BlendingJob, BlendingJobArc, BlendingJobRc, BlendingJobRef, BlendingLayer};
pub use ik_aim_job::IKAimJob;
pub use ik_two_bone_job::IKTwoBoneJob;
pub use local_to_model_job::{LocalToModelJob, LocalToModelJobArc, LocalToModelJobRc, LocalToModelJobRef};
pub use math::{SoaMat4, SoaQuat, SoaTransform, SoaVec3};
pub use sampling_job::{
    InterpSoaFloat3, InterpSoaQuaternion, SamplingContext, SamplingJob, SamplingJobArc, SamplingJobRc, SamplingJobRef,
};
pub use skeleton::{JointHashMap, Skeleton};
pub use skinning_job::{SkinningJob, SkinningJobArc, SkinningJobRc, SkinningJobRef};
pub use track::Track;
pub use track_sampling_job::{TrackSamplingJob, TrackSamplingJobArc, TrackSamplingJobRc, TrackSamplingJobRef};
pub use track_triggering_job::{
    Edge, TrackTriggeringJob, TrackTriggeringJobArc, TrackTriggeringJobRc, TrackTriggeringJobRef,
};
