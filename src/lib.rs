//!
//! Ozz-animation-rs is a rust version skeletal animation library with cross-platform deterministic.
//!
//! Ozz-animation-rs is based on [ozz-animation](https://github.com/guillaumeblanc/ozz-animation) library,
//! an open source c++ 3d skeletal animation library and toolset. Ozz-animation-rs only implement ozz-animation's
//! runtime part. You should use this library with ozz-animation's toolset.
//!
//! In order to introduce cross-platform deterministic, ozz-animation-rs does not simply wrap ozz-animation's
//! runtime, but rewrite the full runtime library in rust. So it can be used in network game scenarios, such as
//! lock-step networking synchronize.
//!
//! ```no_run
//! use glam::Mat4;
//! use ozz_animation_rs::*;
//! use ozz_animation_rs::math::*;
//! use std::rc::Rc;
//!
//! // Load resources
//! let skeleton = Rc::new(Skeleton::from_path("./resource/skeleton.ozz").unwrap());
//! let animation = Rc::new(Animation::from_path("./resource/animation.ozz").unwrap());
//!
//! // Init sample job
//! let mut sample_job: SamplingJob = SamplingJob::default();
//! sample_job.set_animation(animation.clone());
//! sample_job.set_context(SamplingContext::new(animation.num_tracks()));
//! let sample_out = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
//! sample_job.set_output(sample_out.clone());
//!
//! // Init local to model job
//! let mut l2m_job: LocalToModelJob = LocalToModelJob::default();
//! l2m_job.set_skeleton(skeleton.clone());
//! l2m_job.set_input(sample_out.clone());
//! let l2m_out = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);
//! l2m_job.set_output(l2m_out.clone());
//!
//! // Run the jobs
//! let ratio = 0.5;
//!
//! sample_job.set_ratio(ratio);
//! sample_job.run().unwrap();
//!
//! l2m_job.run().unwrap();
//! l2m_out.vec().unwrap(); // Outputs here, are model-space matrices
//! ```
//!

#![feature(portable_simd)]

mod animation;
mod archive;
mod base;
mod blending_job;
mod endian;
mod ik_aim_job;
mod ik_two_bone_job;
mod local_to_model_job;
mod sampling_job;
mod skeleton;

pub mod math;

pub use animation::Animation;
pub use archive::{Archive, ArchiveRead};
pub use base::*;
pub use blending_job::{ABlendingJob, BlendingJob, BlendingLayer};
pub use ik_aim_job::IKAimJob;
pub use ik_two_bone_job::IKTwoBoneJob;
pub use local_to_model_job::{ALocalToModelJob, LocalToModelJob};
pub use sampling_job::{ASamplingJob, InterpSoaFloat3, InterpSoaQuaternion, SamplingContext, SamplingJob};
pub use skeleton::Skeleton;
