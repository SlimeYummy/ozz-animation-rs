#![feature(portable_simd)]

mod animation;
mod archive;
mod base;
mod blending_job;
mod endian;
mod ik_aim_job;
mod ik_two_bone_job;
mod local_to_model_job;
mod math;
mod sampling_job;
mod skeleton;

pub mod test_utils;

pub use animation::Animation;
pub use archive::{ArchiveReader, IArchive};
pub use base::*;
pub use blending_job::{BlendingJob, BlendingLayer};
pub use ik_aim_job::IKAimJob;
pub use ik_two_bone_job::IKTwoBoneJob;
pub use local_to_model_job::LocalToModelJob;
pub use math::{f32_acos, f32_asin, f32_cos, f32_sin, f32_sin_cos, SoaMat4, SoaQuat, SoaTransform, SoaVec3};
pub use sampling_job::{InterpSoaFloat3, InterpSoaQuaternion, SamplingContext, SamplingJob};
pub use skeleton::Skeleton;
