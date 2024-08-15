//!
//! Track data structure definition.
//!

use glam::{Quat, Vec2, Vec3, Vec4};
use std::fmt::Debug;
use std::io::Read;

use crate::archive::{Archive, ArchiveRead};
use crate::base::OzzError;

/// Value type that can be stored in a `Track`.
pub trait TrackValue
where
    Self: Debug + Default + Copy + Clone + PartialEq + ArchiveRead<Self>,
{
    /// Ozz file tag in '.ozz' file for `Archive`.
    fn tag() -> &'static str;

    /// Linear interpolation between two values.
    fn lerp(a: Self, b: Self, t: f32) -> Self;

    // Compare two values with a maximum difference.
    fn abs_diff_eq(a: Self, b: Self, diff: f32) -> bool;
}

impl TrackValue for f32 {
    #[inline]
    fn tag() -> &'static str {
        "ozz-float_track"
    }

    #[inline]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    #[inline]
    fn abs_diff_eq(a: f32, b: f32, diff: f32) -> bool {
        (a - b).abs() <= diff
    }
}

impl TrackValue for Vec2 {
    #[inline]
    fn tag() -> &'static str {
        "ozz-float2_track"
    }

    #[inline]
    fn lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        Vec2::lerp(a, b, t)
    }

    #[inline]
    fn abs_diff_eq(a: Vec2, b: Vec2, diff: f32) -> bool {
        Vec2::abs_diff_eq(a, b, diff)
    }
}

impl TrackValue for Vec3 {
    #[inline]
    fn tag() -> &'static str {
        "ozz-float3_track"
    }

    #[inline]
    fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        Vec3::lerp(a, b, t)
    }

    #[inline]
    fn abs_diff_eq(a: Vec3, b: Vec3, diff: f32) -> bool {
        Vec3::abs_diff_eq(a, b, diff)
    }
}

impl TrackValue for Vec4 {
    #[inline]
    fn tag() -> &'static str {
        "ozz-float4_track"
    }

    #[inline]
    fn lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        Vec4::lerp(a, b, t)
    }

    #[inline]
    fn abs_diff_eq(a: Vec4, b: Vec4, diff: f32) -> bool {
        Vec4::abs_diff_eq(a, b, diff)
    }
}

impl TrackValue for Quat {
    #[inline]
    fn tag() -> &'static str {
        "ozz-quat_track"
    }

    #[inline]
    fn lerp(a: Quat, b: Quat, t: f32) -> Quat {
        Quat::lerp(a, b, t)
    }

    #[inline]
    fn abs_diff_eq(a: Quat, b: Quat, diff: f32) -> bool {
        Quat::abs_diff_eq(a, b, diff)
    }
}

/// Runtime user-channel track data.
///
/// Keyframe ratios, values and interpolation mode are all store as separate buffers in order
/// to access the cache coherently. Ratios are usually accessed/read alone from the jobs that
/// all start by looking up the keyframes to interpolate indeed.
#[derive(Debug, Default)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Track<V: TrackValue> {
    key_count: u32,
    ratios: Vec<f32>,
    values: Vec<V>,
    steps: Vec<u8>,
    name: String,
}

impl<V: TrackValue> Track<V> {
    /// `Track` resource file tag for `Archive`.
    #[inline]
    pub fn tag() -> &'static str {
        V::tag()
    }

    /// `Track` resource file version for `Archive`.
    #[inline]
    pub fn version() -> u32 {
        1
    }

    #[cfg(test)]
    pub(crate) fn from_raw(values: &[V], ratios: &[f32], steps: &[u8]) -> Result<Track<V>, OzzError> {
        if values.len() != ratios.len() || (values.len() + 7) / 8 != steps.len() {
            return Err(OzzError::custom("Invalid arguments"));
        }
        Ok(Track {
            key_count: values.len() as u32,
            ratios: ratios.to_vec(),
            values: values.to_vec(),
            steps: steps.to_vec(),
            name: String::new(),
        })
    }

    /// Reads an `Track` from an `Archive`.
    pub fn from_archive(archive: &mut Archive<impl Read>) -> Result<Track<V>, OzzError> {
        if archive.tag() != Self::tag() {
            return Err(OzzError::InvalidTag);
        }
        if archive.version() != Self::version() {
            return Err(OzzError::InvalidVersion);
        }

        let key_count: u32 = archive.read()?;
        let name_len: u32 = archive.read()?;

        let ratios: Vec<f32> = archive.read_vec(key_count as usize)?;
        let values: Vec<V> = archive.read_vec(key_count as usize)?;
        let steps: Vec<u8> = archive.read_vec((key_count + 7) as usize / 8)?;

        let mut name = String::new();
        if name_len != 0 {
            let buf = archive.read_vec(name_len as usize)?;
            name = String::from_utf8(buf).map_err(|e| e.utf8_error())?;
        }

        Ok(Track {
            key_count,
            ratios,
            values,
            steps,
            name,
        })
    }

    /// Reads an `Track` from a file path.
    #[cfg(not(feature = "wasm"))]
    pub fn from_path<P: AsRef<std::path::Path>>(path: P) -> Result<Track<V>, OzzError> {
        let mut archive = Archive::from_path(path)?;
        Track::from_archive(&mut archive)
    }

    // Only for wasm test in NodeJS environment.
    #[cfg(all(feature = "wasm", feature = "nodejs"))]
    pub fn from_path(path: &str) -> Result<Track<V>, OzzError> {
        let mut archive = Archive::from_path(path)?;
        Track::from_archive(&mut archive)
    }
}

impl<V: TrackValue> Track<V> {
    /// The key count in the track.
    #[inline]
    pub fn key_count(&self) -> usize {
        self.key_count as usize
    }

    /// Keyframe values.
    #[inline]
    pub fn values(&self) -> &[V] {
        &self.values
    }

    /// Keyframe ratios (0 is the beginning of the track, 1 is the end).
    #[inline]
    pub fn ratios(&self) -> &[f32] {
        &self.ratios
    }

    /// Keyframe modes (1 bit per key): 1 for step, 0 for linear.
    #[inline]
    pub fn steps(&self) -> &[u8] {
        &self.steps
    }

    /// Track name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
}
