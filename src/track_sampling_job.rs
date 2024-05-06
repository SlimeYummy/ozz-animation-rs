//!
//! Track sampling job.
//!

use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;

use crate::base::{OzzError, OzzObj};
use crate::math::f32_clamp_or_max;
use crate::track::{Track, TrackValue};

/// Track sampling job implementation.
///
/// Track sampling allows to query a track value for a specified ratio.
/// This is a ratio rather than a time because tracks have no duration.
#[derive(Debug)]
pub struct TrackSamplingJob<V, T = Rc<Track<V>>>
where
    V: TrackValue,
    T: OzzObj<Track<V>>,
{
    track: Option<T>,
    ratio: f32,
    result: V,
}

pub type TrackSamplingJobRef<'t, V> = TrackSamplingJob<V, &'t Track<V>>;
pub type TrackSamplingJobRc<V> = TrackSamplingJob<V, Rc<Track<V>>>;
pub type TrackSamplingJobArc<V> = TrackSamplingJob<V, Arc<Track<V>>>;

impl<V, T> Default for TrackSamplingJob<V, T>
where
    V: TrackValue,
    T: OzzObj<Track<V>>,
{
    fn default() -> TrackSamplingJob<V, T> {
        return TrackSamplingJob {
            track: None,
            ratio: 0.0,
            result: V::default(),
        };
    }
}

impl<V, T> TrackSamplingJob<V, T>
where
    V: TrackValue,
    T: OzzObj<Track<V>>,
{
    /// Gets track of `TrackSamplingJob`.
    #[inline]
    pub fn track(&self) -> Option<&T> {
        return self.track.as_ref();
    }

    /// Sets track of `TrackSamplingJob`.
    ///
    /// Track to sample.
    #[inline]
    pub fn set_track(&mut self, track: T) {
        self.track = Some(track);
    }

    /// Clears track of `TrackSamplingJob`.
    #[inline]
    pub fn clear_track(&mut self) {
        self.track = None;
    }

    /// Gets ratio of `TrackSamplingJob`.
    #[inline]
    pub fn ratio(&self) -> f32 {
        return self.ratio;
    }

    /// Sets ratio of `TrackSamplingJob`.
    ///
    /// Ratio used to sample track, clamped in range 0.0-1.0 before job execution.
    /// 0 is the beginning of the track, 1 is the end.
    /// This is a ratio rather than a ratio because tracks have no duration.
    #[inline]
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = f32_clamp_or_max(ratio, 0.0f32, 1.0f32);
    }

    /// Gets **output** result of `TrackSamplingJob`.
    #[inline]
    pub fn result(&self) -> V {
        return self.result;
    }

    /// Clears result of `TrackSamplingJob`.
    #[inline]
    pub fn clear_result(&mut self) {
        self.result = V::default();
    }

    /// Clears all outputs of `TrackSamplingJob`.
    #[inline]
    pub fn clear_outs(&mut self) {
        self.clear_result();
    }

    /// Validates `TrackSamplingJob` parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        return self.track.is_some();
    }

    /// Runs track sampling job's task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        let track = self.track.as_ref().ok_or(OzzError::InvalidJob)?.obj();

        if track.key_count() == 0 {
            self.result = V::default();
            return Ok(());
        }

        let id1 = track
            .ratios()
            .iter()
            .position(|&x| self.ratio < x)
            .unwrap_or(track.key_count());
        let id0 = id1.saturating_sub(1);

        let id0_step = (track.steps()[id0 / 8] & (1 << (id0 & 7))) != 0;
        if id0_step || id1 == track.key_count() {
            self.result = track.values()[id0];
        } else {
            let tk0 = track.ratios()[id0];
            let tk1 = track.ratios()[id1];
            let t = (self.ratio - tk0) / (tk1 - tk0);
            let v0 = track.values()[id0];
            let v1 = track.values()[id1];
            self.result = V::lerp(v0, v1, t);
        }

        return Ok(());
    }
}

#[cfg(test)]
mod track_sampling_tests {
    use glam::{Quat, Vec2, Vec3, Vec4};
    use wasm_bindgen_test::*;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_validity() {
        let mut job: TrackSamplingJob<f32> = TrackSamplingJob::default();
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        let mut job: TrackSamplingJob<Vec3> = TrackSamplingJob::default();
        job.set_track(Rc::new(Track::default()));
        assert!(job.validate());
        assert!(job.run().is_ok());
    }

    fn execute_test<V, T>(job: &mut TrackSamplingJob<V, T>, ratio: f32, result: V)
    where
        V: TrackValue,
        T: OzzObj<Track<V>>,
    {
        job.set_ratio(ratio);
        job.run().unwrap();
        assert!(
            V::abs_diff_eq(job.result(), result, 1e-5f32),
            "{:?} != {:?}",
            job.result(),
            result
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_bounds() {
        let mut job = TrackSamplingJob::default();
        let track = Rc::new(Track::from_raw(&[0.0, 46.0, 0.0, 0.0], &[0.0, 0.5, 0.7, 1.0], &[0x2]).unwrap());
        job.set_track(track.clone());

        execute_test(&mut job, -1e-7, 0.0);
        execute_test(&mut job, 0.0, 0.0);
        execute_test(&mut job, 0.5, 46.0);
        execute_test(&mut job, 1.0, 0.0);
        execute_test(&mut job, 1.0 + 1e-7, 0.0);
        execute_test(&mut job, 1.5, 0.0);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_float() {
        let mut job = TrackSamplingJob::default();
        let track = Rc::new(Track::from_raw(&[0.0, 4.6, 9.2, 0.0, 0.0], &[0.0, 0.5, 0.7, 0.9, 1.0], &[0x2]).unwrap());
        job.set_track(track.clone());

        execute_test(&mut job, 0.0, 0.0);
        execute_test(&mut job, 0.25, 2.3);
        execute_test(&mut job, 0.5, 4.6);
        execute_test(&mut job, 0.6, 4.6);
        execute_test(&mut job, 0.7, 9.2);
        execute_test(&mut job, 0.8, 4.6);
        execute_test(&mut job, 0.9, 0.0);
        execute_test(&mut job, 1.0, 0.0);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_vec2() {
        let mut job = TrackSamplingJob::default();
        let track = Rc::new(
            Track::from_raw(
                &[
                    Vec2::new(0.0, 0.0),
                    Vec2::new(2.3, 4.6),
                    Vec2::new(4.6, 9.2),
                    Vec2::new(0.0, 0.0),
                    Vec2::new(0.0, 0.0),
                ],
                &[0.0, 0.5, 0.7, 0.9, 1.0],
                &[0x2],
            )
            .unwrap(),
        );
        job.set_track(track.clone());

        execute_test(&mut job, 0.0, Vec2::new(0.0, 0.0));
        execute_test(&mut job, 0.25, Vec2::new(1.15, 2.3));
        execute_test(&mut job, 0.5, Vec2::new(2.3, 4.6));
        execute_test(&mut job, 0.6, Vec2::new(2.3, 4.6));
        execute_test(&mut job, 0.7, Vec2::new(4.6, 9.2));
        execute_test(&mut job, 0.8, Vec2::new(2.3, 4.6));
        execute_test(&mut job, 0.9, Vec2::new(0.0, 0.0));
        execute_test(&mut job, 1.0, Vec2::new(0.0, 0.0));
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_vec3() {
        let mut job = TrackSamplingJob::default();
        let track = Rc::new(
            Track::from_raw(
                &[
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 2.3, 4.6),
                    Vec3::new(0.0, 4.6, 9.2),
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 0.0),
                ],
                &[0.0, 0.5, 0.7, 0.9, 1.0],
                &[0x2],
            )
            .unwrap(),
        );
        job.set_track(track.clone());

        execute_test(&mut job, 0.0, Vec3::new(0.0, 0.0, 0.0));
        execute_test(&mut job, 0.25, Vec3::new(0.0, 1.15, 2.3));
        execute_test(&mut job, 0.5, Vec3::new(0.0, 2.3, 4.6));
        execute_test(&mut job, 0.6, Vec3::new(0.0, 2.3, 4.6));
        execute_test(&mut job, 0.7, Vec3::new(0.0, 4.6, 9.2));
        execute_test(&mut job, 0.8, Vec3::new(0.0, 2.3, 4.6));
        execute_test(&mut job, 0.9, Vec3::new(0.0, 0.0, 0.0));
        execute_test(&mut job, 1.0, Vec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_vec4() {
        let mut job = TrackSamplingJob::default();
        let track = Rc::new(
            Track::from_raw(
                &[
                    Vec4::new(0.0, 0.0, 0.0, 0.0),
                    Vec4::new(0.0, 2.3, 0.0, 4.6),
                    Vec4::new(0.0, 4.6, 0.0, 9.2),
                    Vec4::new(0.0, 0.0, 0.0, 0.0),
                    Vec4::new(0.0, 0.0, 0.0, 0.0),
                ],
                &[0.0, 0.5, 0.7, 0.9, 1.0],
                &[0x2],
            )
            .unwrap(),
        );
        job.set_track(track.clone());

        execute_test(&mut job, 0.0, Vec4::new(0.0, 0.0, 0.0, 0.0));
        execute_test(&mut job, 0.25, Vec4::new(0.0, 1.15, 0.0, 2.3));
        execute_test(&mut job, 0.5, Vec4::new(0.0, 2.3, 0.0, 4.6));
        execute_test(&mut job, 0.6, Vec4::new(0.0, 2.3, 0.0, 4.6));
        execute_test(&mut job, 0.7, Vec4::new(0.0, 4.6, 0.0, 9.2));
        execute_test(&mut job, 0.8, Vec4::new(0.0, 2.3, 0.0, 4.6));
        execute_test(&mut job, 0.9, Vec4::new(0.0, 0.0, 0.0, 0.0));
        execute_test(&mut job, 1.0, Vec4::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_quat() {
        let mut job = TrackSamplingJob::default();
        let track = Rc::new(
            Track::from_raw(
                &[
                    Quat::from_xyzw(0.70710677, 0.0, 0.0, 0.70710677),
                    Quat::from_xyzw(0.0, 0.70710677, 0.0, 0.70710677),
                    Quat::from_xyzw(0.70710677, 0.0, 0.0, 0.70710677),
                    Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
                    Quat::from_xyzw(0.0, 0.0, 0.0, 1.0),
                ],
                &[0.0, 0.5, 0.7, 0.9, 1.0],
                &[0x2],
            )
            .unwrap(),
        );
        job.set_track(track.clone());

        execute_test(&mut job, 0.0, Quat::from_xyzw(0.70710677, 0.0, 0.0, 0.70710677));
        execute_test(&mut job, 0.1, Quat::from_xyzw(0.61721331, 0.15430345, 0.0, 0.77151674));
        execute_test(&mut job, 0.4999999, Quat::from_xyzw(0.0, 0.70710677, 0.0, 0.70710677));
        execute_test(&mut job, 0.5, Quat::from_xyzw(0.0, 0.70710677, 0.0, 0.70710677));
        execute_test(&mut job, 0.6, Quat::from_xyzw(0.0, 0.70710677, 0.0, 0.70710677));
        execute_test(&mut job, 0.7, Quat::from_xyzw(0.70710677, 0.0, 0.0, 0.7071067));
        execute_test(&mut job, 0.8, Quat::from_xyzw(0.38268333, 0.0, 0.0, 0.92387962));
        execute_test(&mut job, 0.9, Quat::from_xyzw(0.0, 0.0, 0.0, 1.0));
        execute_test(&mut job, 1.0, Quat::from_xyzw(0.0, 0.0, 0.0, 1.0));
    }
}
