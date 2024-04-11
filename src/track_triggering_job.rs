use std::fmt::Debug;
use std::rc::Rc;
use std::sync::Arc;

use crate::base::{OzzError, OzzObj};

use crate::track::Track;

/// Structure of an edge as detected by the job.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Edge {
    /// Ratio at which track value crossed threshold.
    ratio: f32,
    /// true is edge is rising (getting higher than threshold).
    rising: bool,
}

impl Edge {
    /// Creates a new `Edge`.
    pub fn new(ratio: f32, rising: bool) -> Edge {
        return Edge {
            ratio: ratio,
            rising: rising,
        };
    }
}

#[derive(Debug)]
pub struct TrackTriggeringJob<T = Rc<Track<f32>>>
where
    T: OzzObj<Track<f32>>,
{
    track: Option<T>,
    from: f32,
    to: f32,
    threshold: f32,
}

pub type TrackTriggeringJobRef<'t> = TrackTriggeringJob<&'t Track<f32>>;
pub type TrackTriggeringJobRc = TrackTriggeringJob<Rc<Track<f32>>>;
pub type TrackTriggeringJobArc = TrackTriggeringJob<Arc<Track<f32>>>;

impl<T> Default for TrackTriggeringJob<T>
where
    T: OzzObj<Track<f32>>,
{
    fn default() -> TrackTriggeringJob<T> {
        return TrackTriggeringJob {
            track: None,
            from: 0.0,
            to: 0.0,
            threshold: 0.0,
        };
    }
}

impl<T> TrackTriggeringJob<T>
where
    T: OzzObj<Track<f32>>,
{
    /// Gets track of `TrackTriggeringJob`.
    #[inline]
    pub fn track(&self) -> Option<&T> {
        return self.track.as_ref();
    }

    /// Sets track of `TrackTriggeringJob`.
    ///
    /// Track to sample.
    #[inline]
    pub fn set_track(&mut self, track: T) {
        self.track = Some(track);
    }

    /// Clears track of `TrackTriggeringJob`.
    #[inline]
    pub fn clear_track(&mut self) {
        self.track = None;
    }

    /// Gets from of `TrackTriggeringJob`.
    #[inline]
    pub fn from(&self) -> f32 {
        return self.from;
    }

    /// Sets from of `TrackTriggeringJob`.
    ///
    /// Input range. 0 is the beginning of the track, 1 is the end.
    ///
    /// `from` can be of any sign, any order, and any range. The job will perform accordingly:
    ///
    /// - If difference between `from` and `to` is greater than 1, the iterator will loop multiple
    ///   times on the track.
    /// - If `from` is greater than `to`, then the track is processed backward (rising edges in
    ///   forward become falling ones).
    #[inline]
    pub fn set_from(&mut self, from: f32) {
        self.from = from;
    }

    /// Gets to of `TrackTriggeringJob`.
    #[inline]
    pub fn to(&self) -> f32 {
        return self.to;
    }

    /// Sets to of `TrackTriggeringJob`.
    ///
    /// Input range. 0 is the beginning of the track, 1 is the end.
    ///
    /// `to` can be of any sign, any order, and any range. The job will perform accordingly:
    ///
    /// - If difference between `from` and `to` is greater than 1, the iterator will loop multiple
    ///   times on the track.
    /// - If `from` is greater than `to`, then the track is processed backward (rising edges in
    ///   forward become falling ones).
    #[inline]
    pub fn set_to(&mut self, to: f32) {
        self.to = to;
    }

    /// Gets threshold of `TrackTriggeringJob`.
    #[inline]
    pub fn threshold(&self) -> f32 {
        return self.threshold;
    }

    /// Sets threshold of `TrackTriggeringJob`.
    ///
    /// Edge detection threshold value.
    ///
    /// A rising edge is detected as soon as the track value becomes greater than the threshold.
    ///
    /// A falling edge is detected as soon as the track value becomes smaller or equal than the threshold.
    #[inline]
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Validates `TrackTriggeringJob` parameters.
    #[inline]
    pub fn validate(&self) -> bool {
        return self.track.is_some();
    }

    /// Runs track triggering job's task.
    /// The validate job before any operation is performed.
    ///
    /// Returns an iterator of `Edge` that represents the detected edges.
    pub fn run<'t>(&'t mut self) -> Result<TrackTriggeringIter<'t, T>, OzzError> {
        if !self.track.is_some() {
            return Err(OzzError::InvalidJob);
        }
        return Ok(TrackTriggeringIter::new(self));
    }
}

#[derive(Debug)]
pub struct TrackTriggeringIter<'t, T>
where
    T: OzzObj<Track<f32>>,
{
    job: &'t TrackTriggeringJob<T>,
    track: Option<&'t Track<f32>>, // also used as end marker
    outer: f32,
    inner: isize,
}

impl<'t, T> TrackTriggeringIter<'t, T>
where
    T: OzzObj<Track<f32>>,
{
    fn new(job: &'t TrackTriggeringJob<T>) -> TrackTriggeringIter<'t, T> {
        let track = job.track().unwrap().obj();
        let end = job.from == job.to;
        return TrackTriggeringIter {
            job: job,
            track: if end { None } else { Some(track) },
            outer: job.from.floor(),
            inner: if job.from < job.to {
                0
            } else {
                track.key_count() as isize - 1
            },
        };
    }

    fn detect_edge(&self, it0: usize, it1: usize, forward: bool) -> Option<Edge> {
        let track = self.track.unwrap();
        let val0 = track.values()[it0];
        let val1 = track.values()[it1];

        let mut detected = false;
        let mut edge = Edge::default();
        if val0 <= self.job.threshold && val1 > self.job.threshold {
            // rising edge
            edge.rising = forward;
            detected = true;
        } else if val0 > self.job.threshold && val1 <= self.job.threshold {
            // falling edge
            edge.rising = !forward;
            detected = true;
        }

        if !detected {
            return None;
        }

        let step = (track.steps()[it0 / 8] & (1 << (it0 & 7))) != 0;
        if step {
            edge.ratio = track.ratios()[it1];
        } else {
            if it1 == 0 {
                edge.ratio = 0.0;
            } else {
                let alpha = (self.job.threshold - val0) / (val1 - val0);
                let ratio0 = track.ratios()[it0];
                let ratio1 = track.ratios()[it1];
                edge.ratio = ratio0 + (ratio1 - ratio0) * alpha;
            }
        }
        return Some(edge);
    }
}

impl<'t, T> Iterator for TrackTriggeringIter<'t, T>
where
    T: OzzObj<Track<f32>>,
{
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        let track = self.track?;

        let key_count = track.key_count() as isize;
        if self.job.to > self.job.from {
            // from => to
            while self.outer < self.job.to {
                while self.inner < key_count {
                    let it0 = if self.inner == 0 { key_count - 1 } else { self.inner - 1 };
                    if let Some(mut edge) = self.detect_edge(it0 as usize, self.inner as usize, true) {
                        edge.ratio += self.outer;
                        if edge.ratio >= self.job.from && (edge.ratio < self.job.to || self.job.to >= 1.0 + self.outer)
                        {
                            self.inner += 1;
                            return Some(edge);
                        }
                    }
                    // no edge found
                    if track.ratios()[self.inner as usize] + self.outer >= self.job.to {
                        break;
                    }
                    self.inner += 1;
                }
                self.inner = 0;
                self.outer += 1.0;
            }
        } else {
            // to => from
            while self.outer + 1.0 > self.job.to {
                while self.inner >= 0 {
                    let it0 = if self.inner == 0 { key_count - 1 } else { self.inner - 1 };
                    if let Some(mut edge) = self.detect_edge(it0 as usize, self.inner as usize, false) {
                        edge.ratio += self.outer;
                        if edge.ratio >= self.job.to
                            && (edge.ratio < self.job.from || self.job.from >= 1.0 + self.outer)
                        {
                            self.inner -= 1;
                            return Some(edge);
                        }
                    }
                    // no edge found
                    if track.ratios()[self.inner as usize] + self.outer <= self.job.to {
                        break;
                    }
                    self.inner -= 1;
                }
                self.inner = key_count - 1;
                self.outer -= 1.0;
            }
        }

        // iterator end
        self.track = None;
        return None;
    }
}

