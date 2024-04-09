use core::slice;
use glam::{Quat, Vec2, Vec3, Vec4};
use std::alloc::{self, Layout};
use std::fmt::Debug;
use std::io::Read;
use std::{mem, ptr};

use crate::archive::{Archive, ArchiveRead};
use crate::base::OzzError;

pub trait TrackValue
where
    Self: Debug + Default + Copy + Clone + PartialEq + ArchiveRead<Self>,
{
    fn lerp(a: Self, b: Self, t: f32) -> Self;
    fn abs_diff_eq(a: Self, b: Self, diff: f32) -> bool;
}

impl TrackValue for f32 {
    #[inline]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        return a + (b - a) * t;
    }

    #[inline]
    fn abs_diff_eq(a: f32, b: f32, diff: f32) -> bool {
        return (a - b).abs() <= diff;
    }
}

impl TrackValue for Vec2 {
    #[inline]
    fn lerp(a: Vec2, b: Vec2, t: f32) -> Vec2 {
        return Vec2::lerp(a, b, t);
    }

    #[inline]
    fn abs_diff_eq(a: Vec2, b: Vec2, diff: f32) -> bool {
        return Vec2::abs_diff_eq(a, b, diff);
    }
}

impl TrackValue for Vec3 {
    #[inline]
    fn lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
        return Vec3::lerp(a, b, t);
    }

    #[inline]
    fn abs_diff_eq(a: Vec3, b: Vec3, diff: f32) -> bool {
        return Vec3::abs_diff_eq(a, b, diff);
    }
}

impl TrackValue for Vec4 {
    #[inline]
    fn lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
        return Vec4::lerp(a, b, t);
    }

    #[inline]
    fn abs_diff_eq(a: Vec4, b: Vec4, diff: f32) -> bool {
        return Vec4::abs_diff_eq(a, b, diff);
    }
}

impl TrackValue for Quat {
    #[inline]
    fn lerp(a: Quat, b: Quat, t: f32) -> Quat {
        return Quat::lerp(a, b, t);
    }

    #[inline]
    fn abs_diff_eq(a: Quat, b: Quat, diff: f32) -> bool {
        return Quat::abs_diff_eq(a, b, diff);
    }
}

#[derive(Debug)]
struct TrackInner<V: TrackValue> {
    size: usize,
    key_count: usize,
    values_ptr: *mut V,
    ratios_ptr: *mut f32,
    steps_ptr: *mut u8,
    name_len: usize,
    name_ptr: *mut u8,
}

impl<V: TrackValue> Default for TrackInner<V> {
    fn default() -> Self {
        return TrackInner {
            size: 0,
            key_count: 0,
            values_ptr: ptr::null_mut(),
            ratios_ptr: ptr::null_mut(),
            steps_ptr: ptr::null_mut(),
            name_len: 0,
            name_ptr: ptr::null_mut(),
        };
    }
}

#[derive(Debug)]
pub struct Track<V: TrackValue> {
    inner: *mut TrackInner<V>,
}

impl<V: TrackValue> Clone for Track<V> {
    fn clone(&self) -> Self {
        let inner = self.inner();
        let mut track = Track::new(inner.key_count, inner.name_len);
        track.values_mut().copy_from_slice(self.values());
        track.ratios_mut().copy_from_slice(self.ratios());
        track.steps_mut().copy_from_slice(self.steps());
        if inner.name_len != 0 {
            let name = unsafe { slice::from_raw_parts_mut(track.inner_mut().name_ptr, inner.name_len) };
            name.copy_from_slice(self.name().as_bytes());
        }
        return track;
    }
}

impl<V: TrackValue> Drop for Track<V> {
    fn drop(&mut self) {
        unsafe {
            let align = usize::max(mem::align_of::<V>(), mem::align_of::<TrackInner<V>>());
            let layout = Layout::from_size_align_unchecked(self.inner().size, align);
            alloc::dealloc(self.inner as *mut u8, layout);
        }
    }
}

impl<V: TrackValue> Track<V> {
    #[inline(always)]
    fn inner(&self) -> &TrackInner<V> {
        return unsafe { &*self.inner };
    }

    #[inline(always)]
    fn inner_mut(&mut self) -> &mut TrackInner<V> {
        return unsafe { &mut *self.inner };
    }

    pub fn new(key_count: usize, name_len: usize) -> Track<V> {
        let align = usize::max(mem::align_of::<V>(), mem::align_of::<TrackInner<V>>());
        let inner_size = (mem::size_of::<TrackInner<V>>() + align - 1) & !(align - 1);
        let size = inner_size
            + key_count * mem::size_of::<V>()
            + key_count * mem::size_of::<f32>()
            + (key_count + 7) / 8 * mem::size_of::<u8>()
            + name_len;

        unsafe {
            let layout = Layout::from_size_align_unchecked(size, align);
            let mut ptr = alloc::alloc(layout);
            let mut track = Track {
                inner: ptr as *mut TrackInner<V>,
            };
            ptr = ptr.add(inner_size);

            track.inner_mut().size = size;
            track.inner_mut().key_count = key_count;

            track.inner_mut().values_ptr = ptr as *mut V;
            ptr = ptr.add(track.inner_mut().key_count * mem::size_of::<V>());
            track.inner_mut().ratios_ptr = ptr as *mut f32;
            ptr = ptr.add(track.inner_mut().key_count * mem::size_of::<f32>());
            track.inner_mut().steps_ptr = ptr as *mut u8;
            ptr = ptr.add(((track.inner_mut().key_count + 7) / 8) * mem::size_of::<u8>());
            track.inner_mut().name_len = name_len;
            track.inner_mut().name_ptr = ptr as *mut u8;
            ptr = ptr.add(name_len);
            assert_eq!(ptr, (track.inner as *mut u8).add(size));

            return track;
        }
    }

    #[cfg(test)]
    pub(crate) fn from_raw(values: &[V], ratios: &[f32], steps: &[u8]) -> Result<Track<V>, OzzError> {
        if values.len() != ratios.len() || (values.len() + 7) / 8 != steps.len() {
            return Err(OzzError::Custom("Invalid arguments".into()));
        }
        let mut track = Track::new(values.len(), 0);
        track.values_mut().copy_from_slice(values);
        track.ratios_mut().copy_from_slice(ratios);
        track.steps_mut().copy_from_slice(steps);
        return Ok(track);
    }

    pub fn from_archive(archive: &mut Archive<impl Read>) -> Result<Track<V>, OzzError> {
        // if (_version > 1) {
        //     log::Err() << "Unsupported Track version " << _version << "." << std::endl;
        //     return;
        //   }

        let key_count: u32 = archive.read()?;
        let name_len: u32 = archive.read()?;
        let mut track = Track::new(key_count as usize, name_len as usize);

        for i in 0..key_count {
            track.ratios_mut()[i as usize] = archive.read()?;
        }
        for i in 0..key_count {
            track.values_mut()[i as usize] = archive.read()?;
        }
        for i in 0..(key_count + 7) / 8 {
            track.steps_mut()[i as usize] = archive.read()?;
        }
        if name_len != 0 {
            let name = unsafe { slice::from_raw_parts_mut(track.inner_mut().name_ptr, track.inner_mut().name_len) };
            for i in 0..name_len {
                name[i as usize] = archive.read()?;
            }
            std::str::from_utf8(name)?;
        }

        return Ok(track);
    }

    /// The memory size of the context in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        return self.inner().size;
    }

    /// The key count in the track.
    #[inline]
    pub fn key_count(&self) -> usize {
        return self.inner().key_count;
    }

    /// Keyframe values.
    #[inline]
    pub fn values(&self) -> &[V] {
        return unsafe { std::slice::from_raw_parts(self.inner().values_ptr, self.inner().key_count) };
    }

    #[inline]
    pub fn values_mut(&mut self) -> &mut [V] {
        return unsafe { std::slice::from_raw_parts_mut(self.inner_mut().values_ptr, self.inner().key_count) };
    }

    /// Keyframe ratios (0 is the beginning of the track, 1 is the end).
    #[inline]
    pub fn ratios(&self) -> &[f32] {
        return unsafe { std::slice::from_raw_parts(self.inner().ratios_ptr, self.inner().key_count) };
    }

    #[inline]
    pub fn ratios_mut(&mut self) -> &mut [f32] {
        return unsafe { std::slice::from_raw_parts_mut(self.inner_mut().ratios_ptr, self.inner().key_count) };
    }

    /// Keyframe modes (1 bit per key): 1 for step, 0 for linear.
    #[inline]
    pub fn steps(&self) -> &[u8] {
        return unsafe { std::slice::from_raw_parts(self.inner().steps_ptr, (self.inner().key_count + 7) / 8) };
    }

    #[inline]
    pub fn steps_mut(&mut self) -> &mut [u8] {
        return unsafe { std::slice::from_raw_parts_mut(self.inner_mut().steps_ptr, (self.inner().key_count + 7) / 8) };
    }

    /// Track name.
    #[inline]
    pub fn name(&self) -> &str {
        unsafe {
            let name = slice::from_raw_parts(self.inner().name_ptr, self.inner().name_len);
            return std::str::from_utf8_unchecked(name);
        };
    }
}
