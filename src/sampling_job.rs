//!
//! Sampling Job.
//!

use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
use std::simd::prelude::*;
use std::sync::{Arc, RwLock};
use std::{mem, slice};

use crate::animation::{Animation, Float3Key, QuaternionKey};
use crate::base::{OzzError, OzzMutBuf, OzzObj};
use crate::math::{f32_clamp_or_max, SoaQuat, SoaTransform, SoaVec3};

/// Soa hot `SoaVec3` data to interpolate.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InterpSoaFloat3 {
    #[cfg_attr(feature = "serde", serde(with = "serde_interp"))]
    pub ratio: [f32x4; 2],
    pub value: [SoaVec3; 2],
}

#[cfg(feature = "rkyv")]
const _: () = {
    use bytecheck::CheckBytes;
    use rkyv::{from_archived, to_archived, Archive, Deserialize, Fallible, Serialize};
    use std::io::{Error, ErrorKind};

    impl Archive for InterpSoaFloat3 {
        type Archived = InterpSoaFloat3;
        type Resolver = ();

        #[inline]
        unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
            out.write(to_archived!(*self as Self));
        }
    }

    impl<S: Fallible + ?Sized> Serialize<S> for InterpSoaFloat3 {
        #[inline]
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            return Ok(());
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<InterpSoaFloat3, D> for InterpSoaFloat3 {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<InterpSoaFloat3, D::Error> {
            return Ok(from_archived!(*self));
        }
    }

    impl<C: ?Sized> CheckBytes<C> for InterpSoaFloat3 {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<InterpSoaFloat3>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            return Ok(&*value);
        }
    }
};

/// Soa hot `SoaQuat` data to interpolate.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct InterpSoaQuaternion {
    #[cfg_attr(feature = "serde", serde(with = "serde_interp"))]
    pub ratio: [f32x4; 2],
    pub value: [SoaQuat; 2],
}

#[cfg(feature = "rkyv")]
const _: () = {
    use bytecheck::CheckBytes;
    use rkyv::{from_archived, to_archived, Archive, Deserialize, Fallible, Serialize};
    use std::io::{Error, ErrorKind};

    impl Archive for InterpSoaQuaternion {
        type Archived = InterpSoaQuaternion;
        type Resolver = ();

        #[inline]
        unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
            out.write(to_archived!(*self as Self));
        }
    }

    impl<S: Fallible + ?Sized> Serialize<S> for InterpSoaQuaternion {
        #[inline]
        fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
            return Ok(());
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<InterpSoaQuaternion, D> for InterpSoaQuaternion {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<InterpSoaQuaternion, D::Error> {
            return Ok(from_archived!(*self));
        }
    }

    impl<C: ?Sized> CheckBytes<C> for InterpSoaQuaternion {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<InterpSoaQuaternion>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            return Ok(&*value);
        }
    }
};

#[cfg(feature = "serde")]
mod serde_interp {
    use serde::ser::SerializeSeq;
    use serde::{Deserialize, Deserializer, Serializer};
    use std::simd::prelude::*;

    pub(crate) fn serialize<S: Serializer>(value: &[f32x4; 2], serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(2))?;
        seq.serialize_element(value[0].as_array())?;
        seq.serialize_element(value[1].as_array())?;
        return seq.end();
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f32x4; 2], D::Error> {
        let tmp: [[f32; 4]; 2] = Deserialize::deserialize(deserializer)?;
        return Ok([f32x4::from_array(tmp[0]), f32x4::from_array(tmp[1])]);
    }
}

#[repr(align(16))]
#[derive(Debug)]
struct SamplingContextInner {
    size: usize,
    max_tracks: usize,
    max_soa_tracks: usize,
    num_outdated: usize,

    translations_ptr: *mut InterpSoaFloat3,
    rotations_ptr: *mut InterpSoaQuaternion,
    scales_ptr: *mut InterpSoaFloat3,

    translation_keys_ptr: *mut i32,
    rotation_keys_ptr: *mut i32,
    scale_keys_ptr: *mut i32,

    outdated_translations_ptr: *mut u8,
    outdated_rotations_ptr: *mut u8,
    outdated_scales_ptr: *mut u8,
}

impl Default for SamplingContextInner {
    fn default() -> SamplingContextInner {
        return SamplingContextInner {
            size: 0,
            max_tracks: 0,
            max_soa_tracks: 0,
            num_outdated: 0,

            translations_ptr: std::ptr::null_mut(),
            rotations_ptr: std::ptr::null_mut(),
            scales_ptr: std::ptr::null_mut(),

            translation_keys_ptr: std::ptr::null_mut(),
            rotation_keys_ptr: std::ptr::null_mut(),
            scale_keys_ptr: std::ptr::null_mut(),

            outdated_translations_ptr: std::ptr::null_mut(),
            outdated_rotations_ptr: std::ptr::null_mut(),
            outdated_scales_ptr: std::ptr::null_mut(),
        };
    }
}

/// Declares the context object used by the workload to take advantage of the
/// frame coherency of animation sampling.
pub struct SamplingContext {
    inner: *const SamplingContextInner,
    animation_id: u64,
    ratio: f32,

    translation_cursor: usize,
    rotation_cursor: usize,
    scale_cursor: usize,
}

unsafe impl Send for SamplingContext {}
unsafe impl Sync for SamplingContext {}

impl Debug for SamplingContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return f
            .debug_struct("SamplingContext")
            .field("mem", &self.inner)
            .field("animation_id", &self.animation_id)
            .field("ratio", &self.ratio)
            .finish();
    }
}

impl Clone for SamplingContext {
    fn clone(&self) -> Self {
        let mut ctx = SamplingContext::new(self.max_tracks());
        ctx.animation_id = self.animation_id;
        ctx.ratio = self.ratio;

        ctx.translations_mut().copy_from_slice(self.translations());
        ctx.rotations_mut().copy_from_slice(self.rotations());
        ctx.scales_mut().copy_from_slice(self.scales());

        ctx.translation_keys_mut().copy_from_slice(self.translation_keys());
        ctx.rotation_keys_mut().copy_from_slice(self.rotation_keys());
        ctx.scale_keys_mut().copy_from_slice(self.scale_keys());

        ctx.translation_cursor = self.translation_cursor;
        ctx.rotation_cursor = self.rotation_cursor;
        ctx.scale_cursor = self.scale_cursor;

        ctx.outdated_translations_mut()
            .copy_from_slice(self.outdated_translations());
        ctx.outdated_rotations_mut().copy_from_slice(self.outdated_rotations());
        ctx.outdated_scales_mut().copy_from_slice(self.outdated_scales());

        return ctx;
    }
}

impl PartialEq for SamplingContext {
    fn eq(&self, other: &Self) -> bool {
        return self.max_tracks() == other.max_tracks()
            && self.max_soa_tracks() == other.max_soa_tracks()
            && self.num_outdated() == other.num_outdated()
            && self.animation_id == other.animation_id
            && self.ratio == other.ratio
            && self.translations() == other.translations()
            && self.rotations() == other.rotations()
            && self.scales() == other.scales()
            && self.translation_keys() == other.translation_keys()
            && self.rotation_keys() == other.rotation_keys()
            && self.scale_keys() == other.scale_keys()
            && self.translation_cursor == other.translation_cursor
            && self.rotation_cursor == other.rotation_cursor
            && self.scale_cursor == other.scale_cursor
            && self.outdated_translations() == other.outdated_translations()
            && self.outdated_rotations() == other.outdated_rotations()
            && self.outdated_scales() == other.outdated_scales();
    }
}

impl Drop for SamplingContext {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.size(), mem::size_of::<f32x4>());
                alloc::dealloc(self.inner as *mut u8, layout);
            }
            self.inner = std::ptr::null_mut();
        }
    }
}

impl SamplingContext {
    #[inline(always)]
    fn inner(&self) -> &SamplingContextInner {
        return unsafe { &*self.inner };
    }

    /// Create a new `SamplingContext`
    ///
    /// * `max_tracks` - The maximum number of tracks that the context can handle.
    pub fn new(max_tracks: usize) -> SamplingContext {
        let max_soa_tracks = (max_tracks + 3) / 4;
        let max_tracks = max_soa_tracks * 4;
        let num_outdated = (max_soa_tracks + 7) / 8;
        let size = mem::size_of::<SamplingContextInner>()
            + mem::size_of::<InterpSoaFloat3>() * max_soa_tracks
            + mem::size_of::<InterpSoaQuaternion>() * max_soa_tracks
            + mem::size_of::<InterpSoaFloat3>() * max_soa_tracks
            + mem::size_of::<i32>() * max_tracks * 2 * 3
            + mem::size_of::<u8>() * 3 * num_outdated;

        unsafe {
            let layout = Layout::from_size_align_unchecked(size, mem::size_of::<f32x4>());
            let mut ptr = alloc::alloc(layout);
            let ctx = SamplingContext {
                inner: ptr as *const SamplingContextInner,
                animation_id: 0,
                ratio: 0.0,
                translation_cursor: 0,
                rotation_cursor: 0,
                scale_cursor: 0,
            };

            let inner = &mut *(ptr as *mut SamplingContextInner);
            *inner = SamplingContextInner::default();
            ptr = ptr.add(mem::size_of::<SamplingContextInner>());

            inner.size = size;
            inner.max_soa_tracks = max_soa_tracks;
            inner.max_tracks = max_tracks;
            inner.num_outdated = num_outdated;

            inner.translations_ptr = ptr as *mut InterpSoaFloat3;
            ptr = ptr.add(mem::size_of::<InterpSoaFloat3>() * inner.max_soa_tracks);
            inner.rotations_ptr = ptr as *mut InterpSoaQuaternion;
            ptr = ptr.add(mem::size_of::<InterpSoaQuaternion>() * inner.max_soa_tracks);
            inner.scales_ptr = ptr as *mut InterpSoaFloat3;
            ptr = ptr.add(mem::size_of::<InterpSoaFloat3>() * inner.max_soa_tracks);
            inner.translation_keys_ptr = ptr as *mut i32;
            ptr = ptr.add(mem::size_of::<i32>() * max_tracks * 2);
            inner.rotation_keys_ptr = ptr as *mut i32;
            ptr = ptr.add(mem::size_of::<i32>() * max_tracks * 2);
            inner.scale_keys_ptr = ptr as *mut i32;
            ptr = ptr.add(mem::size_of::<i32>() * max_tracks * 2);
            inner.outdated_translations_ptr = ptr as *mut u8;
            ptr = ptr.add(mem::size_of::<u8>() * inner.num_outdated);
            inner.outdated_rotations_ptr = ptr as *mut u8;
            ptr = ptr.add(mem::size_of::<u8>() * inner.num_outdated);
            inner.outdated_scales_ptr = ptr as *mut u8;
            ptr = ptr.add(mem::size_of::<u8>() * inner.num_outdated);
            assert_eq!(ptr, (ctx.inner as *mut u8).add(size));

            return ctx;
        };
    }

    /// Create a new `SamplingContext` from an `Animation`.
    ///
    /// * `animation` - The animation to sample. Use `animation.num_tracks()` as max_tracks.
    pub fn from_animation(animation: &Animation) -> SamplingContext {
        let mut ctx = SamplingContext::new(animation.num_tracks());
        ctx.animation_id = animation as *const _ as u64;
        return ctx;
    }

    /// Clear the `SamplingContext`.
    #[inline]
    pub fn clear(&mut self) {
        self.animation_id = 0;
        self.translation_cursor = 0;
        self.rotation_cursor = 0;
        self.scale_cursor = 0;
    }

    /// Clone the `SamplingContext` without the animation id. Usually used for serialization.
    #[inline]
    pub fn clone_without_animation_id(&self) -> SamplingContext {
        let mut ctx = self.clone();
        ctx.animation_id = 0;
        return ctx;
    }

    /// The memory size of the context in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        return self.inner().size;
    }

    /// The maximum number of SoA tracks that the context can handle.
    #[inline]
    pub fn max_soa_tracks(&self) -> usize {
        return self.inner().max_soa_tracks;
    }

    /// The maximum number of tracks that the context can handle.
    #[inline]
    pub fn max_tracks(&self) -> usize {
        return self.inner().max_tracks;
    }

    /// The number of tracks that are outdated.
    #[inline]
    pub fn num_outdated(&self) -> usize {
        return self.inner().num_outdated;
    }

    /// Soa hot data to interpolate.
    #[inline]
    pub fn translations(&self) -> &[InterpSoaFloat3] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.translations_ptr, inner.max_soa_tracks) };
    }

    #[inline]
    fn translations_mut(&mut self) -> &mut [InterpSoaFloat3] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.translations_ptr, inner.max_soa_tracks) };
    }

    /// Soa hot data to interpolate.
    #[inline]
    pub fn rotations(&self) -> &[InterpSoaQuaternion] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.rotations_ptr, inner.max_soa_tracks) };
    }

    #[inline]
    fn rotations_mut(&mut self) -> &mut [InterpSoaQuaternion] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.rotations_ptr, inner.max_soa_tracks) };
    }

    /// Soa hot data to interpolate.
    #[inline]
    pub fn scales(&self) -> &[InterpSoaFloat3] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.scales_ptr, inner.max_soa_tracks) };
    }

    #[inline]
    fn scales_mut(&mut self) -> &mut [InterpSoaFloat3] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.scales_ptr, inner.max_soa_tracks) };
    }

    /// The keys in the animation that are valid for the current time ratio.
    #[inline]
    pub fn translation_keys(&self) -> &[i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.translation_keys_ptr, inner.max_tracks * 2) };
    }

    #[inline]
    fn translation_keys_mut(&mut self) -> &mut [i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.translation_keys_ptr, inner.max_tracks * 2) };
    }

    /// The keys in the animation that are valid for the current time ratio.
    #[inline]
    pub fn rotation_keys(&self) -> &[i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.rotation_keys_ptr, inner.max_tracks * 2) };
    }

    #[inline]
    fn rotation_keys_mut(&mut self) -> &mut [i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.rotation_keys_ptr, inner.max_tracks * 2) };
    }

    /// The keys in the animation that are valid for the current time ratio.
    #[inline]
    pub fn scale_keys(&self) -> &[i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.scale_keys_ptr, inner.max_tracks * 2) };
    }

    #[inline]
    fn scale_keys_mut(&mut self) -> &mut [i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.scale_keys_ptr, inner.max_tracks * 2) };
    }

    /// Outdated soa entries. One bit per soa entry (32 joints per byte).
    #[inline]
    pub fn outdated_translations(&self) -> &[u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.outdated_translations_ptr, inner.num_outdated) };
    }

    #[inline]
    fn outdated_translations_mut(&mut self) -> &mut [u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.outdated_translations_ptr, inner.num_outdated) };
    }

    /// Outdated soa entries. One bit per soa entry (32 joints per byte).
    #[inline]
    pub fn outdated_rotations(&self) -> &[u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.outdated_rotations_ptr, inner.num_outdated) };
    }

    #[inline]
    fn outdated_rotations_mut(&mut self) -> &mut [u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.outdated_rotations_ptr, inner.num_outdated) };
    }

    /// Outdated soa entries. One bit per soa entry (32 joints per byte).
    #[inline]
    pub fn outdated_scales(&self) -> &[u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.outdated_scales_ptr, inner.num_outdated) };
    }

    #[inline]
    fn outdated_scales_mut(&mut self) -> &mut [u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts_mut(inner.outdated_scales_ptr, inner.num_outdated) };
    }

    /// The unique identifier of the animation that the context is sampling.
    #[inline]
    pub fn animation_id(&self) -> u64 {
        return self.animation_id;
    }

    /// The current time ratio in the animation.
    #[inline]
    pub fn ratio(&self) -> f32 {
        return self.ratio;
    }

    /// Current cursors in the animation. 0 means that the context is invalid.
    #[inline]
    pub fn translation_cursor(&self) -> usize {
        return self.translation_cursor;
    }

    /// Current cursors in the animation. 0 means that the context is invalid.
    #[inline]
    pub fn rotation_cursor(&self) -> usize {
        return self.rotation_cursor;
    }

    /// Current cursors in the animation. 0 means that the context is invalid.
    #[inline]
    pub fn scale_cursor(&self) -> usize {
        return self.scale_cursor;
    }
}

#[cfg(feature = "rkyv")]
const _: () = {
    use bytecheck::CheckBytes;
    use rkyv::ser::Serializer;
    use rkyv::{from_archived, out_field, Archive, Deserialize, Fallible, RelPtr, Serialize};
    use std::io::{Error, ErrorKind};

    #[cfg(feature = "rkyv")]
    pub struct ArchivedSamplingContext {
        pub size: u32,
        pub max_tracks: u32,
        pub max_soa_tracks: u32,
        pub num_outdated: u32,

        pub animation_id: u64,
        pub ratio: f32,

        translations_ptr: RelPtr<InterpSoaFloat3>,
        rotations_ptr: RelPtr<InterpSoaQuaternion>,
        scales_ptr: RelPtr<InterpSoaFloat3>,

        translation_keys_ptr: RelPtr<i32>,
        rotation_keys_ptr: RelPtr<i32>,
        scale_keys_ptr: RelPtr<i32>,

        pub translation_cursor: u32,
        pub rotation_cursor: u32,
        pub scale_cursor: u32,

        outdated_translations_ptr: RelPtr<u8>,
        outdated_rotations_ptr: RelPtr<u8>,
        outdated_scales_ptr: RelPtr<u8>,
    }

    impl ArchivedSamplingContext {
        pub unsafe fn translations(&self) -> &[InterpSoaFloat3] {
            return slice::from_raw_parts(self.translations_ptr.as_ptr(), self.max_soa_tracks as usize);
        }

        pub unsafe fn rotations(&self) -> &[InterpSoaQuaternion] {
            return slice::from_raw_parts(self.rotations_ptr.as_ptr(), self.max_soa_tracks as usize);
        }

        pub unsafe fn scales(&self) -> &[InterpSoaFloat3] {
            return slice::from_raw_parts(self.scales_ptr.as_ptr(), self.max_soa_tracks as usize);
        }

        pub unsafe fn translation_keys(&self) -> &[i32] {
            return slice::from_raw_parts(self.translation_keys_ptr.as_ptr(), self.max_tracks as usize * 2);
        }

        pub unsafe fn rotation_keys(&self) -> &[i32] {
            return slice::from_raw_parts(self.rotation_keys_ptr.as_ptr(), self.max_tracks as usize * 2);
        }

        pub unsafe fn scale_keys(&self) -> &[i32] {
            return slice::from_raw_parts(self.scale_keys_ptr.as_ptr(), self.max_tracks as usize * 2);
        }

        pub unsafe fn outdated_translations(&self) -> &[u8] {
            return slice::from_raw_parts(self.outdated_translations_ptr.as_ptr(), self.num_outdated as usize);
        }

        pub unsafe fn outdated_rotations(&self) -> &[u8] {
            return slice::from_raw_parts(self.outdated_rotations_ptr.as_ptr(), self.num_outdated as usize);
        }

        pub unsafe fn outdated_scales(&self) -> &[u8] {
            return slice::from_raw_parts(self.outdated_scales_ptr.as_ptr(), self.num_outdated as usize);
        }
    }

    #[derive(Default)]
    pub struct SamplingContextResolver {
        translations_pos: usize,
        rotations_pos: usize,
        scales_pos: usize,

        translation_keys_pos: usize,
        rotation_keys_pos: usize,
        scale_keys_pos: usize,

        outdated_translations_pos: usize,
        outdated_rotations_pos: usize,
        outdated_scales_pos: usize,
    }

    impl Archive for SamplingContext {
        type Archived = ArchivedSamplingContext;
        type Resolver = SamplingContextResolver;

        unsafe fn resolve(&self, pos: usize, resolver: SamplingContextResolver, out: *mut ArchivedSamplingContext) {
            let (fp, fo) = out_field!(out.size);
            usize::resolve(&self.size(), pos + fp, (), fo);
            let (fp, fo) = out_field!(out.max_tracks);
            usize::resolve(&self.max_tracks(), pos + fp, (), fo);
            let (fp, fo) = out_field!(out.max_soa_tracks);
            usize::resolve(&self.max_soa_tracks(), pos + fp, (), fo);
            let (fp, fo) = out_field!(out.num_outdated);
            usize::resolve(&self.num_outdated(), pos + fp, (), fo);

            let (fp, fo) = out_field!(out.animation_id);
            u64::resolve(&self.animation_id, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.ratio);
            f32::resolve(&self.ratio, pos + fp, (), fo);

            let (fp, fo) = out_field!(out.translations_ptr);
            RelPtr::emplace(pos + fp, resolver.translations_pos, fo);
            let (fp, fo) = out_field!(out.rotations_ptr);
            RelPtr::emplace(pos + fp, resolver.rotations_pos, fo);
            let (fp, fo) = out_field!(out.scales_ptr);
            RelPtr::emplace(pos + fp, resolver.scales_pos, fo);

            let (fp, fo) = out_field!(out.translation_keys_ptr);
            RelPtr::emplace(pos + fp, resolver.translation_keys_pos, fo);
            let (fp, fo) = out_field!(out.rotation_keys_ptr);
            RelPtr::emplace(pos + fp, resolver.rotation_keys_pos, fo);
            let (fp, fo) = out_field!(out.scale_keys_ptr);
            RelPtr::emplace(pos + fp, resolver.scale_keys_pos, fo);

            let (fp, fo) = out_field!(out.translation_cursor);
            usize::resolve(&self.translation_cursor, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.rotation_cursor);
            usize::resolve(&self.rotation_cursor, pos + fp, (), fo);
            let (fp, fo) = out_field!(out.scale_cursor);
            usize::resolve(&self.scale_cursor, pos + fp, (), fo);

            let (fp, fo) = out_field!(out.outdated_translations_ptr);
            RelPtr::emplace(pos + fp, resolver.outdated_translations_pos, fo);
            let (fp, fo) = out_field!(out.outdated_rotations_ptr);
            RelPtr::emplace(pos + fp, resolver.outdated_rotations_pos, fo);
            let (fp, fo) = out_field!(out.outdated_scales_ptr);
            RelPtr::emplace(pos + fp, resolver.outdated_scales_pos, fo);
        }
    }

    impl<S: Serializer + Fallible + ?Sized> Serialize<S> for SamplingContext {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            let mut resolver = SamplingContextResolver::default();

            resolver.translations_pos = serializer.align_for::<InterpSoaFloat3>()?;
            serializer.write(unsafe {
                slice::from_raw_parts(
                    self.translations().as_ptr() as *const u8,
                    self.translations().len() * mem::size_of::<InterpSoaFloat3>(),
                )
            })?;
            resolver.rotations_pos = serializer.align_for::<InterpSoaQuaternion>()?;
            serializer.write(unsafe {
                slice::from_raw_parts(
                    self.rotations().as_ptr() as *const u8,
                    self.rotations().len() * mem::size_of::<InterpSoaQuaternion>(),
                )
            })?;
            resolver.scales_pos = serializer.align_for::<InterpSoaFloat3>()?;
            serializer.write(unsafe {
                slice::from_raw_parts(
                    self.scales().as_ptr() as *const u8,
                    self.scales().len() * mem::size_of::<InterpSoaFloat3>(),
                )
            })?;

            resolver.translation_keys_pos = serializer.align_for::<i32>()?;
            serializer.write(unsafe {
                slice::from_raw_parts(
                    self.translation_keys().as_ptr() as *const u8,
                    self.translation_keys().len() * mem::size_of::<i32>(),
                )
            })?;
            resolver.rotation_keys_pos = serializer.align_for::<i32>()?;
            serializer.write(unsafe {
                slice::from_raw_parts(
                    self.rotation_keys().as_ptr() as *const u8,
                    self.rotation_keys().len() * mem::size_of::<i32>(),
                )
            })?;
            resolver.scale_keys_pos = serializer.align_for::<i32>()?;
            serializer.write(unsafe {
                slice::from_raw_parts(
                    self.scale_keys().as_ptr() as *const u8,
                    self.scale_keys().len() * mem::size_of::<i32>(),
                )
            })?;

            resolver.outdated_translations_pos = serializer.align_for::<u8>()?;
            serializer.write(self.outdated_translations())?;
            resolver.outdated_rotations_pos = serializer.align_for::<u8>()?;
            serializer.write(self.outdated_rotations())?;
            resolver.outdated_scales_pos = serializer.align_for::<u8>()?;
            serializer.write(self.outdated_scales())?;

            return Ok(resolver);
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<SamplingContext, D> for ArchivedSamplingContext {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<SamplingContext, D::Error> {
            let archived = from_archived!(self);
            let mut context = SamplingContext::new(archived.max_tracks as usize);
            context.animation_id = archived.animation_id;
            context.ratio = archived.ratio;
            unsafe {
                context.translations_mut().copy_from_slice(archived.translations());
                context.rotations_mut().copy_from_slice(archived.rotations());
                context.scales_mut().copy_from_slice(archived.scales());
                context
                    .translation_keys_mut()
                    .copy_from_slice(archived.translation_keys());
                context.rotation_keys_mut().copy_from_slice(archived.rotation_keys());
                context.scale_keys_mut().copy_from_slice(archived.scale_keys());
                context.translation_cursor = archived.translation_cursor as usize;
                context.rotation_cursor = archived.rotation_cursor as usize;
                context.scale_cursor = archived.scale_cursor as usize;
                context
                    .outdated_translations_mut()
                    .copy_from_slice(archived.outdated_translations());
                context
                    .outdated_rotations_mut()
                    .copy_from_slice(archived.outdated_rotations());
                context
                    .outdated_scales_mut()
                    .copy_from_slice(archived.outdated_scales());
            }
            return Ok(context);
        }
    }

    impl<C: ?Sized> CheckBytes<C> for ArchivedSamplingContext {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<f32x4>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            return Ok(&*value);
        }
    }
};

#[cfg(feature = "serde")]
const _: () = {
    use serde::de::{self, MapAccess, Visitor};
    use serde::ser::SerializeMap;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for SamplingContext {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut map = serializer.serialize_map(None)?;
            map.serialize_entry("size", &self.size())?;
            map.serialize_entry("max_tracks", &self.max_tracks())?;
            map.serialize_entry("max_soa_tracks", &self.max_soa_tracks())?;
            map.serialize_entry("num_outdated", &self.num_outdated())?;
            map.serialize_entry("animation_id", &self.animation_id())?;
            map.serialize_entry("ratio", &self.ratio())?;
            map.serialize_entry("translations", self.translations())?;
            map.serialize_entry("rotations", self.rotations())?;
            map.serialize_entry("scales", self.scales())?;
            map.serialize_entry("translations_keys", self.translation_keys())?;
            map.serialize_entry("rotations_keys", self.rotation_keys())?;
            map.serialize_entry("scales_keys", self.scale_keys())?;
            map.serialize_entry("translation_cursor", &self.translation_cursor())?;
            map.serialize_entry("rotation_cursor", &self.rotation_cursor())?;
            map.serialize_entry("scale_cursor", &self.scale_cursor())?;
            map.serialize_entry("outdated_translations", self.outdated_translations())?;
            map.serialize_entry("outdated_rotations", self.outdated_rotations())?;
            map.serialize_entry("outdated_scales", self.outdated_scales())?;
            return map.end();
        }
    }

    impl<'de> Deserialize<'de> for SamplingContext {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<SamplingContext, D::Error> {
            return deserializer.deserialize_map(SamplingContextVisitor);
        }
    }

    struct SamplingContextVisitor;

    impl<'de> Visitor<'de> for SamplingContextVisitor {
        type Value = SamplingContext;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            return formatter.write_str("struct SamplingContext");
        }

        fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<Self::Value, V::Error> {
            let mut max_tracks: Option<usize> = None;
            while let Some(key) = map.next_key::<&str>()? {
                if key == "max_tracks" {
                    max_tracks = Some(map.next_value()?);
                    break;
                } else {
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
            let mut ctx: SamplingContext = match max_tracks {
                Some(max_tracks) => SamplingContext::new(max_tracks),
                None => return Err(de::Error::custom("Miss field max_tracks")),
            };

            while let Some(key) = map.next_key()? {
                match key {
                    "animation_id" => ctx.animation_id = map.next_value()?,
                    "ratio" => ctx.ratio = map.next_value()?,
                    "translations" => {
                        let translations: Vec<InterpSoaFloat3> = map.next_value()?;
                        ctx.translations_mut().copy_from_slice(&translations);
                    }
                    "rotations" => {
                        let rotations: Vec<InterpSoaQuaternion> = map.next_value()?;
                        ctx.rotations_mut().copy_from_slice(&rotations);
                    }
                    "scales" => {
                        let scales: Vec<InterpSoaFloat3> = map.next_value()?;
                        ctx.scales_mut().copy_from_slice(&scales);
                    }
                    "translations_keys" => {
                        let translations_keys: Vec<i32> = map.next_value()?;
                        ctx.translation_keys_mut().copy_from_slice(&translations_keys);
                    }
                    "rotations_keys" => {
                        let rotations_keys: Vec<i32> = map.next_value()?;
                        ctx.rotation_keys_mut().copy_from_slice(&rotations_keys);
                    }
                    "scales_keys" => {
                        let scales_keys: Vec<i32> = map.next_value()?;
                        ctx.scale_keys_mut().copy_from_slice(&scales_keys);
                    }
                    "translation_cursor" => ctx.translation_cursor = map.next_value()?,
                    "rotation_cursor" => ctx.rotation_cursor = map.next_value()?,
                    "scale_cursor" => ctx.scale_cursor = map.next_value()?,
                    "outdated_translations" => {
                        let outdated_translations: Vec<u8> = map.next_value()?;
                        ctx.outdated_translations_mut().copy_from_slice(&outdated_translations);
                    }
                    "outdated_rotations" => {
                        let outdated_rotations: Vec<u8> = map.next_value()?;
                        ctx.outdated_rotations_mut().copy_from_slice(&outdated_rotations);
                    }
                    "outdated_scales" => {
                        let outdated_scales: Vec<u8> = map.next_value()?;
                        ctx.outdated_scales_mut().copy_from_slice(&outdated_scales);
                    }
                    _ => {
                        map.next_value::<serde::de::IgnoredAny>()?;
                    }
                }
            }
            return Ok(ctx);
        }
    }
};

pub trait AsSamplingContext {
    fn as_ref(&self) -> &SamplingContext;
    fn as_mut(&mut self) -> &mut SamplingContext;
}

impl AsSamplingContext for SamplingContext {
    #[inline(always)]
    fn as_ref(&self) -> &SamplingContext {
        return self;
    }

    #[inline(always)]
    fn as_mut(&mut self) -> &mut SamplingContext {
        return self;
    }
}

impl AsSamplingContext for &'_ mut SamplingContext {
    #[inline(always)]
    fn as_ref(&self) -> &SamplingContext {
        return self;
    }

    #[inline(always)]
    fn as_mut(&mut self) -> &mut SamplingContext {
        return *self;
    }
}

///
/// Samples an animation at a given time ratio in the unit interval 0.0-1.0 (where 0.0 is the beginning of
/// the animation, 1.0 is the end), to output the corresponding posture in local-space.
///
/// `SamplingJob` uses `SamplingContext` to store intermediate values (decompressed animation keyframes...)
/// while sampling.
/// This context also stores pre-computed values that allows drastic optimization while playing/sampling the
/// animation forward.
/// Backward sampling works, but isn't optimized through the context. The job does not owned the buffers
/// (in/output) and will thus not delete them during job's destruction.
///
#[derive(Debug)]
pub struct SamplingJob<A = Rc<Animation>, O = Rc<RefCell<Vec<SoaTransform>>>, C = SamplingContext>
where
    A: OzzObj<Animation>,
    O: OzzMutBuf<SoaTransform>,
    C: AsSamplingContext,
{
    animation: Option<A>,
    context: Option<C>,
    ratio: f32,
    output: Option<O>,
}

pub type SamplingJobRef<'t> = SamplingJob<&'t Animation, &'t mut [SoaTransform], &'t mut SamplingContext>;
pub type SamplingJobRc = SamplingJob<Rc<Animation>, Rc<RefCell<Vec<SoaTransform>>>, SamplingContext>;
pub type SamplingJobArc = SamplingJob<Arc<Animation>, Arc<RwLock<Vec<SoaTransform>>>, SamplingContext>;

impl<A, O, C> Default for SamplingJob<A, O, C>
where
    A: OzzObj<Animation>,
    O: OzzMutBuf<SoaTransform>,
    C: AsSamplingContext,
{
    fn default() -> SamplingJob<A, O, C> {
        return SamplingJob {
            animation: None,
            context: None,
            ratio: 0.0,
            output: None,
        };
    }
}

impl<A, O, C> SamplingJob<A, O, C>
where
    A: OzzObj<Animation>,
    O: OzzMutBuf<SoaTransform>,
    C: AsSamplingContext,
{
    /// Gets animation to sample of `SamplingJob`.
    #[inline]
    pub fn animation(&self) -> Option<&A> {
        return self.animation.as_ref();
    }

    /// Sets animation to sample of `SamplingJob`.
    #[inline]
    pub fn set_animation(&mut self, animation: A) {
        self.animation = Some(animation);
    }

    /// Clears animation to sample of `SamplingJob`.
    #[inline]
    pub fn clear_animation(&mut self) {
        self.animation = None;
    }

    /// Gets context of `SamplingJob`. See [SamplingContext].
    #[inline]
    pub fn context(&self) -> Option<&C> {
        return self.context.as_ref();
    }

    /// Sets context of `SamplingJob`. See [SamplingContext].
    #[inline]
    pub fn set_context(&mut self, ctx: C) {
        self.context = Some(ctx);
    }

    /// Clears context of `SamplingJob`. See [SamplingContext].
    #[inline]
    pub fn clear_context(&mut self) {
        self.context = None;
    }

    /// Takes context of `SamplingJob`. See [SamplingContext].
    #[inline]
    pub fn take_context(&mut self) -> Option<C> {
        return self.context.take();
    }

    /// Gets the time ratio of `SamplingJob`.
    #[inline]
    pub fn ratio(&self) -> f32 {
        return self.ratio;
    }

    /// Sets the time ratio of `SamplingJob`.
    ///
    /// Time ratio in the unit interval 0.0-1.0 used to sample animation (where 0 is the beginning of
    /// the animation, 1 is the end). It should be computed as the current time in the animation,
    /// divided by animation duration.
    ///
    /// This ratio is clamped before job execution in order to resolves any approximation issue on range
    /// bounds.
    #[inline]
    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = f32_clamp_or_max(ratio, 0.0f32, 1.0f32);
    }

    /// Gets output of `SamplingJob`.
    #[inline]
    pub fn output(&self) -> Option<&O> {
        return self.output.as_ref();
    }

    /// Sets output of `SamplingJob`.
    ///
    /// The output range to be filled with sampled joints during job execution.
    ///
    /// If there are less joints in the animation compared to the output range, then remaining
    /// `SoaTransform` are left unchanged.
    /// If there are more joints in the animation, then the last joints are not sampled.
    #[inline]
    pub fn set_output(&mut self, output: O) {
        self.output = Some(output);
    }

    /// Clears output of `SamplingJob`.
    #[inline]
    pub fn clear_output(&mut self) {
        self.output = None;
    }

    /// Validates `SamplingJob` parameters.
    pub fn validate(&self) -> bool {
        return (|| {
            let animation = self.animation.as_ref()?.obj();
            let context = self.context.as_ref()?;
            let output = self.output.as_ref()?.buf().ok()?;

            let mut ok = context.as_ref().max_soa_tracks() >= animation.num_soa_tracks();
            ok &= output.len() >= animation.num_soa_tracks();
            return Some(ok);
        })()
        .unwrap_or(false);
    }

    /// Runs job's sampling task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        let animation = self.animation.as_ref().ok_or(OzzError::InvalidJob)?.obj();
        let ctx = self.context.as_mut().ok_or(OzzError::InvalidJob)?;
        let mut output = self.output.as_mut().ok_or(OzzError::InvalidJob)?.mut_buf()?;

        let mut ok = ctx.as_ref().max_soa_tracks() >= animation.num_soa_tracks();
        ok &= output.len() >= animation.num_soa_tracks();
        if !ok {
            return Err(OzzError::InvalidJob);
        }

        if animation.num_soa_tracks() == 0 {
            return Ok(());
        }

        Self::step_context(animation, ctx.as_mut(), self.ratio);

        Self::update_translation_cursor(animation, ctx.as_mut(), self.ratio);
        Self::update_translation_key_frames(animation, ctx.as_mut());

        Self::update_rotation_cursor(animation, ctx.as_mut(), self.ratio);
        Self::update_rotation_key_frames(animation, ctx.as_mut());

        Self::update_scale_cursor(animation, ctx.as_mut(), self.ratio);
        Self::update_scale_key_frames(animation, ctx.as_mut());

        Self::interpolates(animation, ctx.as_mut(), self.ratio, &mut output)?;

        return Ok(());
    }

    fn step_context(animation: &Animation, ctx: &mut SamplingContext, ratio: f32) {
        let animation_id = animation as *const _ as u64;
        if (ctx.animation_id != animation_id) || ratio < ctx.ratio {
            ctx.animation_id = animation_id;
            ctx.translation_cursor = 0;
            ctx.rotation_cursor = 0;
            ctx.scale_cursor = 0;
        }
        ctx.ratio = ratio;
    }

    fn update_translation_cursor(animation: &Animation, ctx: &mut SamplingContext, ratio: f32) {
        if ctx.translation_cursor == 0 {
            for i in 0..animation.num_soa_tracks() {
                let in_idx0 = i * 4;
                let in_idx1 = in_idx0 + animation.num_aligned_tracks();
                let out_idx = i * 4 * 2;
                ctx.translation_keys_mut()[out_idx + 0] = (in_idx0 + 0) as i32;
                ctx.translation_keys_mut()[out_idx + 1] = (in_idx1 + 0) as i32;
                ctx.translation_keys_mut()[out_idx + 2] = (in_idx0 + 1) as i32;
                ctx.translation_keys_mut()[out_idx + 3] = (in_idx1 + 1) as i32;
                ctx.translation_keys_mut()[out_idx + 4] = (in_idx0 + 2) as i32;
                ctx.translation_keys_mut()[out_idx + 5] = (in_idx1 + 2) as i32;
                ctx.translation_keys_mut()[out_idx + 6] = (in_idx0 + 3) as i32;
                ctx.translation_keys_mut()[out_idx + 7] = (in_idx1 + 3) as i32;
            }

            ctx.outdated_translations_mut().iter_mut().for_each(|x| *x = 0xFF);
            let last_offset = ((animation.num_soa_tracks() + 7) / 8 * 8) - animation.num_soa_tracks();
            ctx.outdated_translations_mut()
                .last_mut()
                .map(|x| *x = 0xFF >> last_offset);

            ctx.translation_cursor = animation.num_aligned_tracks() * 2;
        }

        while ctx.translation_cursor < animation.translations().len() {
            let track = animation.translations()[ctx.translation_cursor].track as usize;
            let key_idx = ctx.translation_keys()[track * 2 + 1] as usize;
            let ani_ratio = animation.translations()[key_idx].ratio;
            if ani_ratio > ratio {
                break;
            }

            ctx.outdated_translations_mut()[track / 32] |= 1 << ((track & 0x1F) / 4);
            let base = (animation.translations()[ctx.translation_cursor].track as usize) * 2;
            ctx.translation_keys_mut()[base] = ctx.translation_keys()[base + 1];
            ctx.translation_keys_mut()[base + 1] = ctx.translation_cursor as i32;
            ctx.translation_cursor = ctx.translation_cursor + 1;
        }
    }

    fn update_translation_key_frames(animation: &Animation, ctx: &mut SamplingContext) {
        let num_outdated_flags = (animation.num_soa_tracks() + 7) / 8;
        for j in 0..num_outdated_flags {
            let mut outdated = ctx.outdated_translations()[j];
            for i in (8 * j)..(8 * j + 8) {
                if outdated & 1 == 0 {
                    continue;
                }
                let base = i * 4 * 2;

                let k00 = animation.translations()[ctx.translation_keys()[base + 0] as usize];
                let k10 = animation.translations()[ctx.translation_keys()[base + 2] as usize];
                let k20 = animation.translations()[ctx.translation_keys()[base + 4] as usize];
                let k30 = animation.translations()[ctx.translation_keys()[base + 6] as usize];
                ctx.translations_mut()[i].ratio[0] = f32x4::from_array([k00.ratio, k10.ratio, k20.ratio, k30.ratio]);
                Float3Key::simd_decompress(&k00, &k10, &k20, &k30, &mut ctx.translations_mut()[i].value[0]);

                let k01 = animation.translations()[ctx.translation_keys()[base + 1] as usize];
                let k11 = animation.translations()[ctx.translation_keys()[base + 3] as usize];
                let k21 = animation.translations()[ctx.translation_keys()[base + 5] as usize];
                let k31 = animation.translations()[ctx.translation_keys()[base + 7] as usize];
                ctx.translations_mut()[i].ratio[1] = f32x4::from_array([k01.ratio, k11.ratio, k21.ratio, k31.ratio]);
                Float3Key::simd_decompress(&k01, &k11, &k21, &k31, &mut ctx.translations_mut()[i].value[1]);

                outdated >>= 1;
            }
        }
    }

    fn update_rotation_cursor(animation: &Animation, ctx: &mut SamplingContext, ratio: f32) {
        if ctx.rotation_cursor == 0 {
            for i in 0..animation.num_soa_tracks() {
                let in_idx0 = i * 4;
                let in_idx1 = in_idx0 + animation.num_aligned_tracks();
                let out_idx = i * 4 * 2;
                ctx.rotation_keys_mut()[out_idx + 0] = (in_idx0 + 0) as i32;
                ctx.rotation_keys_mut()[out_idx + 1] = (in_idx1 + 0) as i32;
                ctx.rotation_keys_mut()[out_idx + 2] = (in_idx0 + 1) as i32;
                ctx.rotation_keys_mut()[out_idx + 3] = (in_idx1 + 1) as i32;
                ctx.rotation_keys_mut()[out_idx + 4] = (in_idx0 + 2) as i32;
                ctx.rotation_keys_mut()[out_idx + 5] = (in_idx1 + 2) as i32;
                ctx.rotation_keys_mut()[out_idx + 6] = (in_idx0 + 3) as i32;
                ctx.rotation_keys_mut()[out_idx + 7] = (in_idx1 + 3) as i32;
            }

            ctx.outdated_rotations_mut().iter_mut().for_each(|x| *x = 0xFF);
            let last_offset = ((animation.num_soa_tracks() + 7) / 8 * 8) - animation.num_soa_tracks();
            ctx.outdated_rotations_mut()
                .last_mut()
                .map(|x| *x = 0xFF >> last_offset);

            ctx.rotation_cursor = animation.num_aligned_tracks() * 2;
        }

        while ctx.rotation_cursor < animation.rotations().len() {
            let track = animation.rotations()[ctx.rotation_cursor].track() as usize;
            let key_idx = ctx.rotation_keys()[track * 2 + 1] as usize;
            let ani_ratio = animation.rotations()[key_idx].ratio;
            if ani_ratio > ratio {
                break;
            }

            ctx.outdated_rotations_mut()[track / 32] |= 1 << ((track & 0x1F) / 4);
            let base = (animation.rotations()[ctx.rotation_cursor].track() as usize) * 2;
            ctx.rotation_keys_mut()[base] = ctx.rotation_keys()[base + 1];
            ctx.rotation_keys_mut()[base + 1] = ctx.rotation_cursor as i32;
            ctx.rotation_cursor = ctx.rotation_cursor + 1;
        }
    }

    fn update_rotation_key_frames(animation: &Animation, ctx: &mut SamplingContext) {
        let num_outdated_flags = (animation.num_soa_tracks() + 7) / 8;
        for j in 0..num_outdated_flags {
            let mut outdated = ctx.outdated_rotations()[j];
            for i in (8 * j)..(8 * j + 8) {
                if outdated & 1 == 0 {
                    continue;
                }
                let base = i * 4 * 2;

                let k00 = animation.rotations()[ctx.rotation_keys()[base + 0] as usize];
                let k10 = animation.rotations()[ctx.rotation_keys()[base + 2] as usize];
                let k20 = animation.rotations()[ctx.rotation_keys()[base + 4] as usize];
                let k30 = animation.rotations()[ctx.rotation_keys()[base + 6] as usize];
                ctx.rotations_mut()[i].ratio[0] = f32x4::from_array([k00.ratio, k10.ratio, k20.ratio, k30.ratio]);
                QuaternionKey::simd_decompress(&k00, &k10, &k20, &k30, &mut ctx.rotations_mut()[i].value[0]);

                let k01 = animation.rotations()[ctx.rotation_keys()[base + 1] as usize];
                let k11 = animation.rotations()[ctx.rotation_keys()[base + 3] as usize];
                let k21 = animation.rotations()[ctx.rotation_keys()[base + 5] as usize];
                let k31 = animation.rotations()[ctx.rotation_keys()[base + 7] as usize];
                ctx.rotations_mut()[i].ratio[1] = f32x4::from_array([k01.ratio, k11.ratio, k21.ratio, k31.ratio]);
                QuaternionKey::simd_decompress(&k01, &k11, &k21, &k31, &mut ctx.rotations_mut()[i].value[1]);

                outdated >>= 1;
            }
        }
    }

    fn update_scale_cursor(animation: &Animation, ctx: &mut SamplingContext, ratio: f32) {
        if ctx.scale_cursor == 0 {
            for i in 0..animation.num_soa_tracks() {
                let in_idx0 = i * 4;
                let in_idx1 = in_idx0 + animation.num_aligned_tracks();
                let out_idx = i * 4 * 2;
                ctx.scale_keys_mut()[out_idx + 0] = (in_idx0 + 0) as i32;
                ctx.scale_keys_mut()[out_idx + 1] = (in_idx1 + 0) as i32;
                ctx.scale_keys_mut()[out_idx + 2] = (in_idx0 + 1) as i32;
                ctx.scale_keys_mut()[out_idx + 3] = (in_idx1 + 1) as i32;
                ctx.scale_keys_mut()[out_idx + 4] = (in_idx0 + 2) as i32;
                ctx.scale_keys_mut()[out_idx + 5] = (in_idx1 + 2) as i32;
                ctx.scale_keys_mut()[out_idx + 6] = (in_idx0 + 3) as i32;
                ctx.scale_keys_mut()[out_idx + 7] = (in_idx1 + 3) as i32;
            }

            ctx.outdated_scales_mut().iter_mut().for_each(|x| *x = 0xFF);
            let last_offset = ((animation.num_soa_tracks() + 7) / 8 * 8) - animation.num_soa_tracks();
            ctx.outdated_scales_mut().last_mut().map(|x| *x = 0xFF >> last_offset);

            ctx.scale_cursor = animation.num_aligned_tracks() * 2;
        }

        while ctx.scale_cursor < animation.scales().len() {
            let track = animation.scales()[ctx.scale_cursor].track as usize;
            let key_idx = ctx.scale_keys()[track * 2 + 1] as usize;
            let ani_ratio = animation.scales()[key_idx].ratio;
            if ani_ratio > ratio {
                break;
            }

            ctx.outdated_scales_mut()[track / 32] |= 1 << ((track & 0x1F) / 4);
            let base = (animation.scales()[ctx.scale_cursor].track as usize) * 2;
            ctx.scale_keys_mut()[base] = ctx.scale_keys()[base + 1];
            ctx.scale_keys_mut()[base + 1] = ctx.scale_cursor as i32;
            ctx.scale_cursor = ctx.scale_cursor + 1;
        }
    }

    fn update_scale_key_frames(animation: &Animation, ctx: &mut SamplingContext) {
        let num_outdated_flags = (animation.num_soa_tracks() + 7) / 8;
        for j in 0..num_outdated_flags {
            let mut outdated = ctx.outdated_scales()[j];
            for i in (8 * j)..(8 * j + 8) {
                if outdated & 1 == 0 {
                    continue;
                }
                let base = i * 4 * 2;

                let k00 = animation.scales()[ctx.scale_keys()[base + 0] as usize];
                let k10 = animation.scales()[ctx.scale_keys()[base + 2] as usize];
                let k20 = animation.scales()[ctx.scale_keys()[base + 4] as usize];
                let k30 = animation.scales()[ctx.scale_keys()[base + 6] as usize];
                ctx.scales_mut()[i].ratio[0] = f32x4::from_array([k00.ratio, k10.ratio, k20.ratio, k30.ratio]);
                Float3Key::simd_decompress(&k00, &k10, &k20, &k30, &mut ctx.scales_mut()[i].value[0]);

                let k01 = animation.scales()[ctx.scale_keys()[base + 1] as usize];
                let k11 = animation.scales()[ctx.scale_keys()[base + 3] as usize];
                let k21 = animation.scales()[ctx.scale_keys()[base + 5] as usize];
                let k31 = animation.scales()[ctx.scale_keys()[base + 7] as usize];
                ctx.scales_mut()[i].ratio[1] = f32x4::from_array([k01.ratio, k11.ratio, k21.ratio, k31.ratio]);
                Float3Key::simd_decompress(&k01, &k11, &k21, &k31, &mut ctx.scales_mut()[i].value[1]);

                outdated >>= 1;
            }
        }
    }

    fn interpolates(
        animation: &Animation,
        ctx: &mut SamplingContext,
        ratio: f32,
        output: &mut [SoaTransform],
    ) -> Result<(), OzzError> {
        let ratio4 = f32x4::splat(ratio);
        for idx in 0..animation.num_soa_tracks() {
            let translation = &ctx.translations()[idx];
            let translation_ratio = (ratio4 - translation.ratio[0]) / (translation.ratio[1] - translation.ratio[0]);
            output[idx].translation = SoaVec3::lerp(&translation.value[0], &translation.value[1], translation_ratio);

            let rotation = &ctx.rotations()[idx];
            let rotation_ratio = (ratio4 - rotation.ratio[0]) / (rotation.ratio[1] - rotation.ratio[0]);
            output[idx].rotation = SoaQuat::nlerp(&rotation.value[0], &rotation.value[1], rotation_ratio);

            let scale = &ctx.scales()[idx];
            let scale_ratio = (ratio4 - scale.ratio[0]) / (scale.ratio[1] - scale.ratio[0]);
            output[idx].scale = SoaVec3::lerp(&scale.value[0], &scale.value[1], scale_ratio);
        }

        return Ok(());
    }
}

#[cfg(test)]
mod sampling_tests {
    use glam::{Quat, Vec3};
    use wasm_bindgen_test::*;

    use super::*;
    use crate::base::OzzBuf;

    fn make_buf<T>(v: Vec<T>) -> Rc<RefCell<Vec<T>>> {
        return Rc::new(RefCell::new(v));
    }

    // f16 -> f32
    // ignore overflow, infinite, NaN
    pub fn f16(f: f32) -> u16 {
        let n = unsafe { mem::transmute::<f32, u32>(f) };
        if (n & 0x7FFFFFFF) == 0 {
            return (n >> 16) as u16;
        }
        let sign = (n >> 16) & 0x8000;
        let expo = (((n & 0x7f800000) - 0x38000000) >> 13) & 0x7c00;
        let base = (n >> 13) & 0x03ff;
        return (sign | expo | base) as u16;
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_validity() {
        let animation = Rc::new(Animation::from_path("./resource/animation-blending-1.ozz").unwrap());
        let aligned_tracks = animation.num_aligned_tracks();

        // invalid output
        let mut job: SamplingJob = SamplingJob::default();
        job.set_animation(animation.clone());
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid animation
        let mut job: SamplingJob = SamplingJob::default();
        job.set_output(make_buf(vec![SoaTransform::default(); animation.num_soa_tracks() + 10]));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // invalid cache size
        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(5));
        job.set_output(make_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(aligned_tracks));
        job.set_output(make_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        assert!(job.validate());
        assert!(job.run().is_ok());
    }

    fn new_translations() -> Vec<Float3Key> {
        return vec![
            Float3Key::new(0.0, 0, [f16(0.0); 3]),
            Float3Key::new(0.0, 1, [f16(0.0); 3]),
            Float3Key::new(0.0, 2, [f16(0.0); 3]),
            Float3Key::new(0.0, 3, [f16(0.0); 3]),
            Float3Key::new(1.0, 0, [f16(0.0); 3]),
            Float3Key::new(1.0, 1, [f16(0.0); 3]),
            Float3Key::new(1.0, 2, [f16(0.0); 3]),
            Float3Key::new(1.0, 3, [f16(0.0); 3]),
        ];
    }

    fn new_rotations() -> Vec<QuaternionKey> {
        return vec![
            QuaternionKey::new(0.0, (0 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(0.0, (1 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(0.0, (2 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(0.0, (3 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (0 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (1 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (2 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (3 << 3) + (3 << 1), [0, 0, 0]),
        ];
    }

    fn new_scales() -> Vec<Float3Key> {
        return vec![
            Float3Key::new(0.0, 0, [f16(1.0); 3]),
            Float3Key::new(0.0, 1, [f16(1.0); 3]),
            Float3Key::new(0.0, 2, [f16(1.0); 3]),
            Float3Key::new(0.0, 3, [f16(1.0); 3]),
            Float3Key::new(1.0, 0, [f16(1.0); 3]),
            Float3Key::new(1.0, 1, [f16(1.0); 3]),
            Float3Key::new(1.0, 2, [f16(1.0); 3]),
            Float3Key::new(1.0, 3, [f16(1.0); 3]),
        ];
    }

    const V0: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    const V1: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    const QU: Quat = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0);
    const TX: SoaTransform = SoaTransform {
        translation: SoaVec3::splat_col([1234.5678; 3]),
        rotation: SoaQuat::splat_col([1234.5678; 4]),
        scale: SoaVec3::splat_col([1234.5678; 3]),
    };

    struct Frame<const S: usize> {
        ratio: f32,
        transform: [(Vec3, Quat, Vec3); S],
    }

    fn execute_test<const T: usize>(
        duration: f32,
        translations: Vec<Float3Key>,
        rotations: Vec<QuaternionKey>,
        scales: Vec<Float3Key>,
        frames: Vec<Frame<T>>,
    ) {
        let animation = Rc::new(Animation::from_raw(
            duration,
            T,
            String::new(),
            translations,
            rotations,
            scales,
        ));
        let mut job = SamplingJob::default();
        job.set_animation(animation);
        job.set_context(SamplingContext::new(T));
        let output = make_buf(vec![TX; T + 1]);

        for frame in &frames {
            job.set_output(output.clone());
            job.set_ratio(frame.ratio);
            job.run().unwrap();

            if T == 0 {
                assert_eq!(output.borrow()[0], TX);
            }

            for idx in 0..T {
                let out = output.borrow()[idx / 4];
                assert_eq!(
                    out.translation.col(idx % 4),
                    frame.transform[idx].0,
                    "ratio={} translation idx={}",
                    frame.ratio,
                    idx
                );
                assert_eq!(
                    out.rotation.col(idx % 4),
                    frame.transform[idx].1,
                    "ratio={} rotation idx={}",
                    frame.ratio,
                    idx
                );
                assert_eq!(
                    out.scale.col(idx % 4),
                    frame.transform[idx].2,
                    "ratio={} scale idx={}",
                    frame.ratio,
                    idx
                );
            }

            assert_eq!(job.context().unwrap().ratio, f32_clamp_or_max(frame.ratio, 0.0, 1.0));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling() {
        fn frame(ratio: f32, t1: f32, t2: f32, t3: f32, t4: f32) -> Frame<4> {
            return Frame {
                ratio,
                transform: [
                    (Vec3::new(t1, 0.0, 0.0), QU, V1),
                    (Vec3::new(t2, 0.0, 0.0), QU, V1),
                    (Vec3::new(t3, 0.0, 0.0), QU, V1),
                    (Vec3::new(t4, 0.0, 0.0), QU, V1),
                ],
            };
        }

        execute_test::<4>(
            1.0,
            vec![
                Float3Key::new(0.0, 0, [f16(-1.0), 0, 0]),
                Float3Key::new(0.0, 1, [f16(0.0), 0, 0]),
                Float3Key::new(0.0, 2, [f16(2.0), 0, 0]),
                Float3Key::new(0.0, 3, [f16(7.0), 0, 0]),
                Float3Key::new(1.0, 0, [f16(-1.0), 0, 0]),
                Float3Key::new(1.0, 1, [f16(0.0), 0, 0]),
                Float3Key::new(0.200000003, 2, [f16(6.0), 0, 0]),
                Float3Key::new(0.200000003, 3, [f16(7.0), 0, 0]),
                Float3Key::new(0.400000006, 2, [f16(8.0), 0, 0]),
                Float3Key::new(0.600000024, 3, [f16(9.0), 0, 0]),
                Float3Key::new(0.600000024, 2, [f16(10.0), 0, 0]),
                Float3Key::new(1.0, 2, [f16(11.0), 0, 0]),
                Float3Key::new(1.0, 3, [f16(9.0), 0, 0]),
            ],
            new_rotations(),
            new_scales(),
            vec![
                frame(-0.2, -1.0, 0.0, 2.0, 7.0),
                frame(0.0, -1.0, 0.0, 2.0, 7.0),
                frame(0.0000001, -1.0, 0.0, 2.000002, 7.0),
                frame(0.1, -1.0, 0.0, 4.0, 7.0),
                frame(0.2, -1.0, 0.0, 6.0, 7.0),
                frame(0.3, -1.0, 0.0, 7.0, 7.5),
                frame(0.4, -1.0, 0.0, 8.0, 8.0),
                frame(0.3999999, -1.0, 0.0, 7.999999, 7.9999995),
                frame(0.4000001, -1.0, 0.0, 8.000001, 8.0),
                frame(0.5, -1.0, 0.0, 9.0, 8.5),
                frame(0.6, -1.0, 0.0, 10.0, 9.0),
                frame(0.9999999, -1.0, 0.0, 11.0, 9.0),
                frame(1.0, -1.0, 0.0, 11.0, 9.0),
                frame(1.000001, -1.0, 0.0, 11.0, 9.0),
                frame(0.5, -1.0, 0.0, 9.0, 8.5),
                // frame(0.9999999, -1.0, 0.0, 11.0, 9.0),
                // frame(0.0000001, -1.0, 0.0, 2.000002, 7.0),
            ],
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling_no_track() {
        execute_test::<0>(46.0, vec![], vec![], vec![], vec![]);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling_1_track_0_key() {
        execute_test::<1>(
            46.0,
            new_translations(),
            new_rotations(),
            new_scales(),
            (-2..12)
                .map(|x| Frame {
                    ratio: x as f32 / 10.0,
                    transform: [(V0, QU, V1)],
                })
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling_1_track_1_key() {
        let mut translations = new_translations();
        translations[0] = Float3Key::new(0.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);
        translations[4] = Float3Key::new(1.0 / 46.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);

        execute_test::<1>(
            46.0,
            translations,
            new_rotations(),
            new_scales(),
            (-2..12)
                .map(|x| Frame {
                    ratio: x as f32 / 10.0,
                    transform: [(Vec3::new(1.0, -1.0, 5.0), QU, V1)],
                })
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling_1_track_2_key() {
        fn frame(ratio: f32, v: Vec3) -> Frame<2> {
            return Frame {
                ratio,
                transform: [(v, QU, V1), (V0, QU, V1)],
            };
        }

        execute_test::<2>(
            46.0,
            vec![
                Float3Key::new(0.0, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(0.0, 1, [f16(0.0); 3]),
                Float3Key::new(0.0, 2, [f16(0.0); 3]),
                Float3Key::new(0.0, 3, [f16(0.0); 3]),
                Float3Key::new(0.5 / 46.0, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(1.0, 1, [f16(0.0); 3]),
                Float3Key::new(1.0, 2, [f16(0.0); 3]),
                Float3Key::new(1.0, 3, [f16(0.0); 3]),
                Float3Key::new(1.0 / 46.0, 0, [f16(2.0), f16(4.0), f16(8.0)]),
                Float3Key::new(1.0, 0, [f16(2.0), f16(4.0), f16(8.0)]),
            ],
            new_rotations(),
            new_scales(),
            vec![
                frame(0.0, Vec3::new(1.0, 2.0, 4.0)),
                frame(0.5 / 46.0, Vec3::new(1.0, 2.0, 4.0)),
                frame(1.0 / 46.0, Vec3::new(2.0, 4.0, 8.0)),
                frame(1.0, Vec3::new(2.0, 4.0, 8.0)),
                frame(0.75 / 46.0, Vec3::new(1.5, 3.0, 6.0)),
            ],
        );
    }

    #[test]
    #[wasm_bindgen_test]
    #[rustfmt::skip]
    fn test_sampling_4_track_2_key() {
        execute_test::<4>(
            1.0,
            vec![
                Float3Key::new(0.0, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(0.0, 1, [f16(0.0); 3]),
                Float3Key::new(0.0, 2, [f16(0.0); 3]),
                Float3Key::new(0.0, 3, [f16(-1.0), f16(-2.0), f16(-4.0)]),
                Float3Key::new(0.5, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(1.0, 1, [f16(0.0); 3]),
                Float3Key::new(1.0, 2, [f16(0.0); 3]),
                Float3Key::new(1.0, 3, [f16(-2.0), f16(-4.0), f16(-8.0)]),
                Float3Key::new(0.8, 0, [f16(2.0), f16(4.0), f16(8.0)]),
                Float3Key::new(1.0, 0, [f16(2.0), f16(4.0), f16(8.0)]),
            ],
            vec![
                QuaternionKey::new(0.0, (0 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(0.0, (1 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(0.0, (2 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(0.0, (3 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (0 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (1 << 3) + (1 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (2 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (3 << 3) + (3 << 1), [0, 0, 0]),
            ],
            vec![
                Float3Key::new(0.0, 0, [f16(1.0); 3]),
                Float3Key::new(0.0, 1, [f16(1.0); 3]),
                Float3Key::new(0.0, 2, [f16(0.0); 3]),
                Float3Key::new(0.0, 3, [f16(1.0); 3]),
                Float3Key::new(1.0, 0, [f16(1.0); 3]),
                Float3Key::new(1.0, 1, [f16(1.0); 3]),
                Float3Key::new(0.5, 2, [f16(0.0); 3]),
                Float3Key::new(1.0, 3, [f16(1.0); 3]),
                Float3Key::new(0.8, 2, [f16(-1.0); 3]),
                Float3Key::new(1.0, 2, [f16(-1.0); 3]),
            ],
            vec![
                Frame {ratio: 0.0, transform: [
                    (Vec3::new(1.0, 2.0, 4.0), QU, V1),
                    (V0, QU, V1),
                    (V0, QU, V0),
                    (Vec3::new(-1.0, -2.0, -4.0), QU, V1),
                ]},
                Frame {ratio: 0.5, transform: [
                    (Vec3::new(1.0, 2.0, 4.0), QU, V1),
                    (V0, Quat::from_xyzw(0.0, 0.70710677, 0.0, 0.70710677), V1),
                    (V0, QU, V0),
                    (Vec3::new(-1.5, -3.0, -6.0), QU, V1),
                ]},
                Frame {ratio: 1.0, transform: [
                    (Vec3::new(2.0, 4.0, 8.0), QU, V1),
                    (V0, Quat::from_xyzw(0.0, 1.0, 0.0, 0.0), V1),
                    (V0, QU, -V1),
                    (Vec3::new(-2.0, -4.0, -8.0), QU, V1),
                ]},
            ],
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_cache() {
        let mut translations = new_translations();
        translations[0] = Float3Key::new(0.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);
        translations[4] = Float3Key::new(1.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);

        let animation1 = Rc::new(Animation::from_raw(
            46.0,
            1,
            String::new(),
            translations.clone(),
            new_rotations(),
            new_scales(),
        ));

        let animation2 = Rc::new(Animation::from_raw(
            46.0,
            1,
            String::new(),
            translations.clone(),
            new_rotations(),
            new_scales(),
        ));

        let mut job = SamplingJob::default();
        job.set_animation(animation1.clone());
        job.set_context(SamplingContext::new(animation1.num_tracks()));

        fn run_test(job: &mut SamplingJob) -> Result<(), OzzError> {
            let output = make_buf(vec![TX; 1]);
            job.set_output(output.clone());
            job.run()?;
            for item in output.buf().unwrap().iter() {
                assert_eq!(item.translation.col(0), Vec3::new(1.0, -1.0, 5.0));
                assert_eq!(item.rotation.col(0), Quat::from_xyzw(0.0, 0.0, 0.0, 1.0));
                assert_eq!(item.scale.col(0), Vec3::new(1.0, 1.0, 1.0));
            }
            return Ok(());
        }

        job.set_ratio(0.0);
        run_test(&mut job).unwrap();

        // reuse cache
        run_test(&mut job).unwrap();

        // reset cache
        job.set_context(SamplingContext::new(animation1.num_tracks()));
        run_test(&mut job).unwrap();

        // change animation
        job.set_animation(animation2.clone());
        run_test(&mut job).unwrap();

        // change animation
        job.set_animation(animation2);
        run_test(&mut job).unwrap();
    }

    #[cfg(feature = "rkyv")]
    #[test]
    #[wasm_bindgen_test]
    fn test_rkyv() {
        use rkyv::Deserialize;

        let animation = Rc::new(Animation::from_path("./resource/animation-blending-1.ozz").unwrap());
        let aligned_tracks = animation.num_aligned_tracks();

        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(aligned_tracks));
        job.set_output(make_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        job.set_ratio(0.5);
        job.run().unwrap();

        let ctx: SamplingContext = job.context().unwrap().clone();
        let bytes = rkyv::to_bytes::<_, 256>(&ctx).unwrap();
        let archived = rkyv::check_archived_root::<SamplingContext>(&bytes[..]).unwrap();
        assert_eq!(archived.size as usize, ctx.size());
        assert_eq!(archived.animation_id, ctx.animation_id);
        assert_eq!(archived.ratio, ctx.ratio);
        assert_eq!(archived.translation_cursor as usize, ctx.translation_cursor);
        assert_eq!(archived.rotation_cursor as usize, ctx.rotation_cursor);
        assert_eq!(archived.scale_cursor as usize, ctx.scale_cursor);
        unsafe {
            assert_eq!(archived.translations(), ctx.translations());
            assert_eq!(archived.rotations(), ctx.rotations());
            assert_eq!(archived.scales(), ctx.scales());
            assert_eq!(archived.translation_keys(), ctx.translation_keys());
            assert_eq!(archived.rotation_keys(), ctx.rotation_keys());
            assert_eq!(archived.scale_keys(), ctx.scale_keys());
            assert_eq!(archived.outdated_translations(), ctx.outdated_translations());
            assert_eq!(archived.outdated_rotations(), ctx.outdated_rotations());
            assert_eq!(archived.outdated_scales(), ctx.outdated_scales());
        }

        let ctx_de: SamplingContext = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(ctx_de.size(), ctx.size());
        assert_eq!(ctx_de.animation_id, ctx.animation_id);
        assert_eq!(ctx_de.ratio, ctx.ratio);
        assert_eq!(ctx_de.translation_cursor, ctx.translation_cursor);
        assert_eq!(ctx_de.rotation_cursor, ctx.rotation_cursor);
        assert_eq!(ctx_de.scale_cursor, ctx.scale_cursor);
        assert_eq!(ctx_de.translations(), ctx.translations());
        assert_eq!(ctx_de.rotations(), ctx.rotations());
        assert_eq!(ctx_de.scales(), ctx.scales());
        assert_eq!(ctx_de.translation_keys(), ctx.translation_keys());
        assert_eq!(ctx_de.rotation_keys(), ctx.rotation_keys());
        assert_eq!(ctx_de.scale_keys(), ctx.scale_keys());
        assert_eq!(ctx_de.outdated_translations(), ctx.outdated_translations());
        assert_eq!(ctx_de.outdated_rotations(), ctx.outdated_rotations());
        assert_eq!(ctx_de.outdated_scales(), ctx.outdated_scales());
    }

    #[cfg(feature = "serde")]
    #[test]
    #[wasm_bindgen_test]
    fn test_serde() {
        let animation = Rc::new(Animation::from_path("./resource/animation-blending-1.ozz").unwrap());
        let aligned_tracks = animation.num_aligned_tracks();

        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(aligned_tracks));
        job.set_output(make_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        job.set_ratio(0.5);
        job.run().unwrap();

        let ctx: SamplingContext = job.context().unwrap().clone();
        let json = serde_json::to_string(&ctx).unwrap();
        let ctx_de: SamplingContext = serde_json::from_str(&json).unwrap();

        assert_eq!(ctx_de.size(), ctx.size());
        assert_eq!(ctx_de.animation_id, ctx.animation_id);
        assert_eq!(ctx_de.ratio, ctx.ratio);
        assert_eq!(ctx_de.translation_cursor, ctx.translation_cursor);
        assert_eq!(ctx_de.rotation_cursor, ctx.rotation_cursor);
        assert_eq!(ctx_de.scale_cursor, ctx.scale_cursor);
        assert_eq!(ctx_de.translations(), ctx.translations());
        assert_eq!(ctx_de.rotations(), ctx.rotations());
        assert_eq!(ctx_de.scales(), ctx.scales());
        assert_eq!(ctx_de.translation_keys(), ctx.translation_keys());
        assert_eq!(ctx_de.rotation_keys(), ctx.rotation_keys());
        assert_eq!(ctx_de.scale_keys(), ctx.scale_keys());
        assert_eq!(ctx_de.outdated_translations(), ctx.outdated_translations());
        assert_eq!(ctx_de.outdated_rotations(), ctx.outdated_rotations());
        assert_eq!(ctx_de.outdated_scales(), ctx.outdated_scales());
    }
}
