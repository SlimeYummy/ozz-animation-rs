//!
//! Sampling Job.
//!

use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
use std::simd::prelude::*;
use std::sync::{Arc, RwLock};
use std::{mem, ptr, slice};

use crate::animation::{Animation, Float3Key, KeyframesCtrl, QuaternionKey};
use crate::base::{align_ptr, align_usize, OzzError, OzzMutBuf, OzzObj};
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
            Ok(())
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<InterpSoaFloat3, D> for InterpSoaFloat3 {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<InterpSoaFloat3, D::Error> {
            Ok(from_archived!(*self))
        }
    }

    impl<C: ?Sized> CheckBytes<C> for InterpSoaFloat3 {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<InterpSoaFloat3>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            Ok(&*value)
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
            Ok(())
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<InterpSoaQuaternion, D> for InterpSoaQuaternion {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<InterpSoaQuaternion, D::Error> {
            Ok(from_archived!(*self))
        }
    }

    impl<C: ?Sized> CheckBytes<C> for InterpSoaQuaternion {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<InterpSoaQuaternion>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            Ok(&*value)
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
        seq.end()
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f32x4; 2], D::Error> {
        let tmp: [[f32; 4]; 2] = Deserialize::deserialize(deserializer)?;
        Ok([f32x4::from_array(tmp[0]), f32x4::from_array(tmp[1])])
    }
}

#[derive(Debug)]
struct SamplingContextInner {
    size: usize,
    max_tracks: usize,
    max_soa_tracks: usize,
    max_outdated: usize,
    animation_id: u64,
    ratio: f32,

    translations: *mut InterpSoaFloat3,
    translation_entries: *mut u32,
    translation_outdated: *mut u8,
    translation_next: usize,

    rotations: *mut InterpSoaQuaternion,
    rotation_entries: *mut u32,
    rotation_outdated: *mut u8,
    rotation_next: usize,

    scales: *mut InterpSoaFloat3,
    scale_entries: *mut u32,
    scale_outdated: *mut u8,
    scale_next: usize,
}

impl Default for SamplingContextInner {
    fn default() -> SamplingContextInner {
        SamplingContextInner {
            size: 0,
            max_tracks: 0,
            max_soa_tracks: 0,
            max_outdated: 0,
            animation_id: 0,
            ratio: 0.0,

            translations: ptr::null_mut(),
            translation_entries: ptr::null_mut(),
            translation_outdated: ptr::null_mut(),
            translation_next: 0,

            rotations: ptr::null_mut(),
            rotation_entries: ptr::null_mut(),
            rotation_outdated: ptr::null_mut(),
            rotation_next: 0,

            scales: ptr::null_mut(),
            scale_entries: ptr::null_mut(),
            scale_outdated: ptr::null_mut(),
            scale_next: 0,
        }
    }
}

/// Declares the context object used by the workload to take advantage of the
/// frame coherency of animation sampling.
pub struct SamplingContext(*mut SamplingContextInner);

unsafe impl Send for SamplingContext {}
unsafe impl Sync for SamplingContext {}

impl Debug for SamplingContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.0.is_null() {
            return f.debug_struct("SamplingContext").finish();
        }
        f.debug_struct("SamplingContext")
            .field("mem", &self.0)
            .field("animation_id", &self.animation_id())
            .field("ratio", &self.ratio())
            .finish()
    }
}

impl Clone for SamplingContext {
    fn clone(&self) -> Self {
        let mut ctx = SamplingContext::new(self.max_tracks());
        ctx.set_animation_id(self.animation_id());
        ctx.set_ratio(self.ratio());

        ctx.translations_mut().copy_from_slice(self.translations());
        ctx.translation_entries_mut()
            .copy_from_slice(self.translation_entries());
        ctx.translation_outdated_mut()
            .copy_from_slice(self.translation_outdated());
        ctx.set_translation_next(self.translation_next());

        ctx.rotations_mut().copy_from_slice(self.rotations());
        ctx.rotation_entries_mut().copy_from_slice(self.rotation_entries());
        ctx.rotation_outdated_mut().copy_from_slice(self.rotation_outdated());
        ctx.set_rotation_next(self.rotation_next());

        ctx.scales_mut().copy_from_slice(self.scales());
        ctx.scale_entries_mut().copy_from_slice(self.scale_entries());
        ctx.scale_outdated_mut().copy_from_slice(self.scale_outdated());
        ctx.set_scale_next(self.scale_next());
        ctx
    }
}

impl PartialEq for SamplingContext {
    fn eq(&self, other: &Self) -> bool {
        self.max_tracks() == other.max_tracks()
            && self.max_soa_tracks() == other.max_soa_tracks()
            && self.max_outdated() == other.max_outdated()
            && self.animation_id() == other.animation_id()
            && self.ratio() == other.ratio()
            && self.translations() == other.translations()
            && self.translation_entries() == other.translation_entries()
            && self.translation_outdated() == other.translation_outdated()
            && self.translation_next() == other.translation_next()
            && self.rotations() == other.rotations()
            && self.rotation_entries() == other.rotation_entries()
            && self.rotation_outdated() == other.rotation_outdated()
            && self.rotation_next() == other.rotation_next()
            && self.scales() == other.scales()
            && self.scale_entries() == other.scale_entries()
            && self.scale_outdated() == other.scale_outdated()
            && self.scale_next() == other.scale_next()
    }
}

impl Drop for SamplingContext {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.size(), mem::size_of::<f32x4>());
                alloc::dealloc(self.0 as *mut u8, layout);
            }
            self.0 = std::ptr::null_mut();
        }
    }
}

impl SamplingContext {
    #[inline(always)]
    fn inner(&self) -> &SamplingContextInner {
        unsafe { &*self.0 }
    }

    #[inline(always)]
    fn inner_mut(&mut self) -> &mut SamplingContextInner {
        unsafe { &mut *self.0 }
    }

    /// Create a new `SamplingContext`
    ///
    /// * `max_tracks` - The maximum number of tracks that the context can handle.
    pub fn new(max_tracks: usize) -> SamplingContext {
        const ALIGN: usize = mem::size_of::<f32x4>();
        let max_soa_tracks = max_tracks.div_ceil(4);
        let max_tracks = max_soa_tracks * 4;
        let max_outdated = max_soa_tracks.div_ceil(8);
        let translation_size = mem::size_of::<InterpSoaFloat3>() * max_soa_tracks
            + mem::size_of::<i32>() * max_tracks
            + mem::size_of::<u8>() * max_outdated;
        let rotation_size = mem::size_of::<InterpSoaQuaternion>() * max_soa_tracks
            + mem::size_of::<i32>() * max_tracks
            + mem::size_of::<u8>() * max_outdated;
        let scale_size = mem::size_of::<InterpSoaFloat3>() * max_soa_tracks
            + mem::size_of::<i32>() * max_tracks
            + mem::size_of::<u8>() * max_outdated;
        let size = align_usize(mem::size_of::<SamplingContextInner>(), ALIGN)
            + align_usize(translation_size, ALIGN)
            + align_usize(rotation_size, ALIGN)
            + align_usize(scale_size, ALIGN);

        unsafe {
            let layout = Layout::from_size_align_unchecked(size, mem::size_of::<f32x4>());
            let mut ptr = alloc::alloc(layout);
            let mut ctx = SamplingContext(ptr as *mut SamplingContextInner);
            let inner = ctx.inner_mut();
            *inner = SamplingContextInner::default();
            ptr = ptr.add(mem::size_of::<SamplingContextInner>());
            ptr = align_ptr(ptr, ALIGN);

            inner.size = size;
            inner.max_soa_tracks = max_soa_tracks;
            inner.max_tracks = max_tracks;
            inner.max_outdated = max_outdated;
            inner.animation_id = 0;
            inner.ratio = 0.0;

            inner.translations = ptr as *mut InterpSoaFloat3;
            ptr = ptr.add(mem::size_of::<InterpSoaFloat3>() * inner.max_soa_tracks);
            inner.translation_entries = ptr as *mut u32;
            ptr = ptr.add(mem::size_of::<u32>() * inner.max_tracks);
            inner.translation_outdated = ptr;
            ptr = ptr.add(inner.max_outdated);
            ptr = align_ptr(ptr, ALIGN);

            inner.rotations = ptr as *mut InterpSoaQuaternion;
            ptr = ptr.add(mem::size_of::<InterpSoaQuaternion>() * inner.max_soa_tracks);
            inner.rotation_entries = ptr as *mut u32;
            ptr = ptr.add(mem::size_of::<u32>() * inner.max_tracks);
            inner.rotation_outdated = ptr;
            ptr = ptr.add(inner.max_outdated);
            ptr = align_ptr(ptr, ALIGN);

            inner.scales = ptr as *mut InterpSoaFloat3;
            ptr = ptr.add(mem::size_of::<InterpSoaFloat3>() * inner.max_soa_tracks);
            inner.scale_entries = ptr as *mut u32;
            ptr = ptr.add(mem::size_of::<u32>() * inner.max_tracks);
            inner.scale_outdated = ptr;
            ptr = ptr.add(inner.max_outdated);
            ptr = align_ptr(ptr, ALIGN);

            assert_eq!(ptr, (ctx.0 as *mut u8).add(size));
            ctx
        }
    }

    /// Create a new `SamplingContext` from an `Animation`.
    ///
    /// * `animation` - The animation to sample. Use `animation.num_tracks()` as max_tracks.
    pub fn from_animation(animation: &Animation) -> SamplingContext {
        let mut ctx = SamplingContext::new(animation.num_tracks());
        ctx.inner_mut().animation_id = animation as *const _ as u64;
        ctx
    }

    /// Clear the `SamplingContext`.
    #[inline]
    pub fn clear(&mut self) {
        self.set_animation_id(0);
        self.set_ratio(0.0);
        self.set_translation_next(0);
        self.set_rotation_next(0);
        self.set_scale_next(0);
    }

    /// Clone the `SamplingContext` without the animation id. Usually used for serialization.
    #[inline]
    pub fn clone_without_animation_id(&self) -> SamplingContext {
        let mut ctx = self.clone();
        ctx.set_animation_id(0);
        ctx
    }

    /// The memory size of the context in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.inner().size
    }

    /// The maximum number of SoA tracks that the context can handle.
    #[inline]
    pub fn max_soa_tracks(&self) -> usize {
        self.inner().max_soa_tracks
    }

    /// The maximum number of tracks that the context can handle.
    #[inline]
    pub fn max_tracks(&self) -> usize {
        self.inner().max_tracks
    }

    /// The number of tracks that are outdated.
    #[inline]
    pub fn max_outdated(&self) -> usize {
        self.inner().max_outdated
    }

    /// The unique identifier of the animation that the context is sampling.
    #[inline]
    pub fn animation_id(&self) -> u64 {
        self.inner().animation_id
    }

    fn set_animation_id(&mut self, animation_id: u64) {
        self.inner_mut().animation_id = animation_id;
    }

    /// The current time ratio in the animation.
    #[inline]
    pub fn ratio(&self) -> f32 {
        self.inner().ratio
    }

    fn set_ratio(&mut self, ratio: f32) {
        self.inner_mut().ratio = ratio;
    }

    /// Soa hot data to interpolate.
    #[inline]
    pub fn translations(&self) -> &[InterpSoaFloat3] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.translations, inner.max_soa_tracks) }
    }

    #[inline]
    fn translations_mut(&mut self) -> &mut [InterpSoaFloat3] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.translations, inner.max_soa_tracks) }
    }

    /// The keys in the animation that are valid for the current time ratio.
    #[inline]
    pub fn translation_entries(&self) -> &[u32] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.translation_entries, inner.max_tracks) }
    }

    #[inline]
    fn translation_entries_mut(&mut self) -> &mut [u32] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.translation_entries, inner.max_tracks) }
    }

    /// Outdated soa entries. One bit per soa entry (32 joints per byte).
    #[inline]
    pub fn translation_outdated(&self) -> &[u8] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.translation_outdated, inner.max_outdated) }
    }

    #[inline]
    fn translation_outdated_mut(&mut self) -> &mut [u8] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.translation_outdated, inner.max_outdated) }
    }

    /// Next key to process in the animation.
    #[inline]
    pub fn translation_next(&self) -> usize {
        self.inner().translation_next
    }

    #[inline]
    fn set_translation_next(&mut self, next: usize) {
        self.inner_mut().translation_next = next;
    }

    /// Soa hot data to interpolate.
    #[inline]
    pub fn rotations(&self) -> &[InterpSoaQuaternion] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.rotations, inner.max_soa_tracks) }
    }

    #[inline]
    fn rotations_mut(&mut self) -> &mut [InterpSoaQuaternion] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.rotations, inner.max_soa_tracks) }
    }

    /// The keys in the animation that are valid for the current time ratio.
    #[inline]
    pub fn rotation_entries(&self) -> &[u32] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.rotation_entries, inner.max_tracks) }
    }

    #[inline]
    fn rotation_entries_mut(&mut self) -> &mut [u32] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.rotation_entries, inner.max_tracks) }
    }

    /// Outdated soa entries. One bit per soa entry (32 joints per byte).
    #[inline]
    pub fn rotation_outdated(&self) -> &[u8] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.rotation_outdated, inner.max_outdated) }
    }

    #[inline]
    fn rotation_outdated_mut(&mut self) -> &mut [u8] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.rotation_outdated, inner.max_outdated) }
    }

    /// Next key to process in the animation.
    #[inline]
    pub fn rotation_next(&self) -> usize {
        self.inner().rotation_next
    }

    #[inline]
    fn set_rotation_next(&mut self, next: usize) {
        self.inner_mut().rotation_next = next;
    }

    /// Soa hot data to interpolate.
    #[inline]
    pub fn scales(&self) -> &[InterpSoaFloat3] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.scales, inner.max_soa_tracks) }
    }

    #[inline]
    fn scales_mut(&mut self) -> &mut [InterpSoaFloat3] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.scales, inner.max_soa_tracks) }
    }

    /// The keys in the animation that are valid for the current time ratio.
    #[inline]
    pub fn scale_entries(&self) -> &[u32] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.scale_entries, inner.max_tracks) }
    }

    #[inline]
    fn scale_entries_mut(&mut self) -> &mut [u32] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.scale_entries, inner.max_tracks) }
    }

    /// Outdated soa entries. One bit per soa entry (32 joints per byte).
    #[inline]
    pub fn scale_outdated(&self) -> &[u8] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts(inner.scale_outdated, inner.max_outdated) }
    }

    #[inline]
    fn scale_outdated_mut(&mut self) -> &mut [u8] {
        let inner = self.inner();
        unsafe { slice::from_raw_parts_mut(inner.scale_outdated, inner.max_outdated) }
    }

    /// Next key to process in the animation.
    #[inline]
    pub fn scale_next(&self) -> usize {
        self.inner().scale_next
    }

    #[inline]
    fn set_scale_next(&mut self, next: usize) {
        self.inner_mut().scale_next = next;
    }

    #[inline]
    fn translation_update_args<'t>(&'t mut self, animation: &Animation) -> UpdateArgs<'t> {
        let inner = self.inner_mut();
        UpdateArgs {
            num_tracks: animation.num_tracks(),
            num_soa_tracks: animation.num_soa_tracks(),
            entries: unsafe { slice::from_raw_parts_mut(inner.translation_entries, inner.max_tracks) },
            outdated: unsafe { slice::from_raw_parts_mut(inner.translation_outdated, inner.max_outdated) },
            next: &mut inner.translation_next,
        }
    }

    #[inline]
    fn translation_decompress_args(&self) -> DecompressArgs<'_, InterpSoaFloat3> {
        let inner = self.inner();
        DecompressArgs {
            entries: unsafe { slice::from_raw_parts(inner.translation_entries, inner.max_tracks) },
            outdated: unsafe { slice::from_raw_parts_mut(inner.translation_outdated, inner.max_outdated) },
            values: unsafe { slice::from_raw_parts_mut(inner.translations, inner.max_soa_tracks) },
        }
    }

    #[inline]
    fn rotation_update_args(&mut self, animation: &Animation) -> UpdateArgs<'_> {
        let inner = self.inner_mut();
        UpdateArgs {
            num_tracks: animation.num_tracks(),
            num_soa_tracks: animation.num_soa_tracks(),
            entries: unsafe { slice::from_raw_parts_mut(inner.rotation_entries, inner.max_tracks) },
            outdated: unsafe { slice::from_raw_parts_mut(inner.rotation_outdated, inner.max_outdated) },
            next: &mut inner.rotation_next,
        }
    }

    #[inline]
    fn rotation_decompress_args(&self) -> DecompressArgs<'_, InterpSoaQuaternion> {
        let inner = self.inner();
        DecompressArgs {
            entries: unsafe { slice::from_raw_parts(inner.rotation_entries, inner.max_tracks) },
            outdated: unsafe { slice::from_raw_parts_mut(inner.rotation_outdated, inner.max_outdated) },
            values: unsafe { slice::from_raw_parts_mut(inner.rotations, inner.max_soa_tracks) },
        }
    }

    #[inline]
    fn scale_update_args(&mut self, animation: &Animation) -> UpdateArgs<'_> {
        let inner = self.inner_mut();
        UpdateArgs {
            num_tracks: animation.num_tracks(),
            num_soa_tracks: animation.num_soa_tracks(),
            entries: unsafe { slice::from_raw_parts_mut(inner.scale_entries, inner.max_tracks) },
            outdated: unsafe { slice::from_raw_parts_mut(inner.scale_outdated, inner.max_outdated) },
            next: &mut inner.scale_next,
        }
    }

    #[inline]
    fn scale_decompress_args(&self) -> DecompressArgs<'_, InterpSoaFloat3> {
        let inner = self.inner();
        DecompressArgs {
            entries: unsafe { slice::from_raw_parts(inner.scale_entries, inner.max_tracks) },
            outdated: unsafe { slice::from_raw_parts_mut(inner.scale_outdated, inner.max_outdated) },
            values: unsafe { slice::from_raw_parts_mut(inner.scales, inner.max_soa_tracks) },
        }
    }
}

#[cfg(feature = "rkyv")]
pub struct ArchivedSamplingContext {
    pub max_tracks: u32,
    pub animation_id: u64,
    pub ratio: f32,

    pub translations: rkyv::vec::ArchivedVec<InterpSoaFloat3>,
    pub translation_entries: rkyv::vec::ArchivedVec<u32>,
    pub translation_outdated: rkyv::vec::ArchivedVec<u8>,
    pub translation_next: u32,

    pub rotations: rkyv::vec::ArchivedVec<InterpSoaQuaternion>,
    pub rotation_entries: rkyv::vec::ArchivedVec<u32>,
    pub rotation_outdated: rkyv::vec::ArchivedVec<u8>,
    pub rotation_next: u32,

    pub scales: rkyv::vec::ArchivedVec<InterpSoaFloat3>,
    pub scale_entries: rkyv::vec::ArchivedVec<u32>,
    pub scale_outdated: rkyv::vec::ArchivedVec<u8>,
    pub scale_next: u32,
}

#[cfg(feature = "rkyv")]
const _: () = {
    use bytecheck::CheckBytes;
    use rkyv::ser::{ScratchSpace, Serializer};
    use rkyv::vec::{ArchivedVec, VecResolver};
    use rkyv::{from_archived, out_field, Archive, Deserialize, Fallible, Serialize};
    use std::io::{Error, ErrorKind};

    pub struct SamplingContextResolver {
        translations: VecResolver,
        translation_entries: VecResolver,
        translation_outdateds: VecResolver,

        rotations: VecResolver,
        rotation_entries: VecResolver,
        rotation_outdateds: VecResolver,

        scales: VecResolver,
        scale_entries: VecResolver,
        scale_outdateds: VecResolver,
    }

    impl Archive for SamplingContext {
        type Archived = ArchivedSamplingContext;
        type Resolver = SamplingContextResolver;

        unsafe fn resolve(&self, pos: usize, resolver: SamplingContextResolver, out: *mut ArchivedSamplingContext) {
            let (fp, fo) = out_field!(out.max_tracks);
            usize::resolve(&self.max_tracks(), pos + fp, (), fo);
            let (fp, fo) = out_field!(out.animation_id);
            u64::resolve(&self.animation_id(), pos + fp, (), fo);
            let (fp, fo) = out_field!(out.ratio);
            f32::resolve(&self.ratio(), pos + fp, (), fo);

            let (fp, fo) = out_field!(out.translations);
            ArchivedVec::resolve_from_slice(self.translations(), pos + fp, resolver.translations, fo);
            let (fp, fo) = out_field!(out.rotations);
            ArchivedVec::resolve_from_slice(self.rotations(), pos + fp, resolver.rotations, fo);
            let (fp, fo) = out_field!(out.scales);
            ArchivedVec::resolve_from_slice(self.scales(), pos + fp, resolver.scales, fo);

            let (fp, fo) = out_field!(out.translation_entries);
            ArchivedVec::resolve_from_slice(self.translation_entries(), pos + fp, resolver.translation_entries, fo);
            let (fp, fo) = out_field!(out.rotation_entries);
            ArchivedVec::resolve_from_slice(self.rotation_entries(), pos + fp, resolver.rotation_entries, fo);
            let (fp, fo) = out_field!(out.scale_entries);
            ArchivedVec::resolve_from_slice(self.scale_entries(), pos + fp, resolver.scale_entries, fo);

            let (fp, fo) = out_field!(out.translation_outdated);
            ArchivedVec::resolve_from_slice(
                self.translation_outdated(),
                pos + fp,
                resolver.translation_outdateds,
                fo,
            );
            let (fp, fo) = out_field!(out.rotation_outdated);
            ArchivedVec::resolve_from_slice(self.rotation_outdated(), pos + fp, resolver.rotation_outdateds, fo);
            let (fp, fo) = out_field!(out.scale_outdated);
            ArchivedVec::resolve_from_slice(self.scale_outdated(), pos + fp, resolver.scale_outdateds, fo);

            let (fp, fo) = out_field!(out.translation_next);
            usize::resolve(&self.translation_next(), pos + fp, (), fo);
            let (fp, fo) = out_field!(out.rotation_next);
            usize::resolve(&self.rotation_next(), pos + fp, (), fo);
            let (fp, fo) = out_field!(out.scale_next);
            usize::resolve(&self.scale_next(), pos + fp, (), fo);
        }
    }

    impl<S: Serializer + ScratchSpace + ?Sized> Serialize<S> for SamplingContext {
        fn serialize(&self, serializer: &mut S) -> Result<Self::Resolver, S::Error> {
            serializer.align_for::<InterpSoaFloat3>()?;
            Ok(SamplingContextResolver {
                translations: ArchivedVec::serialize_from_slice(self.translations(), serializer)?,
                rotations: ArchivedVec::serialize_from_slice(self.rotations(), serializer)?,
                scales: ArchivedVec::serialize_from_slice(self.scales(), serializer)?,
                translation_entries: ArchivedVec::serialize_from_slice(self.translation_entries(), serializer)?,
                rotation_entries: ArchivedVec::serialize_from_slice(self.rotation_entries(), serializer)?,
                scale_entries: ArchivedVec::serialize_from_slice(self.scale_entries(), serializer)?,
                translation_outdateds: ArchivedVec::serialize_from_slice(self.translation_outdated(), serializer)?,
                rotation_outdateds: ArchivedVec::serialize_from_slice(self.rotation_outdated(), serializer)?,
                scale_outdateds: ArchivedVec::serialize_from_slice(self.scale_outdated(), serializer)?,
            })
        }
    }

    impl<D: Fallible + ?Sized> Deserialize<SamplingContext, D> for ArchivedSamplingContext {
        #[inline]
        fn deserialize(&self, _: &mut D) -> Result<SamplingContext, D::Error> {
            let archived = from_archived!(self);
            let mut context = SamplingContext::new(archived.max_tracks as usize);
            context.set_animation_id(archived.animation_id);
            context.set_ratio(archived.ratio);
            context.translations_mut().copy_from_slice(&archived.translations);
            context.rotations_mut().copy_from_slice(&archived.rotations);
            context.scales_mut().copy_from_slice(&archived.scales);
            context
                .translation_entries_mut()
                .copy_from_slice(&archived.translation_entries);
            context
                .rotation_entries_mut()
                .copy_from_slice(&archived.rotation_entries);
            context.scale_entries_mut().copy_from_slice(&archived.scale_entries);
            context
                .translation_outdated_mut()
                .copy_from_slice(&archived.translation_outdated);
            context
                .rotation_outdated_mut()
                .copy_from_slice(&archived.rotation_outdated);
            context.scale_outdated_mut().copy_from_slice(&archived.scale_outdated);
            context.set_translation_next(archived.translation_next as usize);
            context.set_rotation_next(archived.rotation_next as usize);
            context.set_scale_next(archived.scale_next as usize);
            Ok(context)
        }
    }

    impl<C: ?Sized> CheckBytes<C> for ArchivedSamplingContext {
        type Error = Error;

        #[inline]
        unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
            if value as usize % mem::align_of::<f32x4>() != 0 {
                return Err(Error::new(ErrorKind::InvalidData, "must be aligned to 16 bytes"));
            }
            Ok(&*value)
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
            map.serialize_entry("max_outdated", &self.max_outdated())?;
            map.serialize_entry("animation_id", &self.animation_id())?;
            map.serialize_entry("ratio", &self.ratio())?;
            map.serialize_entry("translations", self.translations())?;
            map.serialize_entry("rotations", self.rotations())?;
            map.serialize_entry("scales", self.scales())?;
            map.serialize_entry("translation_entries", self.translation_entries())?;
            map.serialize_entry("translation_outdated", self.translation_outdated())?;
            map.serialize_entry("translation_next", &self.translation_next())?;
            map.serialize_entry("rotation_entries", self.rotation_entries())?;
            map.serialize_entry("rotation_outdated", self.rotation_outdated())?;
            map.serialize_entry("rotation_next", &self.rotation_next())?;
            map.serialize_entry("scale_entries", self.scale_entries())?;
            map.serialize_entry("scale_outdated", self.scale_outdated())?;
            map.serialize_entry("scale_next", &self.scale_next())?;
            map.end()
        }
    }

    impl<'de> Deserialize<'de> for SamplingContext {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<SamplingContext, D::Error> {
            deserializer.deserialize_map(SamplingContextVisitor)
        }
    }

    struct SamplingContextVisitor;

    impl<'de> Visitor<'de> for SamplingContextVisitor {
        type Value = SamplingContext;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("struct SamplingContext")
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
                    "animation_id" => ctx.set_animation_id(map.next_value()?),
                    "ratio" => ctx.set_ratio(map.next_value()?),
                    "translations" => {
                        ctx.translations_mut()
                            .copy_from_slice(&map.next_value::<Vec<InterpSoaFloat3>>()?);
                    }
                    "translation_entries" => {
                        ctx.translation_entries_mut()
                            .copy_from_slice(&map.next_value::<Vec<u32>>()?);
                    }
                    "translation_outdated" => {
                        ctx.translation_outdated_mut()
                            .copy_from_slice(&map.next_value::<Vec<u8>>()?);
                    }
                    "translation_next" => ctx.set_translation_next(map.next_value()?),
                    "rotations" => {
                        ctx.rotations_mut()
                            .copy_from_slice(&map.next_value::<Vec<InterpSoaQuaternion>>()?);
                    }
                    "rotation_entries" => {
                        ctx.rotation_entries_mut()
                            .copy_from_slice(&map.next_value::<Vec<u32>>()?);
                    }
                    "rotation_outdated" => {
                        ctx.rotation_outdated_mut()
                            .copy_from_slice(&map.next_value::<Vec<u8>>()?);
                    }
                    "rotation_next" => ctx.set_rotation_next(map.next_value()?),
                    "scales" => {
                        ctx.scales_mut()
                            .copy_from_slice(&map.next_value::<Vec<InterpSoaFloat3>>()?);
                    }
                    "scale_entries" => {
                        ctx.scale_entries_mut().copy_from_slice(&map.next_value::<Vec<u32>>()?);
                    }
                    "scale_outdated" => {
                        ctx.scale_outdated_mut().copy_from_slice(&map.next_value::<Vec<u8>>()?);
                    }
                    "scale_next" => ctx.set_scale_next(map.next_value()?),
                    _ => {
                        map.next_value::<serde::de::IgnoredAny>()?;
                    }
                }
            }
            Ok(ctx)
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
        self
    }

    #[inline(always)]
    fn as_mut(&mut self) -> &mut SamplingContext {
        self
    }
}

impl AsSamplingContext for &'_ mut SamplingContext {
    #[inline(always)]
    fn as_ref(&self) -> &SamplingContext {
        self
    }

    #[inline(always)]
    fn as_mut(&mut self) -> &mut SamplingContext {
        self
    }
}

struct UpdateArgs<'t> {
    num_tracks: usize,
    num_soa_tracks: usize,
    entries: &'t mut [u32],
    outdated: &'t mut [u8],
    next: &'t mut usize,
}

struct DecompressArgs<'t, T> {
    entries: &'t [u32],
    outdated: &'t mut [u8],
    values: &'t mut [T],
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
        SamplingJob {
            animation: None,
            context: None,
            ratio: 0.0,
            output: None,
        }
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
        self.animation.as_ref()
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
        self.context.as_ref()
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
        self.context.take()
    }

    /// Gets the time ratio of `SamplingJob`.
    #[inline]
    pub fn ratio(&self) -> f32 {
        self.ratio
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
        self.output.as_ref()
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
        (|| {
            let animation = self.animation.as_ref()?.obj();
            let context = self.context.as_ref()?;
            let output = self.output.as_ref()?.buf().ok()?;

            let mut ok = context.as_ref().max_soa_tracks() >= animation.num_soa_tracks();
            ok &= output.len() >= animation.num_soa_tracks();
            Some(ok)
        })()
        .unwrap_or(false)
    }

    /// Runs job's sampling task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        let anim = self.animation.as_ref().ok_or(OzzError::InvalidJob)?.obj();
        let ctx = self.context.as_mut().ok_or(OzzError::InvalidJob)?;
        let mut output = self.output.as_mut().ok_or(OzzError::InvalidJob)?.mut_buf()?;

        let mut ok = ctx.as_ref().max_soa_tracks() >= anim.num_soa_tracks();
        ok &= output.len() >= anim.num_soa_tracks();
        if !ok {
            return Err(OzzError::InvalidJob);
        }

        if anim.num_soa_tracks() == 0 {
            return Ok(());
        }

        let prev_ratio = Self::step_context(ctx.as_mut(), anim, self.ratio);

        let args = ctx.as_mut().translation_update_args(anim);
        Self::update_cache(args, anim, &anim.translations_ctrl(), self.ratio, prev_ratio);
        let args = ctx.as_mut().translation_decompress_args();
        Self::decompress_float3(args, anim.timepoints(), &anim.translations_ctrl(), anim.translations());

        let args = ctx.as_mut().rotation_update_args(anim);
        Self::update_cache(args, anim, &anim.rotations_ctrl(), self.ratio, prev_ratio);
        let args = ctx.as_mut().rotation_decompress_args();
        Self::decompress_quat(args, anim.timepoints(), &anim.rotations_ctrl(), anim.rotations());

        let args = ctx.as_mut().scale_update_args(anim);
        Self::update_cache(args, anim, &anim.scales_ctrl(), self.ratio, prev_ratio);
        let args = ctx.as_mut().scale_decompress_args();
        Self::decompress_float3(args, anim.timepoints(), &anim.scales_ctrl(), anim.scales());

        Self::interpolates(anim, ctx.as_mut(), self.ratio, &mut output)?;
        Ok(())
    }

    #[inline]
    fn step_context(ctx: &mut SamplingContext, animation: &Animation, ratio: f32) -> f32 {
        let animation_id = animation as *const _ as u64;
        if ctx.animation_id() != animation_id {
            ctx.set_animation_id(animation_id);
            ctx.set_translation_next(0);
            ctx.set_rotation_next(0);
            ctx.set_scale_next(0);
        }
        let prev_ratio = ctx.ratio();
        ctx.set_ratio(ratio);
        prev_ratio
    }

    fn update_cache(
        args: UpdateArgs<'_>,
        animation: &Animation,
        ctrl: &KeyframesCtrl<'_>,
        ratio: f32,
        prev_ratio: f32,
    ) {
        assert!(ctrl.previouses.len() >= args.num_tracks * 2);
        let num_keys = ctrl.previouses.len();

        let mut next = *args.next;
        assert!(next == 0 || (next >= args.num_tracks * 2 && next <= num_keys));

        // Initialize
        let delta = ratio - prev_ratio;
        if next == 0 || (delta.abs() > ctrl.iframe_interval / 2.0) {
            let mut iframe = -1;
            if !ctrl.iframe_desc.is_empty() {
                iframe = (0.5 + ratio / ctrl.iframe_interval) as i32;
            } else if next == 0 || delta < 0.0 {
                iframe = 0;
            }

            if iframe >= 0 {
                next = Self::initialize_cache(ctrl, iframe as usize, args.entries);
                assert!(next >= args.num_tracks * 2 && next <= num_keys);
                Self::outdate_cache(args.outdated, args.num_soa_tracks);
            }
        }

        // Forward
        let mut track = 0;
        while next < num_keys
            && Self::key_ratio(ctrl, animation.timepoints(), next - ctrl.previouses[next] as usize) <= ratio
        {
            track = Self::track_forward(args.entries, ctrl.previouses, next, track, args.num_tracks);
            assert!((args.entries[track] as usize) == next - (ctrl.previouses[next] as usize));
            args.outdated[track / 32] |= 1 << ((track & 0x1F) / 4);
            args.entries[track] = next as u32;
            next += 1;
        }

        // Rewinds
        while Self::key_ratio(
            ctrl,
            animation.timepoints(),
            next - 1 - ctrl.previouses[next - 1] as usize,
        ) > ratio
        {
            assert!(next > args.num_tracks * 2);
            track = Self::track_backward(args.entries, next - 1, track, args.num_tracks);
            args.outdated[track / 32] |= 1 << ((track & 0x1F) / 4);
            assert!((args.entries[track] as usize) == next - 1);
            let previous = ctrl.previouses[args.entries[track] as usize];
            assert!((args.entries[track] as usize) >= (previous as usize) + args.num_tracks);
            args.entries[track] -= previous as u32;
            next -= 1;
        }

        assert!(next >= args.num_tracks * 2 && next <= num_keys);
        *args.next = next;
    }

    #[inline]
    fn initialize_cache(ctrl: &KeyframesCtrl<'_>, iframe: usize, entries: &mut [u32]) -> usize {
        if iframe > 0 {
            let iframe = (iframe - 1) * 2;
            let offset = ctrl.iframe_desc[iframe] as usize;
            decode_gv4_stream(&ctrl.iframe_entries[offset..], entries);
            (ctrl.iframe_desc[iframe + 1] + 1) as usize
        } else {
            let num_tracks = entries.len() as u32;
            for i in 0..num_tracks {
                entries[i as usize] = i + num_tracks;
            }
            (num_tracks * 2) as usize
        }
    }

    #[inline]
    fn outdate_cache(outdated: &mut [u8], num_soa_tracks: usize) {
        let num_outdated_flags = num_soa_tracks.div_ceil(8);
        let mut i = 0;
        while i < num_outdated_flags - 1 {
            outdated[i] = 0xFF;
            i += 1;
        }
        outdated[i] = 0xFF >> (num_outdated_flags * 8 - num_soa_tracks);
    }

    #[inline]
    fn track_forward(cache: &[u32], previouses: &[u16], key: usize, last_track: usize, num_tracks: usize) -> usize {
        #![allow(clippy::needless_range_loop)] // to keep same style with track_backward
        assert!(key < previouses.len());
        assert!(last_track < num_tracks);

        let target = key - previouses[key] as usize;
        for entry in last_track..num_tracks {
            if (cache[entry] as usize) == target {
                return entry;
            }
        }
        for entry in 0..num_tracks {
            if (cache[entry] as usize) == target {
                return entry;
            }
            assert!(entry < last_track);
        }
        0
    }

    #[inline]
    fn track_backward(cache: &[u32], target: usize, last_track: usize, num_tracks: usize) -> usize {
        assert!(last_track < num_tracks);

        for entry in (0..=last_track).rev() {
            if (cache[entry] as usize) == target {
                return entry;
            }
        }
        for entry in (0..=num_tracks - 1).rev() {
            if (cache[entry] as usize) == target {
                return entry;
            }
            assert!(entry > last_track);
        }
        0
    }

    #[inline(always)]
    fn key_ratio(ctrl: &KeyframesCtrl<'_>, timepoints: &[f32], at: usize) -> f32 {
        timepoints[ctrl.ratios[at] as usize]
    }

    #[inline(always)]
    fn key_ratio_simd(ctrl: &KeyframesCtrl<'_>, timepoints: &[f32], ats: &[u32]) -> f32x4 {
        f32x4::from_array([
            timepoints[ctrl.ratios[ats[0] as usize] as usize],
            timepoints[ctrl.ratios[ats[1] as usize] as usize],
            timepoints[ctrl.ratios[ats[2] as usize] as usize],
            timepoints[ctrl.ratios[ats[3] as usize] as usize],
        ])
    }

    fn decompress_float3(
        args: DecompressArgs<'_, InterpSoaFloat3>,
        timepoints: &[f32],
        ctrl: &KeyframesCtrl<'_>,
        compressed: &[Float3Key],
    ) {
        for j in 0..args.outdated.len() {
            let mut outdated = args.outdated[j];
            for i in (8 * j)..(8 * j + 8) {
                if outdated & 1 != 0 {
                    let rights = &args.entries[i * 4..i * 4 + 4];
                    let lefts = [
                        rights[0] - (ctrl.previouses[rights[0] as usize] as u32),
                        rights[1] - (ctrl.previouses[rights[1] as usize] as u32),
                        rights[2] - (ctrl.previouses[rights[2] as usize] as u32),
                        rights[3] - (ctrl.previouses[rights[3] as usize] as u32),
                    ];

                    let k00 = compressed[lefts[0] as usize];
                    let k10 = compressed[lefts[1] as usize];
                    let k20 = compressed[lefts[2] as usize];
                    let k30 = compressed[lefts[3] as usize];
                    args.values[i].ratio[0] = Self::key_ratio_simd(ctrl, timepoints, &lefts);
                    Float3Key::simd_decompress(&k00, &k10, &k20, &k30, &mut args.values[i].value[0]);

                    let k01 = compressed[rights[0] as usize];
                    let k11 = compressed[rights[1] as usize];
                    let k21 = compressed[rights[2] as usize];
                    let k31 = compressed[rights[3] as usize];
                    args.values[i].ratio[1] = Self::key_ratio_simd(ctrl, timepoints, rights);
                    Float3Key::simd_decompress(&k01, &k11, &k21, &k31, &mut args.values[i].value[1]);
                }
                outdated >>= 1;
            }
        }
    }

    fn decompress_quat(
        args: DecompressArgs<'_, InterpSoaQuaternion>,
        timepoints: &[f32],
        ctrl: &KeyframesCtrl<'_>,
        compressed: &[QuaternionKey],
    ) {
        for j in 0..args.outdated.len() {
            let mut outdated = args.outdated[j];
            for i in (8 * j)..(8 * j + 8) {
                if outdated & 1 != 0 {
                    let rights = &args.entries[i * 4..i * 4 + 4];
                    let lefts = [
                        rights[0] - (ctrl.previouses[rights[0] as usize] as u32),
                        rights[1] - (ctrl.previouses[rights[1] as usize] as u32),
                        rights[2] - (ctrl.previouses[rights[2] as usize] as u32),
                        rights[3] - (ctrl.previouses[rights[3] as usize] as u32),
                    ];

                    let k00 = compressed[lefts[0] as usize];
                    let k10 = compressed[lefts[1] as usize];
                    let k20 = compressed[lefts[2] as usize];
                    let k30 = compressed[lefts[3] as usize];
                    args.values[i].ratio[0] = Self::key_ratio_simd(ctrl, timepoints, &lefts);
                    QuaternionKey::simd_decompress(&k00, &k10, &k20, &k30, &mut args.values[i].value[0]);

                    let k01 = compressed[rights[0] as usize];
                    let k11 = compressed[rights[1] as usize];
                    let k21 = compressed[rights[2] as usize];
                    let k31 = compressed[rights[3] as usize];
                    args.values[i].ratio[1] = Self::key_ratio_simd(ctrl, timepoints, rights);
                    QuaternionKey::simd_decompress(&k01, &k11, &k21, &k31, &mut args.values[i].value[1]);
                }
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
        for (idx, out) in output.iter_mut().enumerate().take(animation.num_soa_tracks()) {
            let translation = &ctx.translations()[idx];
            let translation_ratio = (ratio4 - translation.ratio[0]) / (translation.ratio[1] - translation.ratio[0]);
            out.translation = SoaVec3::lerp(&translation.value[0], &translation.value[1], translation_ratio);

            let rotation = &ctx.rotations()[idx];
            let rotation_ratio = (ratio4 - rotation.ratio[0]) / (rotation.ratio[1] - rotation.ratio[0]);
            out.rotation = SoaQuat::nlerp(&rotation.value[0], &rotation.value[1], rotation_ratio);

            let scale = &ctx.scales()[idx];
            let scale_ratio = (ratio4 - scale.ratio[0]) / (scale.ratio[1] - scale.ratio[0]);
            out.scale = SoaVec3::lerp(&scale.value[0], &scale.value[1], scale_ratio);
        }
        Ok(())
    }
}

#[inline]
fn decode_gv4<'t>(buffer: &'t [u8], output: &mut [u32]) -> &'t [u8] {
    assert!(buffer.len() >= 5, "Input buffer is too small.");
    assert!(output.len() == 4, "Output size must be 4");

    #[inline]
    fn load(input: &[u8]) -> u32 {
        input[0] as u32 | ((input[1] as u32) << 8) | ((input[2] as u32) << 16) | ((input[3] as u32) << 24)
    }

    let mut in_buf = &buffer[1..];
    let prefix = buffer[0];

    const MASK: [u32; 4] = [0xff, 0xffff, 0xffffff, 0xffffffff];
    let k0 = (prefix & 0x3) as usize;
    output[0] = load(in_buf) & MASK[k0];
    in_buf = &in_buf[k0 + 1..];
    let k1 = ((prefix >> 2) & 0x3) as usize;
    output[1] = load(in_buf) & MASK[k1];
    in_buf = &in_buf[k1 + 1..];
    let k2 = ((prefix >> 4) & 0x3) as usize;
    output[2] = load(in_buf) & MASK[k2];
    in_buf = &in_buf[k2 + 1..];
    let k3 = (prefix >> 6) as usize;
    output[3] = load(in_buf) & MASK[k3];
    in_buf = &in_buf[k3 + 1..];
    in_buf
}

fn decode_gv4_stream<'t>(buffer: &'t [u8], stream: &mut [u32]) -> &'t [u8] {
    assert!(stream.len() % 4 == 0, "Input stream must be multiple of 4");
    assert!(
        buffer.len() >= (stream.len() + stream.len() / 4),
        "Output buffer is too small"
    );

    let mut in_buf = buffer;
    for chunk in stream.chunks_mut(4) {
        in_buf = decode_gv4(in_buf, chunk);
    }
    in_buf
}

#[cfg(test)]
mod sampling_tests {
    use glam::{Quat, Vec3};
    use wasm_bindgen_test::*;

    use super::*;
    use crate::animation::AnimationRaw;
    use crate::base::OzzBuf;

    fn make_buf<T>(v: Vec<T>) -> Rc<RefCell<Vec<T>>> {
        Rc::new(RefCell::new(v))
    }

    // f16 -> f32
    // ignore overflow, infinite, NaN
    pub fn f16(f: f32) -> u16 {
        let n = f.to_bits();
        if (n & 0x7FFFFFFF) == 0 {
            return (n >> 16) as u16;
        }
        let sign = (n >> 16) & 0x8000;
        let expo = (((n & 0x7f800000) - 0x38000000) >> 13) & 0x7c00;
        let base = (n >> 13) & 0x03ff;
        (sign | expo | base) as u16
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_validity() {
        let animation = Rc::new(Animation::from_path("./resource/playback/animation.ozz").unwrap());
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

    const V0: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    const V1: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    const QU: Quat = Quat::from_xyzw(0.0, 0.0, 0.0, 1.0);
    const TX: SoaTransform = SoaTransform {
        translation: SoaVec3::splat_vec3(Vec3::new(1234.5678, 1234.5678, 1234.5678)),
        rotation: SoaQuat::splat_quat(Quat::from_xyzw(1234.5678, 1234.5678, 1234.5678, 1234.5678)),
        scale: SoaVec3::splat_vec3(Vec3::new(1234.5678, 1234.5678, 1234.5678)),
    };

    fn empty_translations() -> Vec<Float3Key> {
        vec![Float3Key::new([f16(0.0); 3]); 8]
    }

    fn empty_rotations() -> Vec<QuaternionKey> {
        vec![QuaternionKey::new([65531, 65533, 32766]); 8]
    }

    fn empty_scales() -> Vec<Float3Key> {
        vec![Float3Key::new([f16(1.0); 3]); 8]
    }

    fn empty_ratios(n: u16) -> Vec<u16> {
        vec![0, 0, 0, 0, n, n, n, n]
    }

    fn empty_previouses() -> Vec<u16> {
        vec![0, 0, 0, 0, 4, 4, 4, 4]
    }

    fn empty_animation_raw<const S: usize>(duration: f32) -> AnimationRaw {
        AnimationRaw {
            duration,
            num_tracks: S as u32,
            timepoints: vec![],
            translations: empty_translations(),
            t_ratios: empty_ratios(S as u16),
            t_previouses: empty_previouses(),
            rotations: empty_rotations(),
            r_ratios: empty_ratios(S as u16),
            r_previouses: empty_previouses(),
            scales: empty_scales(),
            s_ratios: empty_ratios(S as u16),
            s_previouses: empty_previouses(),
            ..Default::default()
        }
    }

    #[derive(Debug, Clone)]
    struct Frame<const S: usize> {
        ratio: f32,
        transform: [(Vec3, Quat, Vec3); S],
    }

    fn execute_test<const S: usize>(animation: AnimationRaw, frames: Vec<Frame<S>>) {
        let animation = Rc::new(Animation::from_raw(&animation));
        let mut job = SamplingJob::default();
        job.set_animation(animation);
        job.set_context(SamplingContext::new(S));
        let output = make_buf(vec![TX; S + 1]);

        for frame in frames.iter() {
            job.set_output(output.clone());
            job.set_ratio(frame.ratio);
            job.run().unwrap();

            if S == 0 {
                assert_eq!(output.borrow()[0], TX);
            }

            for idx in 0..S {
                let out = output.borrow()[idx / 4];
                assert!(
                    out.translation.vec3(idx % 4).abs_diff_eq(frame.transform[idx].0, 1e-6),
                    "ratio={} translation idx={} left={}, right={}",
                    frame.ratio,
                    idx,
                    out.translation.vec3(idx % 4),
                    frame.transform[idx].0
                );
                assert!(
                    out.rotation.quat(idx % 4).abs_diff_eq(frame.transform[idx].1, 5e-5),
                    "ratio={} rotation idx={} left={}, right={}",
                    frame.ratio,
                    idx,
                    out.rotation.quat(idx % 4),
                    frame.transform[idx].1
                );
                assert_eq!(
                    out.scale.vec3(idx % 4),
                    frame.transform[idx].2,
                    "ratio={} scale idx={}",
                    frame.ratio,
                    idx
                );
            }

            assert_eq!(job.context().unwrap().ratio(), f32_clamp_or_max(frame.ratio, 0.0, 1.0));
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling() {
        fn frame(ratio: f32, t1: f32, t2: f32, t3: f32, t4: f32) -> Frame<4> {
            Frame {
                ratio,
                transform: [
                    (Vec3::new(t1, 0.0, 0.0), QU, V1),
                    (Vec3::new(t2, 0.0, 0.0), QU, V1),
                    (Vec3::new(t3, 0.0, 0.0), QU, V1),
                    (Vec3::new(t4, 0.0, 0.0), QU, V1),
                ],
            }
        }

        let mut ar = empty_animation_raw::<4>(1.0);
        ar.timepoints = vec![0.0, 0.2, 0.4, 0.6, 1.0];
        ar.translations = vec![
            Float3Key::new([f16(-1.000000), 0, 0]),
            Float3Key::new([f16(0.000000), 0, 0]),
            Float3Key::new([f16(2.000000), 0, 0]),
            Float3Key::new([f16(7.000000), 0, 0]),
            Float3Key::new([f16(-1.000000), 0, 0]),
            Float3Key::new([f16(0.000000), 0, 0]),
            Float3Key::new([f16(6.000000), 0, 0]),
            Float3Key::new([f16(7.000000), 0, 0]),
            Float3Key::new([f16(8.000000), 0, 0]),
            Float3Key::new([f16(9.000000), 0, 0]),
            Float3Key::new([f16(10.000000), 0, 0]),
            Float3Key::new([f16(11.000000), 0, 0]),
            Float3Key::new([f16(9.000000), 0, 0]),
        ];
        ar.t_ratios = vec![0, 0, 0, 0, 4, 4, 1, 1, 2, 3, 3, 4, 4];
        ar.t_previouses = vec![0, 0, 0, 0, 4, 4, 4, 4, 2, 2, 2, 1, 3];
        execute_test::<4>(
            ar,
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
                frame(0.9999999, -1.0, 0.0, 11.0, 9.0),
                frame(0.0000001, -1.0, 0.0, 2.000002, 7.0),
            ],
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling_no_track() {
        execute_test::<0>(empty_animation_raw::<0>(46.0), vec![]);
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_sampling_1_track_0_key() {
        let mut ar = empty_animation_raw::<1>(46.0);
        ar.timepoints = vec![0.0, 1.0];
        execute_test::<1>(
            ar,
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
        let mut ar = empty_animation_raw::<1>(46.0);
        ar.timepoints = vec![0.0, 1.0];
        ar.translations = empty_translations();
        ar.translations[0] = Float3Key::new([f16(1.0), f16(-1.0), f16(5.0)]);
        ar.translations[4] = Float3Key::new([f16(1.0), f16(-1.0), f16(5.0)]);

        execute_test::<1>(
            ar,
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
            Frame {
                ratio,
                transform: [(v, QU, V1), (V0, QU, V1)],
            }
        }

        let mut ar = empty_animation_raw::<2>(46.0);
        ar.timepoints = vec![0.0, 0.5 / 46.0, 0.8 / 46.0, 1.0];
        ar.t_ratios = vec![0, 0, 0, 0, 1, 3, 3, 3, 2, 3];
        ar.t_previouses = vec![0, 0, 0, 0, 4, 4, 4, 4, 4, 1];
        ar.r_ratios = empty_ratios(3);
        ar.s_ratios = empty_ratios(3);
        ar.translations = vec![
            Float3Key::new([f16(1.0), f16(2.0), f16(4.0)]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(1.0), f16(2.0), f16(4.0)]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(2.0), f16(4.0), f16(8.0)]),
            Float3Key::new([f16(2.0), f16(4.0), f16(8.0)]),
        ];

        execute_test::<2>(
            ar,
            vec![
                // forward
                frame(0.0, Vec3::new(1.0, 2.0, 4.0)),
                frame(0.5 / 46.0, Vec3::new(1.0, 2.0, 4.0)),
                frame(0.65 / 46.0, Vec3::new(1.5, 3.0, 6.0)),
                frame(0.8 / 46.0, Vec3::new(2.0, 4.0, 8.0)),
                frame(1.0, Vec3::new(2.0, 4.0, 8.0)),
                // backward
                frame(0.8 / 46.0, Vec3::new(2.0, 4.0, 8.0)),
                frame(0.65 / 46.0, Vec3::new(1.5, 3.0, 6.0)),
                frame(0.5 / 46.0, Vec3::new(1.0, 2.0, 4.0)),
                frame(0.0, Vec3::new(1.0, 2.0, 4.0)),
            ],
        );
    }

    #[test]
    #[wasm_bindgen_test]
    #[rustfmt::skip]
    fn test_sampling_4_track_2_key() {
        let mut ar = empty_animation_raw::<4>(1.0);
        ar.timepoints=vec![0.0, 0.5, 0.8, 1.0];
        ar.t_ratios=vec![0, 0, 0, 0, 1, 3, 3, 3, 2, 3];
        ar.t_previouses=vec![0, 0, 0, 0, 4, 4, 4, 4, 4, 1];
        ar.r_ratios=empty_ratios(3);
        ar.r_previouses=empty_previouses();
        ar.s_ratios=vec![0, 0, 0, 0, 3, 3, 1, 3, 2, 3];
        ar.s_previouses=vec![0, 0, 0, 0, 4, 4, 4, 4, 2, 1];

        ar.translations = vec![
            Float3Key::new([f16(1.0), f16(2.0), f16(4.0)]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(-1.0), f16(-2.0), f16(-4.0)]),
            Float3Key::new([f16(1.0), f16(2.0), f16(4.0)]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(-2.0), f16(-4.0), f16(-8.0)]),
            Float3Key::new([f16(2.0), f16(4.0), f16(8.0)]),
            Float3Key::new([f16(2.0), f16(4.0), f16(8.0)]),
        ];

        ar.rotations = empty_rotations();
        ar.rotations[5] = QuaternionKey::new([65529, 65533, 32766]);

        ar.scales = vec![
            Float3Key::new([f16(1.0); 3]),
            Float3Key::new([f16(1.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(1.0); 3]),
            Float3Key::new([f16(1.0); 3]),
            Float3Key::new([f16(1.0); 3]),
            Float3Key::new([f16(0.0); 3]),
            Float3Key::new([f16(1.0); 3]),
            Float3Key::new([f16(-1.0); 3]),
            Float3Key::new([f16(-1.0); 3]),
        ];

        let mut frames_raw = vec![
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
        ];
        let mut frames = frames_raw.clone();
        frames_raw.reverse();
        frames.append(&mut frames_raw);

        execute_test::<4>(ar,
            frames,
        );
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_cache() {
        let mut translations = empty_translations();
        translations[0] = Float3Key::new([f16(1.0), f16(-1.0), f16(5.0)]);
        translations[4] = Float3Key::new([f16(1.0), f16(-1.0), f16(5.0)]);

        let animation_raw = AnimationRaw {
            duration: 46.0,
            num_tracks: 1,
            timepoints: vec![0.0, 1.0],
            translations,
            t_ratios: empty_ratios(1),
            t_previouses: empty_previouses(),
            rotations: empty_rotations(),
            r_ratios: empty_ratios(1),
            r_previouses: empty_previouses(),
            scales: empty_scales(),
            s_ratios: empty_ratios(1),
            s_previouses: empty_previouses(),
            ..Default::default()
        };
        let animation1 = Rc::new(Animation::from_raw(&animation_raw));
        let animation2 = Rc::new(Animation::from_raw(&animation_raw));

        let mut job = SamplingJob::default();
        job.set_animation(animation1.clone());
        job.set_context(SamplingContext::new(animation1.num_tracks()));

        fn run_test(job: &mut SamplingJob) -> Result<(), OzzError> {
            let output = make_buf(vec![TX; 1]);
            job.set_output(output.clone());
            job.run()?;
            for item in output.buf().unwrap().iter() {
                assert_eq!(item.translation.vec3(0), Vec3::new(1.0, -1.0, 5.0));
                assert!(item
                    .rotation
                    .quat(0)
                    .abs_diff_eq(Quat::from_xyzw(0.0, 0.0, 0.0, 1.0), 5e-1));
                assert_eq!(item.scale.vec3(0), Vec3::new(1.0, 1.0, 1.0));
            }
            Ok(())
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

        let animation = Rc::new(Animation::from_path("./resource/playback/animation.ozz").unwrap());
        let aligned_tracks = animation.num_aligned_tracks();

        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(aligned_tracks));
        job.set_output(make_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        job.set_ratio(0.5);
        job.run().unwrap();

        let ctx: SamplingContext = job.context().unwrap().clone();
        let bytes = rkyv::to_bytes::<_, 4096>(&ctx).unwrap();
        let archived = rkyv::check_archived_root::<SamplingContext>(&bytes[..]).unwrap();
        assert_eq!(archived.animation_id, ctx.animation_id());
        assert_eq!(archived.ratio, ctx.ratio());
        assert_eq!(&archived.translations, ctx.translations());
        assert_eq!(&archived.translation_entries, ctx.translation_entries());
        assert_eq!(&archived.translation_outdated, ctx.translation_outdated());
        assert_eq!(archived.translation_next as usize, ctx.translation_next());
        assert_eq!(&archived.rotations, ctx.rotations());
        assert_eq!(&archived.rotation_entries, ctx.rotation_entries());
        assert_eq!(&archived.rotation_outdated, ctx.rotation_outdated());
        assert_eq!(archived.rotation_next as usize, ctx.rotation_next());
        assert_eq!(&archived.scales, ctx.scales());
        assert_eq!(&archived.scale_entries, ctx.scale_entries());
        assert_eq!(&archived.scale_outdated, ctx.scale_outdated());
        assert_eq!(archived.scale_next as usize, ctx.scale_next());

        let ctx_de: SamplingContext = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(ctx_de.size(), ctx.size());
        assert_eq!(ctx_de.animation_id(), ctx.animation_id());
        assert_eq!(ctx_de.ratio(), ctx.ratio());
        assert_eq!(ctx_de.translations(), ctx.translations());
        assert_eq!(ctx_de.translation_entries(), ctx.translation_entries());
        assert_eq!(ctx_de.translation_outdated(), ctx.translation_outdated());
        assert_eq!(ctx_de.translation_next(), ctx.translation_next());
        assert_eq!(ctx_de.rotations(), ctx.rotations());
        assert_eq!(ctx_de.rotation_entries(), ctx.rotation_entries());
        assert_eq!(ctx_de.rotation_outdated(), ctx.rotation_outdated());
        assert_eq!(ctx_de.rotation_next(), ctx.rotation_next());
        assert_eq!(ctx_de.scale_entries(), ctx.scale_entries());
        assert_eq!(ctx_de.scale_outdated(), ctx.scale_outdated());
        assert_eq!(ctx_de.scale_next(), ctx.scale_next());
        assert_eq!(ctx_de.scales(), ctx.scales());
    }

    #[cfg(feature = "serde")]
    #[test]
    #[wasm_bindgen_test]
    fn test_serde() {
        let animation = Rc::new(Animation::from_path("./resource/playback/animation.ozz").unwrap());
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
        assert_eq!(ctx_de.animation_id(), ctx.animation_id());
        assert_eq!(ctx_de.ratio(), ctx.ratio());
        assert_eq!(ctx_de.translation_entries(), ctx.translation_entries());
        assert_eq!(ctx_de.rotation_entries(), ctx.rotation_entries());
        assert_eq!(ctx_de.scale_entries(), ctx.scale_entries());
        assert_eq!(ctx_de.translation_outdated(), ctx.translation_outdated());
        assert_eq!(ctx_de.rotation_outdated(), ctx.rotation_outdated());
        assert_eq!(ctx_de.scale_outdated(), ctx.scale_outdated());
        assert_eq!(ctx_de.translation_next(), ctx.translation_next());
        assert_eq!(ctx_de.rotation_next(), ctx.rotation_next());
        assert_eq!(ctx_de.scale_next(), ctx.scale_next());
        assert_eq!(ctx_de.translations(), ctx.translations());
        assert_eq!(ctx_de.rotations(), ctx.rotations());
        assert_eq!(ctx_de.scales(), ctx.scales());
    }
}
