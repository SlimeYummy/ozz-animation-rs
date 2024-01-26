use std::alloc::{self, Layout};
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
use std::simd::prelude::*;
use std::{mem, slice};

use crate::animation::{Animation, Float3Key, QuaternionKey};
use crate::base::{OzzBuf, OzzError, OzzRef};
use crate::math::{f32_clamp_or_max, SoaQuat, SoaTransform, SoaVec3};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct InterpSoaFloat3 {
    pub ratio: [f32x4; 2],
    pub value: [SoaVec3; 2],
}

#[cfg(feature = "rkyv")]
impl rkyv::Archive for InterpSoaFloat3 {
    type Archived = InterpSoaFloat3;
    type Resolver = ();

    #[inline]
    unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
        out.write(rkyv::to_archived!(*self as Self));
    }
}

#[cfg(feature = "rkyv")]
impl<S: rkyv::Fallible + ?Sized> rkyv::Serialize<S> for InterpSoaFloat3 {
    #[inline]
    fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
        return Ok(());
    }
}

#[cfg(feature = "rkyv")]
impl<D: rkyv::Fallible + ?Sized> rkyv::Deserialize<InterpSoaFloat3, D> for InterpSoaFloat3 {
    #[inline]
    fn deserialize(&self, _: &mut D) -> Result<InterpSoaFloat3, D::Error> {
        return Ok(rkyv::from_archived!(*self));
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct InterpSoaQuaternion {
    pub ratio: [f32x4; 2],
    pub value: [SoaQuat; 2],
}

#[cfg(feature = "rkyv")]
impl rkyv::Archive for InterpSoaQuaternion {
    type Archived = InterpSoaQuaternion;
    type Resolver = ();

    #[inline]
    unsafe fn resolve(&self, _: usize, _: Self::Resolver, out: *mut Self::Archived) {
        out.write(rkyv::to_archived!(*self as Self));
    }
}

#[cfg(feature = "rkyv")]
impl<S: rkyv::Fallible + ?Sized> rkyv::Serialize<S> for InterpSoaQuaternion {
    #[inline]
    fn serialize(&self, _: &mut S) -> Result<Self::Resolver, S::Error> {
        return Ok(());
    }
}

#[cfg(feature = "rkyv")]
impl<D: rkyv::Fallible + ?Sized> rkyv::Deserialize<InterpSoaQuaternion, D> for InterpSoaQuaternion {
    #[inline]
    fn deserialize(&self, _: &mut D) -> Result<InterpSoaQuaternion, D::Error> {
        return Ok(rkyv::from_archived!(*self));
    }
}

#[repr(align(16))]
#[derive(Debug)]
struct SamplingContextInner {
    size: usize,
    max_tracks: usize,
    max_soa_tracks: usize,
    num_outdated: usize,

    animation_id: u64,
    ratio: f32,

    translations_ptr: *mut InterpSoaFloat3,
    rotations_ptr: *mut InterpSoaQuaternion,
    scales_ptr: *mut InterpSoaFloat3,

    translation_keys_ptr: *mut i32,
    rotation_keys_ptr: *mut i32,
    scale_keys_ptr: *mut i32,

    translation_cursor: usize,
    rotation_cursor: usize,
    scale_cursor: usize,

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

            animation_id: 0,
            ratio: 0.0,

            translations_ptr: std::ptr::null_mut(),
            rotations_ptr: std::ptr::null_mut(),
            scales_ptr: std::ptr::null_mut(),

            translation_keys_ptr: std::ptr::null_mut(),
            rotation_keys_ptr: std::ptr::null_mut(),
            scale_keys_ptr: std::ptr::null_mut(),

            translation_cursor: 0,
            rotation_cursor: 0,
            scale_cursor: 0,

            outdated_translations_ptr: std::ptr::null_mut(),
            outdated_rotations_ptr: std::ptr::null_mut(),
            outdated_scales_ptr: std::ptr::null_mut(),
        };
    }
}

pub struct SamplingContext(*mut SamplingContextInner);

impl Debug for SamplingContext {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        return self.inner().fmt(f);
    }
}

impl Clone for SamplingContext {
    fn clone(&self) -> Self {
        let mut ctx = SamplingContext::new(self.max_tracks());
        ctx.set_animation_id(self.animation_id());
        ctx.set_ratio(self.ratio());

        ctx.translations_mut().copy_from_slice(self.translations());
        ctx.rotations_mut().copy_from_slice(self.rotations());
        ctx.scales_mut().copy_from_slice(self.scales());

        ctx.translation_keys_mut().copy_from_slice(self.translation_keys());
        ctx.rotation_keys_mut().copy_from_slice(self.rotation_keys());
        ctx.scale_keys_mut().copy_from_slice(self.scale_keys());

        ctx.set_translation_cursor(self.translation_cursor());
        ctx.set_rotation_cursor(self.rotation_cursor());
        ctx.set_scale_cursor(self.scale_cursor());

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
            && self.animation_id() == other.animation_id()
            && self.ratio() == other.ratio()
            && self.translations() == other.translations()
            && self.rotations() == other.rotations()
            && self.scales() == other.scales()
            && self.translation_keys() == other.translation_keys()
            && self.rotation_keys() == other.rotation_keys()
            && self.scale_keys() == other.scale_keys()
            && self.translation_cursor() == other.translation_cursor()
            && self.rotation_cursor() == other.rotation_cursor()
            && self.scale_cursor() == other.scale_cursor()
            && self.outdated_translations() == other.outdated_translations()
            && self.outdated_rotations() == other.outdated_rotations()
            && self.outdated_scales() == other.outdated_scales();
    }
}

impl SamplingContext {
    #[inline(always)]
    fn inner(&self) -> &SamplingContextInner {
        return unsafe { &*self.0 };
    }

    #[inline(always)]
    fn inner_mut(&mut self) -> &mut SamplingContextInner {
        return unsafe { &mut *self.0 };
    }

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
            let mut ctx = SamplingContext(ptr as *mut SamplingContextInner);
            *ctx.0 = SamplingContextInner::default();
            ptr = ptr.add(mem::size_of::<SamplingContextInner>());

            ctx.inner_mut().size = size;
            ctx.inner_mut().max_soa_tracks = max_soa_tracks;
            ctx.inner_mut().max_tracks = max_tracks;
            ctx.inner_mut().num_outdated = num_outdated;

            ctx.inner_mut().translations_ptr = ptr as *mut InterpSoaFloat3;
            ptr = ptr.add(mem::size_of::<InterpSoaFloat3>() * ctx.inner_mut().max_soa_tracks);
            ctx.inner_mut().rotations_ptr = ptr as *mut InterpSoaQuaternion;
            ptr = ptr.add(mem::size_of::<InterpSoaQuaternion>() * ctx.inner_mut().max_soa_tracks);
            ctx.inner_mut().scales_ptr = ptr as *mut InterpSoaFloat3;
            ptr = ptr.add(mem::size_of::<InterpSoaFloat3>() * ctx.inner_mut().max_soa_tracks);
            ctx.inner_mut().translation_keys_ptr = ptr as *mut i32;
            ptr = ptr.add(mem::size_of::<i32>() * max_tracks * 2);
            ctx.inner_mut().rotation_keys_ptr = ptr as *mut i32;
            ptr = ptr.add(mem::size_of::<i32>() * max_tracks * 2);
            ctx.inner_mut().scale_keys_ptr = ptr as *mut i32;
            ptr = ptr.add(mem::size_of::<i32>() * max_tracks * 2);
            ctx.inner_mut().outdated_translations_ptr = ptr as *mut u8;
            ptr = ptr.add(mem::size_of::<u8>() * ctx.inner_mut().num_outdated);
            ctx.inner_mut().outdated_rotations_ptr = ptr as *mut u8;
            ptr = ptr.add(mem::size_of::<u8>() * ctx.inner_mut().num_outdated);
            ctx.inner_mut().outdated_scales_ptr = ptr as *mut u8;
            ptr = ptr.add(mem::size_of::<u8>() * ctx.inner_mut().num_outdated);
            assert_eq!(ptr, (ctx.0 as *mut u8).add(size));

            return ctx;
        };
    }

    pub fn from_animation(animation: &Animation) -> SamplingContext {
        let mut ctx = SamplingContext::new(animation.num_tracks());
        ctx.inner_mut().animation_id = animation as *const _ as u64;
        return ctx;
    }

    pub fn clear(&mut self) {
        self.inner_mut().animation_id = 0;
        self.inner_mut().translation_cursor = 0;
        self.inner_mut().rotation_cursor = 0;
        self.inner_mut().scale_cursor = 0;
    }

    pub fn clone_without_animation_id(&self) -> SamplingContext {
        let mut ctx = self.clone();
        ctx.set_animation_id(0);
        return ctx;
    }

    pub fn size(&self) -> usize {
        return self.inner().size;
    }

    pub fn max_soa_tracks(&self) -> usize {
        return self.inner().max_soa_tracks;
    }

    pub fn max_tracks(&self) -> usize {
        return self.inner().max_tracks;
    }

    pub fn num_outdated(&self) -> usize {
        return self.inner().num_outdated;
    }

    pub fn animation_id(&self) -> u64 {
        return self.inner().animation_id;
    }

    pub fn set_animation_id(&mut self, id: u64) {
        self.inner_mut().animation_id = id;
    }

    pub fn ratio(&self) -> f32 {
        return self.inner().ratio;
    }

    fn set_ratio(&mut self, ratio: f32) {
        self.inner_mut().ratio = ratio;
    }

    pub fn translations(&self) -> &[InterpSoaFloat3] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.translations_ptr, inner.max_soa_tracks) };
    }

    fn translations_mut(&mut self) -> &mut [InterpSoaFloat3] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.translations_ptr, inner.max_soa_tracks) };
    }

    pub fn rotations(&self) -> &[InterpSoaQuaternion] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.rotations_ptr, inner.max_soa_tracks) };
    }

    fn rotations_mut(&mut self) -> &mut [InterpSoaQuaternion] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.rotations_ptr, inner.max_soa_tracks) };
    }

    pub fn scales(&self) -> &[InterpSoaFloat3] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.scales_ptr, inner.max_soa_tracks) };
    }

    fn scales_mut(&mut self) -> &mut [InterpSoaFloat3] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.scales_ptr, inner.max_soa_tracks) };
    }

    pub fn translation_keys(&self) -> &[i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.translation_keys_ptr, inner.max_tracks * 2) };
    }

    fn translation_keys_mut(&mut self) -> &mut [i32] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.translation_keys_ptr, inner.max_tracks * 2) };
    }

    pub fn rotation_keys(&self) -> &[i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.rotation_keys_ptr, inner.max_tracks * 2) };
    }

    fn rotation_keys_mut(&mut self) -> &mut [i32] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.rotation_keys_ptr, inner.max_tracks * 2) };
    }

    pub fn scale_keys(&self) -> &[i32] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.scale_keys_ptr, inner.max_tracks * 2) };
    }

    fn scale_keys_mut(&mut self) -> &mut [i32] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.scale_keys_ptr, inner.max_tracks * 2) };
    }

    pub fn translation_cursor(&self) -> usize {
        return self.inner().translation_cursor;
    }

    fn set_translation_cursor(&mut self, cursor: usize) {
        self.inner_mut().translation_cursor = cursor;
    }

    pub fn rotation_cursor(&self) -> usize {
        return self.inner().rotation_cursor;
    }

    fn set_rotation_cursor(&mut self, cursor: usize) {
        self.inner_mut().rotation_cursor = cursor;
    }

    pub fn scale_cursor(&self) -> usize {
        return self.inner().scale_cursor;
    }

    fn set_scale_cursor(&mut self, cursor: usize) {
        self.inner_mut().scale_cursor = cursor;
    }

    pub fn outdated_translations(&self) -> &[u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.outdated_translations_ptr, inner.num_outdated) };
    }

    fn outdated_translations_mut(&mut self) -> &mut [u8] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.outdated_translations_ptr, inner.num_outdated) };
    }

    pub fn outdated_rotations(&self) -> &[u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.outdated_rotations_ptr, inner.num_outdated) };
    }

    fn outdated_rotations_mut(&mut self) -> &mut [u8] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.outdated_rotations_ptr, inner.num_outdated) };
    }

    pub fn outdated_scales(&self) -> &[u8] {
        let inner = self.inner();
        return unsafe { std::slice::from_raw_parts(inner.outdated_scales_ptr, inner.num_outdated) };
    }

    fn outdated_scales_mut(&mut self) -> &mut [u8] {
        let inner = self.inner_mut();
        return unsafe { std::slice::from_raw_parts_mut(inner.outdated_scales_ptr, inner.num_outdated) };
    }
}

#[cfg(feature = "rkyv")]
pub struct ArchivedSamplingContext {
    pub size: u32,
    pub max_tracks: u32,
    pub max_soa_tracks: u32,
    pub num_outdated: u32,

    pub animation_id: u64,
    pub ratio: f32,

    translations_ptr: rkyv::RelPtr<InterpSoaFloat3>,
    rotations_ptr: rkyv::RelPtr<InterpSoaQuaternion>,
    scales_ptr: rkyv::RelPtr<InterpSoaFloat3>,

    translation_keys_ptr: rkyv::RelPtr<i32>,
    rotation_keys_ptr: rkyv::RelPtr<i32>,
    scale_keys_ptr: rkyv::RelPtr<i32>,

    pub translation_cursor: u32,
    pub rotation_cursor: u32,
    pub scale_cursor: u32,

    outdated_translations_ptr: rkyv::RelPtr<u8>,
    outdated_rotations_ptr: rkyv::RelPtr<u8>,
    outdated_scales_ptr: rkyv::RelPtr<u8>,
}

#[cfg(feature = "rkyv")]
impl ArchivedSamplingContext {
    pub fn translations(&self) -> &[InterpSoaFloat3] {
        return unsafe { slice::from_raw_parts(self.translations_ptr.as_ptr(), self.max_soa_tracks as usize) };
    }

    pub fn rotations(&self) -> &[InterpSoaQuaternion] {
        return unsafe { slice::from_raw_parts(self.rotations_ptr.as_ptr(), self.max_soa_tracks as usize) };
    }

    pub fn scales(&self) -> &[InterpSoaFloat3] {
        return unsafe { slice::from_raw_parts(self.scales_ptr.as_ptr(), self.max_soa_tracks as usize) };
    }

    pub fn translation_keys(&self) -> &[i32] {
        return unsafe { slice::from_raw_parts(self.translation_keys_ptr.as_ptr(), self.max_tracks as usize * 2) };
    }

    pub fn rotation_keys(&self) -> &[i32] {
        return unsafe { slice::from_raw_parts(self.rotation_keys_ptr.as_ptr(), self.max_tracks as usize * 2) };
    }

    pub fn scale_keys(&self) -> &[i32] {
        return unsafe { slice::from_raw_parts(self.scale_keys_ptr.as_ptr(), self.max_tracks as usize * 2) };
    }

    pub fn outdated_translations(&self) -> &[u8] {
        return unsafe { slice::from_raw_parts(self.outdated_translations_ptr.as_ptr(), self.num_outdated as usize) };
    }

    pub fn outdated_rotations(&self) -> &[u8] {
        return unsafe { slice::from_raw_parts(self.outdated_rotations_ptr.as_ptr(), self.num_outdated as usize) };
    }

    pub fn outdated_scales(&self) -> &[u8] {
        return unsafe { slice::from_raw_parts(self.outdated_scales_ptr.as_ptr(), self.num_outdated as usize) };
    }
}

#[derive(Default)]
#[cfg(feature = "rkyv")]
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

#[cfg(feature = "rkyv")]
impl rkyv::Archive for SamplingContext {
    type Archived = ArchivedSamplingContext;
    type Resolver = SamplingContextResolver;

    unsafe fn resolve(&self, pos: usize, resolver: SamplingContextResolver, out: *mut ArchivedSamplingContext) {
        let (fp, fo) = rkyv::out_field!(out.size);
        usize::resolve(&self.size(), pos + fp, (), fo);
        let (fp, fo) = rkyv::out_field!(out.max_tracks);
        usize::resolve(&self.max_tracks(), pos + fp, (), fo);
        let (fp, fo) = rkyv::out_field!(out.max_soa_tracks);
        usize::resolve(&self.max_soa_tracks(), pos + fp, (), fo);
        let (fp, fo) = rkyv::out_field!(out.num_outdated);
        usize::resolve(&self.num_outdated(), pos + fp, (), fo);

        let (fp, fo) = rkyv::out_field!(out.animation_id);
        u64::resolve(&self.animation_id(), pos + fp, (), fo);
        let (fp, fo) = rkyv::out_field!(out.ratio);
        f32::resolve(&self.ratio(), pos + fp, (), fo);

        let (fp, fo) = rkyv::out_field!(out.translations_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.translations_pos, fo);
        let (fp, fo) = rkyv::out_field!(out.rotations_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.rotations_pos, fo);
        let (fp, fo) = rkyv::out_field!(out.scales_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.scales_pos, fo);

        let (fp, fo) = rkyv::out_field!(out.translation_keys_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.translation_keys_pos, fo);
        let (fp, fo) = rkyv::out_field!(out.rotation_keys_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.rotation_keys_pos, fo);
        let (fp, fo) = rkyv::out_field!(out.scale_keys_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.scale_keys_pos, fo);

        let (fp, fo) = rkyv::out_field!(out.translation_cursor);
        usize::resolve(&self.translation_cursor(), pos + fp, (), fo);
        let (fp, fo) = rkyv::out_field!(out.rotation_cursor);
        usize::resolve(&self.rotation_cursor(), pos + fp, (), fo);
        let (fp, fo) = rkyv::out_field!(out.scale_cursor);
        usize::resolve(&self.scale_cursor(), pos + fp, (), fo);

        let (fp, fo) = rkyv::out_field!(out.outdated_translations_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.outdated_translations_pos, fo);
        let (fp, fo) = rkyv::out_field!(out.outdated_rotations_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.outdated_rotations_pos, fo);
        let (fp, fo) = rkyv::out_field!(out.outdated_scales_ptr);
        rkyv::RelPtr::emplace(pos + fp, resolver.outdated_scales_pos, fo);
    }
}

impl<S: rkyv::ser::Serializer + ?Sized> rkyv::Serialize<S> for SamplingContext {
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

#[cfg(feature = "rkyv")]
impl<D: rkyv::Fallible + ?Sized> rkyv::Deserialize<SamplingContext, D> for ArchivedSamplingContext {
    #[inline]
    fn deserialize(&self, _: &mut D) -> Result<SamplingContext, D::Error> {
        let archived = rkyv::from_archived!(self);
        let mut context = SamplingContext::new(archived.max_tracks as usize);
        context.set_animation_id(archived.animation_id);
        context.set_ratio(archived.ratio);
        context.translations_mut().copy_from_slice(archived.translations());
        context.rotations_mut().copy_from_slice(archived.rotations());
        context.scales_mut().copy_from_slice(archived.scales());
        context
            .translation_keys_mut()
            .copy_from_slice(archived.translation_keys());
        context.rotation_keys_mut().copy_from_slice(archived.rotation_keys());
        context.scale_keys_mut().copy_from_slice(archived.scale_keys());
        context.set_translation_cursor(archived.translation_cursor as usize);
        context.set_rotation_cursor(archived.rotation_cursor as usize);
        context.set_scale_cursor(archived.scale_cursor as usize);
        context
            .outdated_translations_mut()
            .copy_from_slice(archived.outdated_translations());
        context
            .outdated_rotations_mut()
            .copy_from_slice(archived.outdated_rotations());
        context
            .outdated_scales_mut()
            .copy_from_slice(archived.outdated_scales());
        return Ok(context);
    }
}

#[cfg(feature = "rkyv")]
impl<C: ?Sized> bytecheck::CheckBytes<C> for ArchivedSamplingContext {
    type Error = std::io::Error;

    #[inline]
    unsafe fn check_bytes<'a>(value: *const Self, _: &mut C) -> Result<&'a Self, Self::Error> {
        if value as usize % mem::align_of::<ArchivedSamplingContext>() != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "must be aligned to 16 bytes",
            ));
        }
        return Ok(&*value);
    }
}

#[derive(Debug)]
pub struct SamplingJob<A = Rc<Animation>, O = Rc<RefCell<Vec<SoaTransform>>>>
where
    A: OzzRef<Animation>,
    O: OzzBuf<SoaTransform>,
{
    animation: Option<A>,
    output: Option<O>,
    verified: bool,
    ratio: f32,
    context: Option<SamplingContext>,
}

impl<A, O> Default for SamplingJob<A, O>
where
    A: OzzRef<Animation>,
    O: OzzBuf<SoaTransform>,
{
    fn default() -> SamplingJob<A, O> {
        return SamplingJob {
            animation: None,
            output: None,
            verified: false,
            ratio: 0.0,
            context: None,
        };
    }
}

impl<A, O> SamplingJob<A, O>
where
    A: OzzRef<Animation>,
    O: OzzBuf<SoaTransform>,
{
    pub fn animation(&self) -> Option<&A> {
        return self.animation.as_ref();
    }

    pub fn set_animation(&mut self, animation: A) {
        self.verified = false;
        self.animation = Some(animation);
    }

    pub fn clear_animation(&mut self) {
        self.verified = false;
        self.animation = None;
    }

    pub fn context(&self) -> Option<&SamplingContext> {
        return self.context.as_ref();
    }

    pub fn set_context(&mut self, ctx: SamplingContext) {
        self.verified = false;
        self.context = Some(ctx);
    }

    pub fn clear_context(&mut self) {
        self.verified = false;
        self.context = None;
    }

    pub fn output(&self) -> Option<&O> {
        return self.output.as_ref();
    }

    pub fn set_output(&mut self, output: O) {
        self.verified = false;
        self.output = Some(output);
    }

    pub fn clear_output(&mut self) {
        self.verified = false;
        self.output = None;
    }

    pub fn ratio(&self) -> f32 {
        return self.ratio;
    }

    pub fn set_ratio(&mut self, ratio: f32) {
        self.ratio = f32_clamp_or_max(ratio, 0.0f32, 1.0f32);
    }

    pub fn validate(&self) -> bool {
        let animation = match &self.animation {
            Some(animation) => animation,
            None => return false,
        };
        let ctx = match &self.context {
            Some(ctx) => ctx,
            None => return false,
        };

        if ctx.max_soa_tracks() < animation.num_soa_tracks() {
            return false;
        }

        let output = match self.output.as_ref() {
            Some(output) => match output.vec() {
                Ok(output) => output,
                Err(_) => return false,
            },
            None => return false,
        };
        if output.len() < animation.num_soa_tracks() {
            return false;
        }

        return true;
    }

    pub fn run(&mut self) -> Result<(), OzzError> {
        if !self.verified {
            if !self.validate() {
                return Err(OzzError::InvalidJob);
            }
            self.verified = true;
        }

        let animation = self.animation.as_ref().unwrap();
        if animation.num_soa_tracks() == 0 {
            return Ok(());
        }

        self.step_context();

        self.update_translation_cursor();
        self.update_translation_key_frames();

        self.update_rotation_cursor();
        self.update_rotation_key_frames();

        self.update_scale_cursor();
        self.update_scale_key_frames();

        self.interpolates()?;

        return Ok(());
    }

    fn step_context(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();

        let animation_id = animation as *const _ as u64;
        if (ctx.animation_id() != animation_id) || self.ratio < ctx.ratio() {
            ctx.set_animation_id(animation_id);
            ctx.set_translation_cursor(0);
            ctx.set_rotation_cursor(0);
            ctx.set_scale_cursor(0);
        }
        ctx.set_ratio(self.ratio);
    }

    fn update_translation_cursor(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();

        if ctx.translation_cursor() == 0 {
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

            ctx.set_translation_cursor(animation.num_aligned_tracks() * 2);
        }

        while ctx.translation_cursor() < animation.translations().len() {
            let track = animation.translations()[ctx.translation_cursor()].track as usize;
            let key_idx = ctx.translation_keys()[track * 2 + 1] as usize;
            let ratio = animation.translations()[key_idx].ratio;
            if ratio > self.ratio {
                break;
            }

            ctx.outdated_translations_mut()[track / 32] |= 1 << ((track & 0x1F) / 4);
            let base = (animation.translations()[ctx.translation_cursor()].track as usize) * 2;
            ctx.translation_keys_mut()[base] = ctx.translation_keys()[base + 1];
            ctx.translation_keys_mut()[base + 1] = ctx.translation_cursor() as i32;
            ctx.set_translation_cursor(ctx.translation_cursor() + 1);
        }
    }

    fn update_translation_key_frames(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();

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

    fn update_rotation_cursor(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();

        if ctx.rotation_cursor() == 0 {
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

            ctx.set_rotation_cursor(animation.num_aligned_tracks() * 2);
        }

        while ctx.rotation_cursor() < animation.rotations().len() {
            let track = animation.rotations()[ctx.rotation_cursor()].track() as usize;
            let key_idx = ctx.rotation_keys()[track * 2 + 1] as usize;
            let ratio = animation.rotations()[key_idx].ratio;
            if ratio > self.ratio {
                break;
            }

            ctx.outdated_rotations_mut()[track / 32] |= 1 << ((track & 0x1F) / 4);
            let base = (animation.rotations()[ctx.rotation_cursor()].track() as usize) * 2;
            ctx.rotation_keys_mut()[base] = ctx.rotation_keys()[base + 1];
            ctx.rotation_keys_mut()[base + 1] = ctx.rotation_cursor() as i32;
            ctx.set_rotation_cursor(ctx.rotation_cursor() + 1);
        }
    }

    fn update_rotation_key_frames(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();

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

    fn update_scale_cursor(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();

        if ctx.scale_cursor() == 0 {
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

            ctx.set_scale_cursor(animation.num_aligned_tracks() * 2);
        }

        while ctx.scale_cursor() < animation.scales().len() {
            let track = animation.scales()[ctx.scale_cursor()].track as usize;
            let key_idx = ctx.scale_keys()[track * 2 + 1] as usize;
            let ratio = animation.scales()[key_idx].ratio;
            if ratio > self.ratio {
                break;
            }

            ctx.outdated_scales_mut()[track / 32] |= 1 << ((track & 0x1F) / 4);
            let base = (animation.scales()[ctx.scale_cursor()].track as usize) * 2;
            ctx.scale_keys_mut()[base] = ctx.scale_keys()[base + 1];
            ctx.scale_keys_mut()[base + 1] = ctx.scale_cursor() as i32;
            ctx.set_scale_cursor(ctx.scale_cursor() + 1);
        }
    }

    fn update_scale_key_frames(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();

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

    fn interpolates(&mut self) -> Result<(), OzzError> {
        let animation = self.animation.as_ref().unwrap();
        let ctx = self.context.as_mut().unwrap();
        let mut output = self.output.as_mut().unwrap().vec_mut()?;

        let ratio4 = f32x4::splat(self.ratio);
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

    use super::*;
    use crate::animation::{Float3Key, QuaternionKey};
    use crate::archive::{ArchiveReader, IArchive};
    use crate::base::ozz_buf;
    use crate::test_utils::f16;

    #[test]
    fn test_validity() {
        let mut archive = IArchive::new("./resource/animation-blending-1.ozz").unwrap();
        let animation = Rc::new(Animation::read(&mut archive).unwrap());
        let aligned_tracks = animation.num_aligned_tracks();

        // invalid output
        let mut job: SamplingJob = SamplingJob::default();
        job.set_animation(animation.clone());
        assert!(!job.validate());

        // invalid animation
        let mut job: SamplingJob = SamplingJob::default();
        job.set_output(ozz_buf(vec![SoaTransform::default(); animation.num_soa_tracks() + 10]));
        assert!(!job.validate());

        // invalid cache size
        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(5));
        job.set_output(ozz_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        assert!(!job.validate());

        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(aligned_tracks));
        job.set_output(ozz_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        assert!(job.validate());
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
        let animation = Rc::new(Animation {
            duration,
            num_tracks: T,
            name: String::new(),
            translations,
            rotations,
            scales,
        });
        let mut job = SamplingJob::default();
        job.set_animation(animation);
        job.set_context(SamplingContext::new(T));
        let output = ozz_buf(vec![TX; T + 1]);

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

            assert_eq!(job.context().unwrap().ratio(), f32_clamp_or_max(frame.ratio, 0.0, 1.0));
        }
    }

    #[test]
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
    fn test_sampling_no_track() {
        execute_test::<0>(46.0, vec![], vec![], vec![], vec![]);
    }

    #[test]
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
    fn test_cache() {
        let mut translations = new_translations();
        translations[0] = Float3Key::new(0.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);
        translations[4] = Float3Key::new(1.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);

        let animation1 = Rc::new(Animation {
            duration: 46.0,
            num_tracks: 1,
            name: String::new(),
            translations: translations.clone(),
            rotations: new_rotations(),
            scales: new_scales(),
        });

        let animation2 = Rc::new(Animation {
            duration: 46.0,
            num_tracks: 1,
            name: String::new(),
            translations: translations.clone(),
            rotations: new_rotations(),
            scales: new_scales(),
        });

        let mut job = SamplingJob::default();
        job.set_animation(animation1.clone());
        job.set_context(SamplingContext::new(animation1.num_tracks()));

        fn run_test(job: &mut SamplingJob) -> Result<(), OzzError> {
            let output = ozz_buf(vec![TX; 1]);
            job.set_output(output.clone());
            job.run()?;
            for item in output.vec().unwrap().iter() {
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
    fn test_rkyv() {
        use rkyv::Deserialize;

        let mut archive = IArchive::new("./resource/animation-blending-1.ozz").unwrap();
        let animation = Rc::new(Animation::read(&mut archive).unwrap());
        let aligned_tracks = animation.num_aligned_tracks();

        let mut job = SamplingJob::default();
        job.set_animation(animation.clone());
        job.set_context(SamplingContext::new(aligned_tracks));
        job.set_output(ozz_buf(vec![SoaTransform::default(); animation.num_soa_tracks()]));
        job.set_ratio(0.5);
        job.run().unwrap();

        let ctx: SamplingContext = job.context().unwrap().clone();
        let bytes = rkyv::to_bytes::<_, 256>(&ctx).unwrap();
        let archived = rkyv::check_archived_root::<SamplingContext>(&bytes[..]).unwrap();
        assert_eq!(archived.size as usize, ctx.size());
        assert_eq!(archived.animation_id, ctx.animation_id());
        assert_eq!(archived.ratio, ctx.ratio());
        assert_eq!(archived.translation_cursor as usize, ctx.translation_cursor());
        assert_eq!(archived.rotation_cursor as usize, ctx.rotation_cursor());
        assert_eq!(archived.scale_cursor as usize, ctx.scale_cursor());
        assert_eq!(archived.translations(), ctx.translations());
        assert_eq!(archived.rotations(), ctx.rotations());
        assert_eq!(archived.scales(), ctx.scales());
        assert_eq!(archived.translation_keys(), ctx.translation_keys());
        assert_eq!(archived.rotation_keys(), ctx.rotation_keys());
        assert_eq!(archived.scale_keys(), ctx.scale_keys());
        assert_eq!(archived.outdated_translations(), ctx.outdated_translations());
        assert_eq!(archived.outdated_rotations(), ctx.outdated_rotations());
        assert_eq!(archived.outdated_scales(), ctx.outdated_scales());

        let ctx_de: SamplingContext = archived.deserialize(&mut rkyv::Infallible).unwrap();
        assert_eq!(ctx_de.size(), ctx.size());
        assert_eq!(ctx_de.animation_id(), ctx.animation_id());
        assert_eq!(ctx_de.ratio(), ctx.ratio());
        assert_eq!(ctx_de.translation_cursor(), ctx.translation_cursor());
        assert_eq!(ctx_de.rotation_cursor(), ctx.rotation_cursor());
        assert_eq!(ctx_de.scale_cursor(), ctx.scale_cursor());
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
