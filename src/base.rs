//!
//! Base types, traits and utils.
//!

use std::cell::{Ref, RefCell, RefMut};
use std::collections::hash_map::DefaultHasher;
use std::fmt::Debug;
use std::hash::BuildHasher;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use thiserror::Error;

/// Ozz error type.
#[derive(Error, Debug)]
pub enum OzzError {
    /// The buffer() is already borrowed or locked.
    ///
    /// Happens when using `Rc<RefCell<T>>` (already borrowed) or
    /// `Arc<RwLock<T>>` (currently locked) as an `OzzBuf<T>` / `OzzMutBuf<T>`.
    #[error("Buffer already borrowed")]
    BorrowError,
    /// Lock poisoned, only happens when using `Arc<RwLock<T>>` as `OzzBuf<T>`.
    #[error("Lock poisoned")]
    LockPoison,

    /// Validates job failed.
    #[error("Invalid job")]
    InvalidJob,
    /// Invalid buffer index.
    #[error("Invalid index")]
    InvalidIndex,

    /// Std io errors.
    #[error("IO error: {0}")]
    IO(std::io::ErrorKind),
    /// Std string errors.
    #[error("Utf8 error: valid_up_to {0}")]
    Utf8(u32),

    /// Read ozz archive tag error.
    #[error("Invalid tag")]
    InvalidTag,
    /// Read ozz archive version error.
    #[error("Invalid version")]
    InvalidVersion,

    /// Resource data failed validation (native `.ozz` archive, rkyv or serde).
    #[error("Invalid resource")]
    InvalidResource,

    /// Invalid repr.
    #[error("Invalid repr")]
    InvalidRepr,

    /// Unexpected error.
    ///
    /// The variant name is kept for backward compatibility, it is misspelled.
    #[error("Unexpected error")]
    Unexpected,
}

impl From<std::io::Error> for OzzError {
    fn from(err: std::io::Error) -> Self {
        OzzError::IO(err.kind())
    }
}

impl From<std::str::Utf8Error> for OzzError {
    fn from(err: std::str::Utf8Error) -> Self {
        OzzError::Utf8(err.valid_up_to() as u32)
    }
}

impl OzzError {
    pub fn is_lock_poison(&self) -> bool {
        matches!(self, OzzError::LockPoison)
    }

    pub fn is_borrow_error(&self) -> bool {
        matches!(self, OzzError::BorrowError)
    }

    pub fn is_invalid_job(&self) -> bool {
        matches!(self, OzzError::InvalidJob)
    }

    pub fn is_io(&self) -> bool {
        matches!(self, OzzError::IO(_))
    }

    pub fn is_utf8(&self) -> bool {
        matches!(self, OzzError::Utf8(_))
    }

    pub fn is_invalid_tag(&self) -> bool {
        matches!(self, OzzError::InvalidTag)
    }

    pub fn is_invalid_version(&self) -> bool {
        matches!(self, OzzError::InvalidVersion)
    }

    pub fn is_invalid_resource(&self) -> bool {
        matches!(self, OzzError::InvalidResource)
    }

    pub fn is_invalid_repr(&self) -> bool {
        matches!(self, OzzError::InvalidRepr)
    }

    pub fn is_unexcepted(&self) -> bool {
        matches!(self, OzzError::Unexpected)
    }
}

/// Defines the maximum number of joints.
/// This is limited in order to control the number of bits required to store
/// a joint index. Limiting the number of joints also helps handling worst
/// size cases, like when it is required to allocate an array of joints on
/// the stack.
pub const SKELETON_MAX_JOINTS: i32 = 1024;

/// Defines the maximum number of SoA elements required to store the maximum
/// number of joints.
pub const SKELETON_MAX_SOA_JOINTS: i32 = (SKELETON_MAX_JOINTS + 3) / 4;

/// Defines the index of the parent of the root joint (which has no parent in fact)
pub const SKELETON_NO_PARENT: i32 = -1;

/// A hasher builder that creates `DefaultHasher` with default keys.
#[derive(Debug, Default, Clone, Copy)]
pub struct DeterministicState;

impl DeterministicState {
    /// Creates a new `DeterministicState` that builds `DefaultHasher` with default keys.
    pub const fn new() -> DeterministicState {
        DeterministicState
    }
}

impl BuildHasher for DeterministicState {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        DefaultHasher::default()
    }
}

/// Allow usize/i32/i16 use as ozz index.
pub trait OzzIndex {
    fn usize(&self) -> usize;
    fn i32(&self) -> i32;
}

macro_rules! ozz_index {
    ($type:ty) => {
        impl OzzIndex for $type {
            #[inline(always)]
            fn usize(&self) -> usize {
                *self as usize
            }

            #[inline(always)]
            fn i32(&self) -> i32 {
                *self as i32
            }
        }
    };
}

ozz_index!(usize);
ozz_index!(i32);
ozz_index!(i16);

#[inline(always)]
pub(crate) fn align_usize(size: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (size + align - 1) & !(align - 1)
}

#[inline(always)]
pub(crate) fn align_ptr(ptr: *mut u8, align: usize) -> *mut u8 {
    debug_assert!(align.is_power_of_two());
    align_usize(ptr as usize, align) as *mut u8
}

/// Represents a reference to the ozz resource object.
/// `T` usually is `Skeleton` or `Animation`.
///
/// We use `OzzObj` to support `T`, `&T`, `Rc<T>` and `Arc<T>` at same time.
/// Or you can implement this trait to support your own reference type.
pub trait OzzObj<T: Debug> {
    fn obj(&self) -> &T;
}

impl<T: Debug> OzzObj<T> for T {
    #[inline(always)]
    fn obj(&self) -> &T {
        self
    }
}

impl<T: Debug> OzzObj<T> for &T {
    #[inline(always)]
    fn obj(&self) -> &T {
        self
    }
}

impl<T: Debug> OzzObj<T> for Rc<T> {
    #[inline(always)]
    fn obj(&self) -> &T {
        self.as_ref()
    }
}

impl<T: Debug> OzzObj<T> for Arc<T> {
    #[inline(always)]
    fn obj(&self) -> &T {
        self.as_ref()
    }
}

/// Represents a reference to the ozz immutable buffers.
/// `T` usually is `SoaTransform`, `Mat4`, .etc.
///
/// We use `OzzBuf` to support `&[T]`, `Vec<T>`, `Rc<RefCell<Vec<T>>>`, `Arc<RwLock<Vec<T>>>` at same time.
/// Or you can implement this trait to support your own immutable buffer types.
pub trait OzzBuf<T: Debug + Clone> {
    type Buf<'t>: Deref<Target = [T]>
    where
        Self: 't;

    fn buf(&self) -> Result<Self::Buf<'_>, OzzError>;
}

/// Represents a reference to the ozz mutable buffers.
/// `T` usually is `SoaTransform`, `Mat4`, .etc.
///
/// We use `OzzMutBuf` to support `&mut [T]`, `Vec<T>`, `Rc<RefCell<Vec<T>>>`, `Arc<RwLock<Vec<T>>>` at same time.
/// Or you can implement this trait to support your own writable buffer types.
///
/// See [`OzzBuf`] for the input/output aliasing caveat.
pub trait OzzMutBuf<T: Debug + Clone>
where
    Self: OzzBuf<T>,
{
    type MutBuf<'t>: DerefMut<Target = [T]>
    where
        Self: 't;

    fn mut_buf(&mut self) -> Result<Self::MutBuf<'_>, OzzError>;
}

//
// &[T]
//

impl<'a, T: 'static + Debug + Clone> OzzBuf<T> for &'a [T] {
    type Buf<'b>
        = ObSliceRef<'b, T>
    where
        'a: 'b;

    #[inline(always)]
    fn buf(&self) -> Result<ObSliceRef<'_, T>, OzzError> {
        Ok(ObSliceRef(self))
    }
}

pub struct ObSliceRef<'t, T>(pub &'t [T]);

impl<T> Deref for ObSliceRef<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

//
// &mut [T]
//

impl<'a, T: 'static + Debug + Clone> OzzBuf<T> for &'a mut [T] {
    type Buf<'b>
        = ObSliceRef<'b, T>
    where
        'a: 'b;

    #[inline(always)]
    fn buf(&self) -> Result<ObSliceRef<'_, T>, OzzError> {
        Ok(ObSliceRef(self))
    }
}

impl<'a, T: 'static + Debug + Clone> OzzMutBuf<T> for &'a mut [T] {
    type MutBuf<'b>
        = ObSliceRefMut<'b, T>
    where
        'a: 'b;

    #[inline(always)]
    fn mut_buf(&mut self) -> Result<ObSliceRefMut<'_, T>, OzzError> {
        Ok(ObSliceRefMut(self))
    }
}

pub struct ObSliceRefMut<'t, T>(pub &'t mut [T]);

impl<T> Deref for ObSliceRefMut<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<T> DerefMut for ObSliceRefMut<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

//
// Vec<T>
//

impl<T: 'static + Debug + Clone> OzzBuf<T> for Vec<T> {
    type Buf<'t> = ObSliceRef<'t, T>;

    #[inline(always)]
    fn buf(&self) -> Result<ObSliceRef<'_, T>, OzzError> {
        Ok(ObSliceRef(self.as_slice()))
    }
}

impl<T: 'static + Debug + Clone> OzzMutBuf<T> for Vec<T> {
    type MutBuf<'t> = ObSliceRefMut<'t, T>;

    #[inline(always)]
    fn mut_buf(&mut self) -> Result<ObSliceRefMut<'_, T>, OzzError> {
        Ok(ObSliceRefMut(self))
    }
}

//
// Rc<RefCell<Vec<T>>>
//

impl<T: 'static + Debug + Clone> OzzBuf<T> for Rc<RefCell<Vec<T>>> {
    type Buf<'t> = ObCellRef<'t, T>;

    #[inline(always)]
    fn buf(&self) -> Result<ObCellRef<'_, T>, OzzError> {
        match self.try_borrow() {
            Ok(guard) => Ok(ObCellRef(guard)),
            Err(_) => Err(OzzError::BorrowError),
        }
    }
}

pub struct ObCellRef<'t, T>(pub Ref<'t, Vec<T>>);

impl<T> Deref for ObCellRef<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl<T: 'static + Debug + Clone> OzzMutBuf<T> for Rc<RefCell<Vec<T>>> {
    type MutBuf<'t> = ObCellRefMut<'t, T>;

    #[inline(always)]
    fn mut_buf(&mut self) -> Result<ObCellRefMut<'_, T>, OzzError> {
        match self.try_borrow_mut() {
            Ok(guard) => Ok(ObCellRefMut(guard)),
            Err(_) => Err(OzzError::BorrowError),
        }
    }
}

pub struct ObCellRefMut<'t, T>(pub RefMut<'t, Vec<T>>);

impl<T> Deref for ObCellRefMut<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl<T> DerefMut for ObCellRefMut<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

//
// Arc<RwLock<Vec<T>>>
//

impl<T: 'static + Debug + Clone> OzzBuf<T> for Arc<RwLock<Vec<T>>> {
    type Buf<'t> = ObRwLockReadGuard<'t, T>;

    #[inline(always)]
    fn buf(&self) -> Result<ObRwLockReadGuard<'_, T>, OzzError> {
        match self.try_read() {
            Ok(guard) => Ok(ObRwLockReadGuard(guard)),
            Err(TryLockError::Poisoned(_)) => Err(OzzError::LockPoison),
            Err(TryLockError::WouldBlock) => Err(OzzError::BorrowError),
        }
    }
}

pub struct ObRwLockReadGuard<'t, T>(pub RwLockReadGuard<'t, Vec<T>>);

impl<T> Deref for ObRwLockReadGuard<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl<T: 'static + Debug + Clone> OzzMutBuf<T> for Arc<RwLock<Vec<T>>> {
    type MutBuf<'t> = ObRwLockWriteGuard<'t, T>;

    #[inline(always)]
    fn mut_buf(&mut self) -> Result<ObRwLockWriteGuard<'_, T>, OzzError> {
        match self.try_write() {
            Ok(guard) => Ok(ObRwLockWriteGuard(guard)),
            Err(TryLockError::Poisoned(_)) => Err(OzzError::LockPoison),
            Err(TryLockError::WouldBlock) => Err(OzzError::BorrowError),
        }
    }
}

pub struct ObRwLockWriteGuard<'t, T>(pub RwLockWriteGuard<'t, Vec<T>>);

impl<T> Deref for ObRwLockWriteGuard<'_, T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0.as_slice()
    }
}

impl<T> DerefMut for ObRwLockWriteGuard<'_, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut_slice()
    }
}

/// Shortcuts for `Rc<RefCell<T>>`.
pub type OzzRcBuf<T> = Rc<RefCell<Vec<T>>>;

/// Creates a new `Rc<RefCell<Vec<T>>>`.
#[inline]
pub fn ozz_rc_buf<T>(v: Vec<T>) -> OzzRcBuf<T> {
    Rc::new(RefCell::new(v))
}

/// Shortcuts for `Arc<RwLock<T>>`.
pub type OzzArcBuf<T> = Arc<RwLock<Vec<T>>>;

/// Creates a new `Arc<RwLock<Vec<T>>>`.
#[inline]
pub fn ozz_arc_buf<T>(v: Vec<T>) -> OzzArcBuf<T> {
    Arc::new(RwLock::new(v))
}

#[cfg(feature = "rkyv")]
pub(crate) trait SliceRkyvExt<T, D>
where
    T: rkyv::Archive,
    T::Archived: rkyv::Deserialize<T, D>,
    D: rkyv::rancor::Fallible + ?Sized,
    D::Error: rkyv::rancor::Source,
{
    fn copy_from_deserialize(
        &mut self,
        deserializer: &mut D,
        avec: &rkyv::vec::ArchivedVec<T::Archived>,
    ) -> Result<(), D::Error>;
}

#[cfg(feature = "rkyv")]
impl<T, D> SliceRkyvExt<T, D> for [T]
where
    T: rkyv::Archive,
    T::Archived: rkyv::Deserialize<T, D>,
    D: rkyv::rancor::Fallible + ?Sized,
    D::Error: rkyv::rancor::Source,
{
    #[inline(always)]
    fn copy_from_deserialize(
        &mut self,
        deserializer: &mut D,
        avec: &rkyv::vec::ArchivedVec<T::Archived>,
    ) -> Result<(), D::Error> {
        use rkyv::rancor::fail;
        use rkyv::Deserialize;
        // Archived data can be corrupted or hand-crafted: writing it into a slice
        // with a different length used to panic (or silently keep stale values for
        // shorter vecs), it must be rejected instead.
        if avec.len() != self.len() {
            fail!(OzzError::InvalidResource);
        }
        for (i, item) in avec.iter().enumerate() {
            self[i] = item.deserialize(deserializer)?;
        }
        Ok(())
    }
}
