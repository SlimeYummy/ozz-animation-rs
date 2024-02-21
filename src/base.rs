use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasher;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use thiserror::Error;

/// Ozz error type.
#[derive(Error, Debug)]
pub enum OzzError {
    /// Lock poisoned, only happens when using `Arc<RWLock<T>>` as `OzzBuf<T>`.
    #[error("Lock poisoned")]
    LockPoison,
    /// Validates job failed.
    #[error("Invalid job")]
    InvalidJob,

    /// Std io errors.
    #[error("IO error")]
    IO(#[from] std::io::Error),
    /// Std string errors.
    #[error("Utf8 error")]
    Utf8(#[from] std::string::FromUtf8Error),

    /// Read ozz archive tag error.
    #[error("Invalid tag")]
    InvalidTag,
    /// Read ozz archive version error.
    #[error("Invalid version")]
    InvalidVersion,
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

/// Represents a reference to the ozz resource object.
/// `T` usually is `Skeleton` or `Animation`.
///
/// We use `OzzRef` to support `Rc<T>` (single thread) and `Arc<T>` (multithread) at same time.
/// Or you can implement this trait to support your own reference type.
pub trait OzzRef<T>
where
    T: ?Sized,
    Self: Clone + AsRef<T> + Deref<Target = T>,
{
}

impl<T: ?Sized> OzzRef<T> for Rc<T> {}
impl<T: ?Sized> OzzRef<T> for Arc<T> {}

/// Represents a reference to the ozz shared buffers.
/// `T` usually is `SoaTransform`, `Mat4`, .etc.
///
/// We use `OzzBuf` to support `Rc<RefCell<Vec<T>>>` (single thread) and `Arc<RwLock<Vec<T>>>` at same time.
/// Or you can implement this trait to support your own shared buffer types.
pub trait OzzBuf<T>
where
    Self: Clone,
{
    fn vec(&self) -> Result<impl Deref<Target = Vec<T>>, OzzError>;
    fn vec_mut(&self) -> Result<impl DerefMut<Target = Vec<T>>, OzzError>;
}

impl<T> OzzBuf<T> for Rc<RefCell<Vec<T>>> {
    fn vec(&self) -> Result<impl Deref<Target = Vec<T>>, OzzError> {
        return Ok(self.borrow());
    }

    fn vec_mut(&self) -> Result<impl DerefMut<Target = Vec<T>>, OzzError> {
        return Ok(self.borrow_mut());
    }
}

impl<T> OzzBuf<T> for Arc<RwLock<Vec<T>>> {
    fn vec(&self) -> Result<impl Deref<Target = Vec<T>>, OzzError> {
        return self.read().map_err(|_| OzzError::LockPoison);
    }

    fn vec_mut(&self) -> Result<impl DerefMut<Target = Vec<T>>, OzzError> {
        return self.write().map_err(|_| OzzError::LockPoison);
    }
}

/// Creates a new `Rc<T>`.
#[inline(always)]
pub fn ozz_rc<T>(r: T) -> Rc<T> {
    return Rc::new(r);
}

/// Creates a new `Arc<T>`.
#[inline(always)]
pub fn ozz_arc<T>(r: T) -> Arc<T> {
    return Arc::new(r);
}

/// Creates a new `Rc<RefCell<Vec<T>>>`.
#[inline(always)]
pub fn ozz_buf<T>(v: Vec<T>) -> Rc<RefCell<Vec<T>>> {
    return Rc::new(RefCell::new(v));
}

/// Creates a new `Arc<RwLock<Vec<T>>>`.
#[inline(always)]
pub fn ozz_abuf<T>(v: Vec<T>) -> Arc<RwLock<Vec<T>>> {
    return Arc::new(RwLock::new(v));
}

/// A hasher builder that creates `DefaultHasher` with default keys.
#[derive(Default)]
pub struct DeterministicState;

impl DeterministicState {
    /// Creates a new `DeterministicState` that builds `DefaultHasher` with default keys.
    pub fn new() -> DeterministicState {
        return DeterministicState;
    }
}

impl BuildHasher for DeterministicState {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> DefaultHasher {
        return DefaultHasher::default();
    }
}
