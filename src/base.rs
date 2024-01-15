use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::BuildHasher;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::{Arc, RwLock};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OzzError {
    #[error("Lock poisoned")]
    LockPoison,
    #[error("Invalid job")]
    InvalidJob,

    #[error("IO error")]
    IO(#[from] std::io::Error),
    #[error("Utf8 error")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("Invalid tag")]
    InvalidTag,
    #[error("Invalid version")]
    InvalidVersion,
}

pub const SKELETON_MAX_JOINTS: i32 = 1024;
pub const SKELETON_NO_PARENT: i32 = -1;

pub trait OzzRef<T>
where
    T: ?Sized,
    Self: Clone + AsRef<T> + Deref<Target = T>,
{
}

impl<T: ?Sized> OzzRef<T> for Rc<T> {}
impl<T: ?Sized> OzzRef<T> for Arc<T> {}

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

#[inline(always)]
pub fn ozz_rc<T>(r: T) -> Rc<T> {
    return Rc::new(r);
}

#[inline(always)]
pub fn ozz_arc<T>(r: T) -> Arc<T> {
    return Arc::new(r);
}

#[inline(always)]
pub fn ozz_buf<T>(v: Vec<T>) -> Rc<RefCell<Vec<T>>> {
    return Rc::new(RefCell::new(v));
}

#[inline(always)]
pub fn ozz_abuf<T>(v: Vec<T>) -> Arc<RwLock<Vec<T>>> {
    return Arc::new(RwLock::new(v));
}

// A hasher builder that creates `DefaultHasher` with default keys.
#[derive(Default)]
pub struct DeterministicState;

impl DeterministicState {
    // Creates a new `DeterministicState` that builds `DefaultHasher` with default keys.
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
