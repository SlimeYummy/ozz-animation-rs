use std::{cell::RefCell, rc::Rc};

pub const SKELETON_MAX_JOINTS: i32 = 1024;
pub const SKELETON_NO_PARENT: i32 = -1;

pub type OzzRes<T> = Rc<T>;
pub type OzzResX<T> = Option<Rc<T>>;
pub type OzzBuf<T> = Rc<RefCell<Vec<T>>>;
pub type OzzBufX<T> = Option<Rc<RefCell<Vec<T>>>>;

pub fn ozz_res<T>(r: T) -> OzzRes<T> {
    return Rc::new(r);
}

pub fn ozz_res_x<T>(r: T) -> OzzResX<T> {
    return Some(Rc::new(r));
}

pub fn ozz_buf<T>(v: Vec<T>) -> OzzBuf<T> {
    return Rc::new(RefCell::new(v));
}

pub fn ozz_buf_x<T>(v: Vec<T>) -> OzzBufX<T> {
    return Some(Rc::new(RefCell::new(v)));
}
