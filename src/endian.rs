use std::mem;

pub trait SwapEndian {
    fn swap(self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Endian {
    Big,
    Little,
}

impl Endian {
    pub fn native() -> Endian {
        let num: u16 = 1;
        let bytes: [u8; 2] = unsafe { mem::transmute(num) };
        if bytes[0] == 0 {
            return Endian::Big;
        } else {
            return Endian::Little;
        }
    }

    pub fn from_tag(tag: u8) -> Endian {
        if tag == 0 {
            return Endian::Big;
        } else {
            return Endian::Little;
        }
    }
}

impl SwapEndian for u8 {
    fn swap(self) -> u8 {
        return self;
    }
}

impl SwapEndian for i8 {
    fn swap(self) -> i8 {
        return self;
    }
}

impl SwapEndian for bool {
    fn swap(self) -> bool {
        return self;
    }
}

impl SwapEndian for u16 {
    fn swap(self) -> u16 {
        let mut tmp: [u8; 2] = unsafe { mem::transmute(self) };
        tmp.swap(0, 1);
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for i16 {
    fn swap(self) -> i16 {
        let mut tmp: [u8; 2] = unsafe { mem::transmute(self) };
        tmp.swap(0, 1);
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for u32 {
    fn swap(self) -> u32 {
        let mut tmp: [u8; 4] = unsafe { mem::transmute(self) };
        tmp.swap(0, 3);
        tmp.swap(1, 2);
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for i32 {
    fn swap(self) -> i32 {
        let mut tmp: [u8; 4] = unsafe { mem::transmute(self) };
        tmp.swap(0, 3);
        tmp.swap(1, 2);
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for f32 {
    fn swap(self) -> f32 {
        let mut tmp: [u8; 4] = unsafe { mem::transmute(self) };
        tmp.swap(0, 3);
        tmp.swap(1, 2);
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for u64 {
    fn swap(self) -> u64 {
        let mut tmp: [u8; 8] = unsafe { mem::transmute(self) };
        tmp.swap(0, 7);
        tmp.swap(1, 6);
        tmp.swap(2, 5);
        tmp.swap(3, 4);
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for i64 {
    fn swap(self) -> i64 {
        let mut tmp: [u8; 8] = unsafe { mem::transmute(self) };
        tmp.swap(0, 7);
        tmp.swap(1, 6);
        tmp.swap(2, 5);
        tmp.swap(3, 4);
        return unsafe { mem::transmute(tmp) };
    }
}

impl<T: Copy + SwapEndian> SwapEndian for Vec<T> {
    fn swap(self) -> Vec<T> {
        let mut vec = self;
        vec.iter_mut().for_each(|e| *e = e.swap());
        return vec;
    }
}
