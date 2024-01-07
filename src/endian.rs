use std::mem;

pub trait SwapEndian {
    fn swap_endian(self) -> Self;
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
    fn swap_endian(self) -> u8 {
        return self.swap_bytes();
    }
}

impl SwapEndian for i8 {
    fn swap_endian(self) -> i8 {
        return self.swap_bytes();
    }
}

impl SwapEndian for bool {
    fn swap_endian(self) -> bool {
        return self;
    }
}

impl SwapEndian for u16 {
    fn swap_endian(self) -> u16 {
        return self.swap_bytes();
    }
}

impl SwapEndian for i16 {
    fn swap_endian(self) -> i16 {
        return self.swap_bytes();
    }
}

impl SwapEndian for u32 {
    fn swap_endian(self) -> u32 {
        return self.swap_bytes();
    }
}

impl SwapEndian for i32 {
    fn swap_endian(self) -> i32 {
        return self.swap_bytes();
    }
}

impl SwapEndian for f32 {
    fn swap_endian(self) -> f32 {
        let mut tmp: u32 = unsafe { mem::transmute(self) };
        tmp = tmp.swap_bytes();
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for f64 {
    fn swap_endian(self) -> f64 {
        let mut tmp: u64 = unsafe { mem::transmute(self) };
        tmp = tmp.swap_bytes();
        return unsafe { mem::transmute(tmp) };
    }
}

impl SwapEndian for u64 {
    fn swap_endian(self) -> u64 {
        return self.swap_bytes();
    }
}

impl SwapEndian for i64 {
    fn swap_endian(self) -> i64 {
        return self.swap_bytes();
    }
}

impl<'t, T: Copy + SwapEndian> SwapEndian for &'t mut [T] {
    fn swap_endian(self) -> &'t mut [T] {
        self.iter_mut().for_each(|e| *e = e.swap_endian());
        return self;
    }
}

impl<T: Copy + SwapEndian> SwapEndian for Vec<T> {
    fn swap_endian(mut self) -> Vec<T> {
        self.iter_mut().for_each(|e| *e = e.swap_endian());
        return self;
    }
}
