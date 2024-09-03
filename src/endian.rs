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
        let bytes: [u8; 2] = num.to_ne_bytes();
        if bytes[0] == 0 {
            Endian::Big
        } else {
            Endian::Little
        }
    }

    pub fn from_tag(tag: u8) -> Endian {
        if tag == 0 {
            Endian::Big
        } else {
            Endian::Little
        }
    }
}

impl SwapEndian for u8 {
    fn swap_endian(self) -> u8 {
        self.swap_bytes()
    }
}

impl SwapEndian for i8 {
    fn swap_endian(self) -> i8 {
        self.swap_bytes()
    }
}

impl SwapEndian for bool {
    fn swap_endian(self) -> bool {
        self
    }
}

impl SwapEndian for u16 {
    fn swap_endian(self) -> u16 {
        self.swap_bytes()
    }
}

impl SwapEndian for i16 {
    fn swap_endian(self) -> i16 {
        self.swap_bytes()
    }
}

impl SwapEndian for u32 {
    fn swap_endian(self) -> u32 {
        self.swap_bytes()
    }
}

impl SwapEndian for i32 {
    fn swap_endian(self) -> i32 {
        self.swap_bytes()
    }
}

impl SwapEndian for f32 {
    fn swap_endian(self) -> f32 {
        f32::from_bits(self.to_bits().swap_bytes())
    }
}

impl SwapEndian for f64 {
    fn swap_endian(self) -> f64 {
        f64::from_bits(self.to_bits().swap_bytes())
    }
}

impl SwapEndian for u64 {
    fn swap_endian(self) -> u64 {
        self.swap_bytes()
    }
}

impl SwapEndian for i64 {
    fn swap_endian(self) -> i64 {
        self.swap_bytes()
    }
}

impl<'t, T: Copy + SwapEndian> SwapEndian for &'t mut [T] {
    fn swap_endian(self) -> &'t mut [T] {
        self.iter_mut().for_each(|e| *e = e.swap_endian());
        self
    }
}

impl<T: Copy + SwapEndian> SwapEndian for Vec<T> {
    fn swap_endian(mut self) -> Vec<T> {
        self.iter_mut().for_each(|e| *e = e.swap_endian());
        self
    }
}
