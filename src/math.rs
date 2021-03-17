use nalgebra::{ComplexField, Matrix4, Quaternion, RealField, Vector3, Vector4};
use std::mem;

pub trait OzzNumber
where
    Self: Default + ComplexField + RealField,
{
    fn convert_f16(n: u16) -> Self;
    fn convert_f32(n: f32) -> Self;
    fn convert_i16(n: i16) -> Self;
}

impl OzzNumber for f32 {
    #[inline]
    fn convert_f16(n: u16) -> Self {
        let sign = (n & 0x8000) as u32;
        let expo = (n & 0x7C00) as u32;
        let base = (n & 0x03FF) as u32;
        if expo == 0x7C00 {
            if base != 0 {
                return f32::NAN;
            }
            if sign == 0x8000 {
                return f32::NEG_INFINITY;
            } else {
                return f32::INFINITY;
            }
        }
        let expmant = (n & 0x7FFF) as u32;
        unsafe {
            let magic = mem::transmute::<u32, f32>((254 - 15) << 23);
            let shifted = mem::transmute::<u32, f32>(expmant << 13);
            let scaled = mem::transmute::<f32, u32>(shifted * magic);
            return mem::transmute::<u32, f32>((sign << 16) | scaled);
        };
    }

    #[inline(always)]
    fn convert_f32(n: f32) -> Self {
        return n;
    }

    #[inline(always)]
    fn convert_i16(n: i16) -> Self {
        return n as f32;
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct OzzTransform<N: OzzNumber> {
    pub translation: Vector3<N>,
    pub rotation: Quaternion<N>,
    pub scale: Vector3<N>,
}

impl<N: OzzNumber> OzzTransform<N> {
    pub fn convert_f32(t: &OzzTransform<f32>) -> OzzTransform<N> {
        return OzzTransform {
            translation: Vector3::new(
                OzzNumber::convert_f32(t.translation.x),
                OzzNumber::convert_f32(t.translation.y),
                OzzNumber::convert_f32(t.translation.z),
            ),
            rotation: Quaternion::new(
                OzzNumber::convert_f32(t.rotation.w),
                OzzNumber::convert_f32(t.rotation.i),
                OzzNumber::convert_f32(t.rotation.j),
                OzzNumber::convert_f32(t.rotation.k),
            ),
            scale: Vector3::new(
                OzzNumber::convert_f32(t.scale.x),
                OzzNumber::convert_f32(t.scale.y),
                OzzNumber::convert_f32(t.scale.z),
            ),
        };
    }
}

pub fn ozz_vec3_lerp<N: OzzNumber>(a: &Vector3<N>, b: &Vector3<N>, f: N) -> Vector3<N> {
    return Vector3::new(
        (b.x - a.x) * f + a.x,
        (b.y - a.y) * f + a.y,
        (b.z - a.z) * f + a.z,
    );
}

pub fn ozz_quat_nlerp<N: OzzNumber>(a: &Quaternion<N>, b: &Quaternion<N>, f: N) -> Quaternion<N> {
    let lerp = Quaternion::new(
        (b.w - a.w) * f + a.w,
        (b.i - a.i) * f + a.i,
        (b.j - a.j) * f + a.j,
        (b.k - a.k) * f + a.k,
    );
    let len2 = lerp.i * lerp.i + lerp.j * lerp.j + lerp.k * lerp.k + lerp.w * lerp.w;
    let inv_len = N::one() / len2.sqrt();
    return Quaternion::new(
        lerp.w * inv_len,
        lerp.i * inv_len,
        lerp.j * inv_len,
        lerp.k * inv_len,
    );
}

pub fn ozz_quat_dot<N: OzzNumber>(a: &Quaternion<N>, b: &Quaternion<N>) -> N {
    return a.i * b.i + a.j * b.j + a.k * b.k + a.w * b.w;
}

pub fn ozz_quat_normalize<N: OzzNumber>(q: &Quaternion<N>) -> Quaternion<N> {
    let len2 = q.i * q.i + q.j * q.j + q.k * q.k + q.w * q.w;
    let inv_len = N::one() / len2.sqrt();
    return Quaternion::new(q.w * inv_len, q.i * inv_len, q.j * inv_len, q.k * inv_len);
}

pub fn ozz_matrix4_new<N: OzzNumber>(
    translation: &Vector3<N>,
    rotation: &Quaternion<N>,
    scale: &Vector3<N>,
) -> Matrix4<N> {
    let xx = rotation.i * rotation.i;
    let xy = rotation.i * rotation.j;
    let xz = rotation.i * rotation.k;
    let xw = rotation.i * rotation.w;
    let yy = rotation.j * rotation.j;
    let yz = rotation.j * rotation.k;
    let yw = rotation.j * rotation.w;
    let zz = rotation.k * rotation.k;
    let zw = rotation.k * rotation.w;
    let zero = N::zero();
    let one = N::one();
    let two = N::one() + N::one();
    return Matrix4::new(
        scale.x * (one - two * (yy + zz)),
        scale.y * two * (xy - zw),
        scale.z * two * (xz + yw),
        translation.x,
        scale.x * two * (xy + zw),
        scale.y * (one - two * (xx + zz)),
        scale.z * two * (yz - xw),
        translation.y,
        scale.x * two * (xz - yw),
        scale.y * two * (yz + xw),
        scale.z * (one - two * (xx + yy)),
        translation.z,
        zero,
        zero,
        zero,
        one,
    );
}

pub fn ozz_matrix4_mul<N: OzzNumber>(a: &Matrix4<N>, b: &Matrix4<N>) -> Matrix4<N> {
    fn dot<N: OzzNumber>(r: &Vector4<N>, c: &Vector4<N>) -> N {
        return r[0] * c[0] + r[1] * c[1] + r[2] * c[2] + r[3] * c[3];
    }
    let ar0 = Vector4::new(a[(0, 0)], a[(0, 1)], a[(0, 2)], a[(0, 3)]);
    let ar1 = Vector4::new(a[(1, 0)], a[(1, 1)], a[(1, 2)], a[(1, 3)]);
    let ar2 = Vector4::new(a[(2, 0)], a[(2, 1)], a[(2, 2)], a[(2, 3)]);
    let ar3 = Vector4::new(a[(3, 0)], a[(3, 1)], a[(3, 2)], a[(3, 3)]);
    let bc0 = Vector4::new(b[(0, 0)], b[(1, 0)], b[(2, 0)], b[(3, 0)]);
    let bc1 = Vector4::new(b[(0, 1)], b[(1, 1)], b[(2, 1)], b[(3, 1)]);
    let bc2 = Vector4::new(b[(0, 2)], b[(1, 2)], b[(2, 2)], b[(3, 2)]);
    let bc3 = Vector4::new(b[(0, 3)], b[(1, 3)], b[(2, 3)], b[(3, 3)]);
    return Matrix4::new(
        dot(&ar0, &bc0),
        dot(&ar0, &bc1),
        dot(&ar0, &bc2),
        dot(&ar0, &bc3),
        dot(&ar1, &bc0),
        dot(&ar1, &bc1),
        dot(&ar1, &bc2),
        dot(&ar1, &bc3),
        dot(&ar2, &bc0),
        dot(&ar2, &bc1),
        dot(&ar2, &bc2),
        dot(&ar2, &bc3),
        dot(&ar3, &bc0),
        dot(&ar3, &bc1),
        dot(&ar3, &bc2),
        dot(&ar3, &bc3),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_half_to_f32() {
        assert_eq!(f32::convert_f16(0b00111100_00000000), 1.0f32);
        assert_eq!(f32::convert_f16(0b10111100_00000000), -1.0f32);
        assert_eq!(f32::convert_f16(0b01000011_00000000), 3.5f32);
        assert_eq!(f32::convert_f16(0b01111100_00000000), f32::INFINITY);
        assert_eq!(f32::convert_f16(0b11111100_00000000), f32::NEG_INFINITY);
        assert!(f32::convert_f16(0xFFFF).is_nan());
        assert_eq!(f32::convert_f16(32791), -1.37090683e-06);
    }
}
