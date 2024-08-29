//!
//! Skinning Job.
//!

use glam::{Mat4, Vec3, Vec4};
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::base::{OzzBuf, OzzError, OzzMutBuf};

/// Skinning job.
///
/// TODO: Deterministic tests.
#[derive(Debug)]
pub struct SkinningJob<
    JM = Rc<RefCell<Vec<Mat4>>>,
    JI = Rc<RefCell<Vec<u16>>>,
    JW = Rc<RefCell<Vec<f32>>>,
    I = Rc<RefCell<Vec<Vec3>>>,
    O = Rc<RefCell<Vec<Vec3>>>,
> where
    JM: OzzBuf<Mat4>,
    JI: OzzBuf<u16>,
    JW: OzzBuf<f32>,
    I: OzzBuf<Vec3>,
    O: OzzMutBuf<Vec3>,
{
    vertex_count: usize,
    influences_count: usize,

    joint_matrices: Option<JM>,
    joint_it_matrices: Option<JM>,
    joint_indices: Option<JI>,
    joint_weights: Option<JW>,

    in_positions: Option<I>,
    in_normals: Option<I>,
    in_tangents: Option<I>,

    out_positions: Option<O>,
    out_normals: Option<O>,
    out_tangents: Option<O>,
}

pub type SkinningJobRef<'t> = SkinningJob<&'t [Mat4], &'t [u16], &'t [f32], &'t [Vec3], &'t mut [Vec3]>;
pub type SkinningJobRc = SkinningJob<
    Rc<RefCell<Vec<Mat4>>>,
    Rc<RefCell<Vec<u16>>>,
    Rc<RefCell<Vec<f32>>>,
    Rc<RefCell<Vec<Vec3>>>,
    Rc<RefCell<Vec<Vec3>>>,
>;
pub type SkinningJobArc = SkinningJob<
    Arc<RwLock<Vec<Mat4>>>,
    Arc<RwLock<Vec<u16>>>,
    Arc<RwLock<Vec<f32>>>,
    Arc<RwLock<Vec<Vec3>>>,
    Arc<RwLock<Vec<Vec3>>>,
>;

impl<JM, JI, JW, I, O> Default for SkinningJob<JM, JI, JW, I, O>
where
    JM: OzzBuf<Mat4>,
    JI: OzzBuf<u16>,
    JW: OzzBuf<f32>,
    I: OzzBuf<Vec3>,
    O: OzzMutBuf<Vec3>,
{
    fn default() -> Self {
        SkinningJob {
            vertex_count: 0,
            influences_count: 0,

            joint_matrices: None,
            joint_it_matrices: None,
            joint_indices: None,
            joint_weights: None,

            in_positions: None,
            in_normals: None,
            in_tangents: None,

            out_positions: None,
            out_normals: None,
            out_tangents: None,
        }
    }
}

impl<JM, JI, JW, I, O> SkinningJob<JM, JI, JW, I, O>
where
    JM: OzzBuf<Mat4>,
    JI: OzzBuf<u16>,
    JW: OzzBuf<f32>,
    I: OzzBuf<Vec3>,
    O: OzzMutBuf<Vec3>,
{
    /// Gets vertex count of `SkinningJob`.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Sets vertex count of `SkinningJob`.
    ///
    /// Number of vertices to transform. All input and output arrays must store at
    /// least this number of vertices.
    #[inline]
    pub fn set_vertex_count(&mut self, vertex_count: usize) {
        self.vertex_count = vertex_count;
    }

    /// Gets influences count of `SkinningJob`.
    #[inline]
    pub fn influences_count(&self) -> usize {
        self.influences_count
    }

    /// Sets influences count of `SkinningJob`.
    ///
    /// Maximum number of matrices influencing each vertex. Must be greater than 0.
    ///
    /// The number of influences drives how joint_indices and joint_weights are sampled:
    /// - influences_count joint indices are red from joint_indices for each vertex.
    /// - influences_count - 1 joint weights are red from joint_weightrs for each vertex.
    ///   The weight of the last joint is restored (weights are normalized).
    #[inline]
    pub fn set_influences_count(&mut self, influences_count: usize) {
        self.influences_count = influences_count;
    }

    /// Gets joint matrices of `SkinningJob`.
    #[inline]
    pub fn joint_matrices(&self) -> Option<&JM> {
        return self.joint_matrices.as_ref();
    }

    /// Sets joint matrices of `SkinningJob`.
    ///
    /// Array of matrices for each joint. Joint are indexed through indices array.
    #[inline]
    pub fn set_joint_matrices(&mut self, joint_matrices: JM) {
        self.joint_matrices = Some(joint_matrices);
    }

    /// Clears joint matrices of `SkinningJob`.
    #[inline]
    pub fn clear_joint_matrices(&mut self) {
        self.joint_matrices = None;
    }

    /// Gets joint inverse transpose matrices of `SkinningJob`.
    #[inline]
    pub fn joint_it_matrices(&self) -> Option<&JM> {
        return self.joint_it_matrices.as_ref();
    }

    /// Sets joint inverse transpose matrices of `SkinningJob`.
    ///
    /// Optional array of inverse transposed matrices for each joint. If provided, this array is used to
    /// transform vectors (normals and tangents), otherwise joint_matrices array is used.
    #[inline]
    pub fn set_joint_it_matrices(&mut self, joint_it_matrices: JM) {
        self.joint_it_matrices = Some(joint_it_matrices);
    }

    /// Clears joint inverse transpose matrices of `SkinningJob`.
    #[inline]
    pub fn clear_joint_it_matrices(&mut self) {
        self.joint_it_matrices = None;
    }

    /// Gets joint indices of `SkinningJob`.
    #[inline]
    pub fn joint_indices(&self) -> Option<&JI> {
        return self.joint_indices.as_ref();
    }

    /// Sets joint indices of `SkinningJob`.
    #[inline]
    pub fn set_joint_indices(&mut self, joint_indices: JI) {
        self.joint_indices = Some(joint_indices);
    }

    /// Clears joint indices of `SkinningJob`.
    #[inline]
    pub fn clear_joint_indices(&mut self) {
        self.joint_indices = None;
    }

    /// Gets joint weights of `SkinningJob`.
    #[inline]
    pub fn joint_weights(&self) -> Option<&JW> {
        return self.joint_weights.as_ref();
    }

    /// Sets joint weights of `SkinningJob`.
    ///
    /// Array of matrices weights. This array is used to associate a weight to every joint that influences
    /// a vertex. The number of weights required per vertex is "influences_max - 1". The weight for the
    /// last joint (for each vertex) is restored at runtime thanks to the fact that the sum of the weights
    /// for each vertex is 1.
    ///
    /// Each vertex has (influences_max - 1) number of weights, meaning that the size of this array must
    /// be at least (influences_max - 1)* `joint_matrices.len()`.
    #[inline]
    pub fn set_joint_weights(&mut self, joint_weights: JW) {
        self.joint_weights = Some(joint_weights);
    }

    /// Clears joint weights of `SkinningJob`.
    #[inline]
    pub fn clear_joint_weights(&mut self) {
        self.joint_weights = None;
    }

    /// Gets input positions of `SkinningJob`.
    #[inline]
    pub fn in_positions(&self) -> Option<&I> {
        return self.in_positions.as_ref();
    }

    /// Sets input positions of `SkinningJob`.
    ///
    /// Array length must be at least `joint_matrices.len()`.
    #[inline]
    pub fn set_in_positions(&mut self, in_positions: I) {
        self.in_positions = Some(in_positions);
    }

    /// Clears input positions of `SkinningJob`.
    #[inline]
    pub fn clear_in_positions(&mut self) {
        self.in_positions = None;
    }

    /// Gets input normals of `SkinningJob`.
    #[inline]
    pub fn in_normals(&self) -> Option<&I> {
        return self.in_normals.as_ref();
    }

    /// Sets input normals of `SkinningJob`.
    ///
    /// Array length must be at least `joint_matrices.len()`.
    #[inline]
    pub fn set_in_normals(&mut self, in_normals: I) {
        self.in_normals = Some(in_normals);
    }

    /// Clears input normals of `SkinningJob`.
    #[inline]
    pub fn clear_in_normals(&mut self) {
        self.in_normals = None;
    }

    /// Gets input tangents of `SkinningJob`.
    #[inline]
    pub fn in_tangents(&self) -> Option<&I> {
        return self.in_tangents.as_ref();
    }

    /// Sets input tangents of `SkinningJob`.
    ///
    /// Array length must be at least `joint_matrices.len()`.
    #[inline]
    pub fn set_in_tangents(&mut self, in_tangents: I) {
        self.in_tangents = Some(in_tangents);
    }

    /// Clears input tangents of `SkinningJob`.
    #[inline]
    pub fn clear_in_tangents(&mut self) {
        self.in_tangents = None;
    }

    /// Gets output positions of `SkinningJob`.
    #[inline]
    pub fn out_positions(&self) -> Option<&O> {
        return self.out_positions.as_ref();
    }

    /// Sets output positions of `SkinningJob`.
    ///
    /// Array length must be at least `joint_matrices.len()`.
    #[inline]
    pub fn set_out_positions(&mut self, out_positions: O) {
        self.out_positions = Some(out_positions);
    }

    /// Clears output positions of `SkinningJob`.
    #[inline]
    pub fn clear_out_positions(&mut self) {
        self.out_positions = None;
    }

    /// Gets output normals of `SkinningJob`.
    #[inline]
    pub fn out_normals(&self) -> Option<&O> {
        return self.out_normals.as_ref();
    }

    /// Sets output normals of `SkinningJob`.
    ///
    /// Array length must be at least `joint_matrices.len()`.
    #[inline]
    pub fn set_out_normals(&mut self, out_normals: O) {
        self.out_normals = Some(out_normals);
    }

    /// Clears output normals of `SkinningJob`.
    #[inline]
    pub fn clear_out_normals(&mut self) {
        self.out_normals = None;
    }

    /// Gets output tangents of `SkinningJob`.
    #[inline]
    pub fn out_tangents(&self) -> Option<&O> {
        return self.out_tangents.as_ref();
    }

    /// Sets output tangents of `SkinningJob`.
    ///
    /// Array length must be at least `joint_matrices.len()`.
    #[inline]
    pub fn set_out_tangents(&mut self, out_tangents: O) {
        self.out_tangents = Some(out_tangents);
    }

    /// Clears output tangents of `SkinningJob`.
    #[inline]
    pub fn clear_out_tangents(&mut self) {
        self.out_tangents = None;
    }

    /// Validates `SkinningJob` parameters.
    pub fn validate(&self) -> bool {
        (|| {
            let mut ok = self.influences_count > 0;
            ok &= !self.joint_matrices.as_ref()?.buf().ok()?.is_empty();

            let joint_indices = self.joint_indices.as_ref()?.buf().ok()?;
            ok &= joint_indices.len() >= self.vertex_count * self.influences_count;

            if self.influences_count > 1 {
                let joint_weights = self.joint_weights.as_ref()?.buf().ok()?;
                ok &= joint_weights.len() >= self.vertex_count * (self.influences_count - 1);
            }

            ok &= self.in_positions.as_ref()?.buf().ok()?.len() >= self.vertex_count;
            ok &= self.out_positions.as_ref()?.buf().ok()?.len() >= self.vertex_count;

            if let Some(in_normals) = &self.in_normals {
                ok &= in_normals.buf().ok()?.len() >= self.vertex_count;
                ok &= self.out_normals.as_ref()?.buf().ok()?.len() >= self.vertex_count;

                if let Some(in_tangents) = &self.in_tangents {
                    ok &= in_tangents.buf().ok()?.len() >= self.vertex_count;
                    ok &= self.out_tangents.as_ref()?.buf().ok()?.len() >= self.vertex_count;
                }
            } else {
                ok &= self.in_tangents.is_none();
            }

            Some(ok)
        })()
        .unwrap_or(false)
    }
}

fn unpack_buf<T, B>(ozz_buf: &Option<B>, size: usize) -> Result<B::Buf<'_>, OzzError>
where
    T: Debug + Clone,
    B: OzzBuf<T>,
{
    let buf = ozz_buf.as_ref().ok_or(OzzError::InvalidJob)?.buf()?;
    if buf.len() < size {
        return Err(OzzError::InvalidJob);
    }
    Ok(buf)
}

fn unpack_mut_buf<T, B>(ozz_buf: &mut Option<B>, size: usize) -> Result<B::MutBuf<'_>, OzzError>
where
    T: Debug + Clone,
    B: OzzMutBuf<T>,
{
    let buf = ozz_buf.as_mut().ok_or(OzzError::InvalidJob)?.mut_buf()?;
    if buf.len() < size {
        return Err(OzzError::InvalidJob);
    }
    Ok(buf)
}

#[rustfmt::skip]
macro_rules! it {
    (_, $s1:stmt) => {};
    (IT, $s1:stmt) => { $s1 };
    (_, $s1:expr, $s2:expr) => { $s2 };
    (IT, $s1:expr, $s2:expr) => { $s1 };
}

#[rustfmt::skip]
macro_rules! pnt {
    (P, $s1:stmt; $s2:stmt; $s3:stmt;) => { $s1 };
    (PN, $s1:stmt; $s2:stmt; $s3:stmt;) => { $s1 $s2 };
    (PNT, $s1:stmt; $s2:stmt; $s3:stmt;) => { $s1 $s2 $s3 };
}

macro_rules! skinning_1 {
    ($fn:ident, $it:tt, $pnt:tt) => {
        fn $fn(&mut self) -> Result<(), OzzError> {
            let matrices = self.joint_matrices.as_ref().ok_or(OzzError::InvalidJob)?.buf()?;
            it!($it, let it_matrices = self.joint_it_matrices.as_ref().ok_or(OzzError::InvalidJob)?.buf()?);
            let indices = unpack_buf(&self.joint_indices, self.vertex_count)?;

            pnt!($pnt,
                let in_positions = unpack_buf(&self.in_positions, self.vertex_count)?;
                let in_normals = unpack_buf(&self.in_normals, self.vertex_count)?;
                let in_tangents = unpack_buf(&self.in_tangents, self.vertex_count)?;
            );
            pnt!($pnt,
                let mut out_positions = unpack_mut_buf(&mut self.out_positions, self.vertex_count)?;
                let mut out_normals = unpack_mut_buf(&mut self.out_normals, self.vertex_count)?;
                let mut out_tangents = unpack_mut_buf(&mut self.out_tangents, self.vertex_count)?;
            );

            for i in 0..self.vertex_count {
                let joint_index = indices[i] as usize;
                let transform = &matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?;
                #[allow(unused_variables)]
                let transform_it = it!($it, &it_matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?, transform);

                pnt!($pnt,
                    out_positions[i] = transform.transform_point3(in_positions[i]);
                    out_normals[i] = transform_it.transform_vector3(in_normals[i]);
                    out_tangents[i] = transform_it.transform_vector3(in_tangents[i]);
                );
            }
            return Ok(());
        }
    };
}

macro_rules! skinning_impl {
    ($self:expr, $n:expr, $it:tt, $pnt:tt) => {
        let matrices = $self.joint_matrices.as_ref().unwrap().buf()?;
        it!($it, let it_matrices = $self.joint_it_matrices.as_ref().ok_or(OzzError::InvalidJob)?.buf()?);
        let indices = unpack_buf(&$self.joint_indices, $self.vertex_count * $n)?;
        let weights = unpack_buf(&$self.joint_weights, $self.vertex_count * ($n - 1))?;

        pnt!($pnt,
            let in_positions = unpack_buf(&$self.in_positions, $self.vertex_count)?;
            let in_normals = unpack_buf(&$self.in_normals, $self.vertex_count)?;
            let in_tangents = unpack_buf(&$self.in_tangents, $self.vertex_count)?;
        );
        pnt!($pnt,
            let mut out_positions = unpack_mut_buf(&mut $self.out_positions, $self.vertex_count)?;
            let mut out_normals = unpack_mut_buf(&mut $self.out_normals, $self.vertex_count)?;
            let mut out_tangents = unpack_mut_buf(&mut $self.out_tangents, $self.vertex_count)?;
        );

        for i in 0..$self.vertex_count {
            let weight_offset = i * ($n - 1);
            let index_offset = i * $n;

            let weight = Vec4::splat(weights[weight_offset]);
            let mut weight_sum = weight;
            let joint_index = indices[index_offset] as usize;
            let mut transform = mat4_col_mul(matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?, weight);
            it!($it, let mut transform_it = mat4_col_mul(it_matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?, weight));

            for j in 1..($n - 1).max(1) {
                let weight = Vec4::splat(weights[weight_offset + j]);
                weight_sum += weight;
                let joint_index = indices[index_offset + j] as usize;
                transform += mat4_col_mul(matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?, weight);
                it!($it, transform_it += mat4_col_mul(it_matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?, weight));
            }

            let weight = Vec4::ONE - weight_sum;
            let joint_index = indices[index_offset + $n - 1] as usize;
            transform += mat4_col_mul(matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?, weight);
            it!($it, transform_it += mat4_col_mul(it_matrices.get(joint_index).ok_or(OzzError::InvalidIndex)?, weight));

            pnt!($pnt,
                out_positions[i] = transform.transform_point3(in_positions[i]);
                out_normals[i] = it!($it, transform_it, transform).transform_vector3(in_normals[i]);
                out_tangents[i] = it!($it, transform_it, transform).transform_vector3(in_tangents[i]);
            );
        }
        return Ok(());
    };
}

#[inline(always)]
fn mat4_col_mul(m: &Mat4, v: Vec4) -> Mat4 {
    Mat4::from_cols(m.x_axis * v, m.y_axis * v, m.z_axis * v, m.w_axis * v)
}

macro_rules! skinning_c {
    ($fn:ident, $n:expr, $it:tt, $pnt:tt) => {
        fn $fn(&mut self) -> Result<(), OzzError> {
            skinning_impl!(self, $n, $it, $pnt);
        }
    };
}

macro_rules! skinning_n {
    ($fn:ident, $it:tt, $pnt:tt) => {
        fn $fn(&mut self) -> Result<(), OzzError> {
            let n = self.influences_count as usize;
            skinning_impl!(self, n, $it, $pnt);
        }
    };
}

impl<JM, JI, JW, I, O> SkinningJob<JM, JI, JW, I, O>
where
    JM: OzzBuf<Mat4>,
    JI: OzzBuf<u16>,
    JW: OzzBuf<f32>,
    I: OzzBuf<Vec3>,
    O: OzzMutBuf<Vec3>,
{
    /// Runs skinning job's task.
    /// The validate job before any operation is performed.
    pub fn run(&mut self) -> Result<(), OzzError> {
        if self.influences_count == 0 {
            return Err(OzzError::InvalidJob);
        }

        let mut branch = (self.influences_count - 1).min(4) * 5;
        if self.in_normals.is_some() {
            branch += 1;
            if self.in_tangents.is_some() {
                branch += 1;
            }
            if self.joint_it_matrices.is_some() {
                branch += 2;
            }
        }

        match branch {
            0 => self.skinning_1_p(),
            1 => self.skinning_1_pn(),
            2 => self.skinning_1_pnt(),
            3 => self.skinning_1_pn_it(),
            4 => self.skinning_1_pnt_it(),
            5 => self.skinning_2_p(),
            6 => self.skinning_2_pn(),
            7 => self.skinning_2_pnt(),
            8 => self.skinning_2_pn_it(),
            9 => self.skinning_2_pnt_it(),
            10 => self.skinning_3_p(),
            11 => self.skinning_3_pn(),
            12 => self.skinning_3_pnt(),
            13 => self.skinning_3_pn_it(),
            14 => self.skinning_3_pnt_it(),
            15 => self.skinning_4_p(),
            16 => self.skinning_4_pn(),
            17 => self.skinning_4_pnt(),
            18 => self.skinning_4_pn_it(),
            19 => self.skinning_4_pnt_it(),
            20 => self.skinning_n_p(),
            21 => self.skinning_n_pn(),
            22 => self.skinning_n_pnt(),
            23 => self.skinning_n_pn_it(),
            24 => self.skinning_n_pnt_it(),
            _ => unreachable!(),
        }
    }

    skinning_1!(skinning_1_p, _, P);
    skinning_1!(skinning_1_pn, _, PN);
    skinning_1!(skinning_1_pnt, _, PNT);
    skinning_1!(skinning_1_pn_it, IT, PN);
    skinning_1!(skinning_1_pnt_it, IT, PNT);
    skinning_c!(skinning_2_p, 2, _, P);
    skinning_c!(skinning_2_pn, 2, _, PN);
    skinning_c!(skinning_2_pnt, 2, _, PNT);
    skinning_c!(skinning_2_pn_it, 2, IT, PN);
    skinning_c!(skinning_2_pnt_it, 2, IT, PNT);
    skinning_c!(skinning_3_p, 3, _, P);
    skinning_c!(skinning_3_pn, 3, _, PN);
    skinning_c!(skinning_3_pnt, 3, _, PNT);
    skinning_c!(skinning_3_pn_it, 3, IT, PN);
    skinning_c!(skinning_3_pnt_it, 3, IT, PNT);
    skinning_c!(skinning_4_p, 4, _, P);
    skinning_c!(skinning_4_pn, 4, _, PN);
    skinning_c!(skinning_4_pnt, 4, _, PNT);
    skinning_c!(skinning_4_pn_it, 4, IT, PN);
    skinning_c!(skinning_4_pnt_it, 4, IT, PNT);
    skinning_n!(skinning_n_p, _, P);
    skinning_n!(skinning_n_pn, _, PN);
    skinning_n!(skinning_n_pnt, _, PNT);
    skinning_n!(skinning_n_pn_it, IT, PN);
    skinning_n!(skinning_n_pnt_it, IT, PNT);
}

#[cfg(test)]
mod skinning_tests {
    use wasm_bindgen_test::*;

    use super::*;

    #[test]
    #[wasm_bindgen_test]
    fn test_validate() {
        let matrices = Rc::new(RefCell::new(vec![Mat4::default(); 2]));
        let it_matrices = Rc::new(RefCell::new(vec![Mat4::default(); 2]));
        let joint_indices = Rc::new(RefCell::new(vec![0; 8]));
        let joint_weights = Rc::new(RefCell::new(vec![0.0f32; 6]));
        let in_positions = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));
        let in_normals = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));
        let in_tangents = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));
        let out_positions = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));
        let out_normals = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));
        let out_tangents = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));

        let mut job: SkinningJob = SkinningJob::default();
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // Valid job with 0 vertex.
        let mut job: SkinningJob = SkinningJob::default();
        job.set_vertex_count(0);
        job.set_influences_count(1);
        job.set_joint_matrices(matrices.clone());
        job.set_joint_indices(joint_indices.clone());
        job.set_in_positions(in_positions.clone());
        job.set_out_positions(out_positions.clone());
        assert!(job.validate());
        assert!(job.run().is_ok());

        // Invalid job with 0 influence.
        let mut job: SkinningJob = SkinningJob::default();
        job.set_vertex_count(0);
        job.set_influences_count(0);
        job.set_joint_matrices(matrices.clone());
        job.set_joint_indices(joint_indices.clone());
        job.set_in_positions(in_positions.clone());
        job.set_out_positions(out_positions.clone());
        assert!(!job.validate());
        assert!(job.run().unwrap_err().is_invalid_job());

        // Valid job with 1/2/3/4 influence.
        {
            let mut job: SkinningJob = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_joint_matrices(matrices.clone());
            job.set_joint_indices(joint_indices.clone());
            job.set_joint_weights(joint_weights.clone());
            job.set_in_positions(in_positions.clone());
            job.set_out_positions(out_positions.clone());

            job.set_influences_count(1);
            assert!(job.validate());
            assert!(job.run().is_ok());

            job.set_influences_count(2);
            assert!(job.validate());
            assert!(job.run().is_ok());

            job.set_influences_count(3);
            assert!(job.validate());
            assert!(job.run().is_ok());

            job.set_influences_count(4);
            assert!(job.validate());
            assert!(job.run().is_ok());

            job.set_joint_it_matrices(it_matrices.clone());
            job.set_influences_count(1);
            assert!(job.validate());
            assert!(job.run().is_ok());

            job.set_influences_count(2);
            assert!(job.validate());
            assert!(job.run().is_ok());

            job.set_in_normals(in_normals.clone());
            job.set_out_normals(out_normals.clone());
            assert!(job.validate());
            assert!(job.run().is_ok());

            job.set_in_tangents(in_tangents.clone());
            job.set_out_tangents(out_tangents.clone());
            assert!(job.validate());
            assert!(job.run().is_ok());
        }

        // Invalid job with 2 influences, missing xxx.
        {
            let mut job: SkinningJob = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_influences_count(2);
            job.set_joint_matrices(matrices.clone());
            job.set_joint_indices(joint_indices.clone());
            job.set_joint_weights(joint_weights.clone());
            job.set_in_positions(in_positions.clone());
            job.set_out_positions(out_positions.clone());

            // indices
            job.clear_joint_indices();
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
            job.set_joint_indices(joint_indices.clone());

            // weights
            job.clear_joint_weights();
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
            job.set_joint_weights(joint_weights.clone());

            // in positions
            job.clear_in_positions();
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
            job.set_in_positions(in_positions.clone());

            // out positions
            job.clear_out_positions();
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
            job.set_out_positions(out_positions.clone());

            job.set_in_normals(in_normals.clone());
            job.set_out_normals(out_normals.clone());

            // in normals
            job.clear_in_normals();
            assert!(job.validate());
            assert!(job.run().is_ok());
            job.set_in_normals(in_normals.clone());

            // out normals
            job.clear_out_normals();
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
            job.set_out_normals(out_normals.clone());

            job.set_in_tangents(in_tangents.clone());
            job.set_out_tangents(out_tangents.clone());

            // in tangents
            job.clear_in_tangents();
            assert!(job.validate());
            assert!(job.run().is_ok());
            job.set_in_tangents(in_tangents.clone());

            // out tangents
            job.clear_out_tangents();
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
        }

        // Invalid job with 2 influences, not enough xxx.
        {
            let mut job: SkinningJob = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_influences_count(2);
            job.set_joint_matrices(matrices.clone());
            job.set_joint_indices(joint_indices.clone());
            job.set_joint_weights(joint_weights.clone());
            job.set_in_positions(in_positions.clone());
            job.set_out_positions(out_positions.clone());

            job.set_joint_indices(Rc::new(RefCell::new(vec![0; 3])));
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
            job.set_joint_indices(joint_indices.clone());

            job.set_joint_weights(Rc::new(RefCell::new(vec![0.0; 1])));
            assert!(!job.validate());
            assert!(job.run().unwrap_err().is_invalid_job());
            job.set_joint_weights(joint_weights.clone());
        }
    }

    #[test]
    #[wasm_bindgen_test]
    fn test_run() {
        type SkinningJobTest<'t> = SkinningJob<&'t [Mat4], &'t [u16], &'t [f32], &'t [Vec3], Rc<RefCell<Vec<Vec3>>>>;

        let matrices: [Mat4; 4] = [
            Mat4::from_cols(
                Vec4::new(-1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, 1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, -1.0, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ),
            Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            Mat4::from_scale(Vec3::new(1.0, 2.0, 3.0)),
            Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        ];
        let it_matrices: [Mat4; 4] = [
            Mat4::from_cols(
                Vec4::new(1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, -1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, 1.0, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ),
            Mat4::IDENTITY,
            Mat4::from_cols(
                Vec4::new(-1.0, 0.0, 0.0, 0.0),
                Vec4::new(0.0, -1.0, 0.0, 0.0),
                Vec4::new(0.0, 0.0, -1.0, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ),
            Mat4::IDENTITY,
        ];

        let in_positions: [Vec3; 2] = [Vec3::new(1.0, 2.0, 3.0), Vec3::new(4.0, 5.0, 6.0)];
        let in_normals: [Vec3; 2] = [Vec3::new(0.1, 0.2, 0.3), Vec3::new(0.4, 0.5, 0.6)];
        let in_tangents: [Vec3; 2] = [Vec3::new(0.01, 0.02, 0.03), Vec3::new(0.04, 0.05, 0.06)];
        let out_positions = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));
        let out_normals = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));
        let out_tangents = Rc::new(RefCell::new(vec![Vec3::ZERO; 2]));

        {
            let mut job: SkinningJobTest = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_influences_count(1);
            job.set_joint_matrices(&matrices);
            job.set_joint_indices(&[0, 3]);
            job.set_in_positions(&in_positions);
            job.set_out_positions(out_positions.clone());

            // P1
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(-1.0, 2.0, -3.0), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(5.0, 7.0, 9.0), 1e-6));

            // PN1 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(-1.0, 2.0, -3.0), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.1, -0.2, 0.3), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(5.0, 7.0, 9.0), 1e-6));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.4, 0.5, 0.6), 1e-6));

            // PNT1
            job.clear_joint_it_matrices();
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(-1.0, 2.0, -3.0), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(-0.1, 0.2, -0.3), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(-0.01, 0.02, -0.03), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(5.0, 7.0, 9.0), 1e-6));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.4, 0.5, 0.6), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.04, 0.05, 0.06), 1e-6));

            // PNT1 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(-1.0, 2.0, -3.0), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.1, -0.2, 0.3), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.01, -0.02, 0.03), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(5.0, 7.0, 9.0), 1e-6));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.4, 0.5, 0.6), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.04, 0.05, 0.06), 1e-6));
        }

        {
            let mut job: SkinningJobTest = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_influences_count(2);
            job.set_joint_matrices(&matrices);
            job.set_joint_indices(&[0, 1, 3, 2]);
            job.set_joint_weights(&[0.5, 0.1]);
            job.set_in_positions(&in_positions);
            job.set_out_positions(out_positions.clone());

            // P2
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.5, 3.0, 1.5), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.1, 9.7, 17.1), 1e-5));

            // PN2
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.5, 3.0, 1.5), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.2, 0.0), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.1, 9.7, 17.1), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.4, 0.95, 1.68), 1e-6));

            // PN2 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.5, 3.0, 1.5), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.1, 0.0, 0.3), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.1, 9.7, 17.1), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(-0.32, -0.4, -0.48), 1e-6));

            // PNT2
            job.clear_joint_it_matrices();
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.5, 3.0, 1.5), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.2, 0.0), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.02, 0.0), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.1, 9.7, 17.1), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.4, 0.95, 1.68), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.04, 0.095, 0.168), 1e-6));

            // PNT2 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.5, 3.0, 1.5), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.1, 0.0, 0.3), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.01, 0.0, 0.03), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.1, 9.7, 17.1), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(-0.32, -0.4, -0.48), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(-0.032, -0.04, -0.048), 1e-6));
        }

        {
            let mut job: SkinningJobTest = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_influences_count(3);
            job.set_joint_matrices(&matrices);
            job.set_joint_indices(&[0, 1, 2, 3, 2, 1]);
            job.set_joint_weights(&[0.5, 0.25, 0.1, 0.25]);
            job.set_in_positions(&in_positions);
            job.set_out_positions(out_positions.clone());

            // P3
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.75, 7.75, 11.25), 1e-5));

            // PN3
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.25, 0.15), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.75, 7.75, 11.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.4, 0.625, 0.9), 1e-6));

            // PN3 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.05, -0.1, 0.15), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.75, 7.75, 11.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.2, 0.25, 0.3), 1e-6));

            // PNT3
            job.clear_joint_it_matrices();
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.25, 0.15), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.025, 0.015), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.75, 7.75, 11.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.4, 0.625, 0.9), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.04, 0.0625, 0.09), 1e-6));

            // PNT3 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.05, -0.1, 0.15), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.005, -0.01, 0.015), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(4.75, 7.75, 11.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.2, 0.25, 0.3), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.02, 0.025, 0.03), 1e-6));
        }

        {
            let mut job: SkinningJobTest = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_influences_count(4);
            job.set_joint_matrices(&matrices);
            job.set_joint_indices(&[0, 1, 2, 3, 3, 2, 1, 0]);
            job.set_joint_weights(&[0.5, 0.25, 0.25, 0.1, 0.25, 0.25]);
            job.set_in_positions(&in_positions);
            job.set_out_positions(out_positions.clone());

            // P4
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(1.15, 6.95, 5.25), 1e-5));

            // PN4
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.25, 0.15), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(1.15, 6.95, 5.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.08, 0.625, 0.42), 1e-6));

            // PN4 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.05, -0.1, 0.15), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(1.15, 6.95, 5.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.2, -0.15, 0.3), 1e-6));

            // PNT4
            job.clear_joint_it_matrices();
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.25, 0.15), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.0, 0.025, 0.015), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(1.15, 6.95, 5.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.08, 0.625, 0.42), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.008, 0.0625, 0.042), 1e-6));

            // PNT4 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.25, 3.0, 2.25), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.05, -0.1, 0.15), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.005, -0.01, 0.015), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(1.15, 6.95, 5.25), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.2, -0.15, 0.3), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.02, -0.015, 0.03), 1e-6));
        }

        {
            let mut job: SkinningJobTest = SkinningJob::default();
            job.set_vertex_count(2);
            job.set_influences_count(5);
            job.set_joint_matrices(&matrices);
            job.set_joint_indices(&[0, 1, 2, 3, 0, 3, 2, 1, 0, 3]);
            job.set_joint_weights(&[0.5, 0.25, 0.25, 0.1, 0.1, 0.25, 0.25, 0.15]);
            job.set_in_positions(&in_positions);
            job.set_out_positions(out_positions.clone());

            // P5
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.55, 3.2, 3.15), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(3.4, 7.45, 9.0), 1e-5));

            // PN5
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.55, 3.2, 3.15), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.02, 0.25, 0.21), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(3.4, 7.45, 9.0), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.28, 0.625, 0.72), 1e-6));

            // PN5 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.55, 3.2, 3.15), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.05, -0.06, 0.15), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(3.4, 7.45, 9.0), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.2, 0.1, 0.3), 1e-6));

            // PNT5
            job.clear_joint_it_matrices();
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.55, 3.2, 3.15), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.02, 0.25, 0.21), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.002, 0.025, 0.021), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(3.4, 7.45, 9.0), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.28, 0.625, 0.72), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.028, 0.0625, 0.072), 1e-6));

            // PNT5 it
            job.set_joint_it_matrices(&it_matrices);
            job.set_in_normals(&in_normals);
            job.set_out_normals(out_normals.clone());
            job.set_in_tangents(&in_tangents);
            job.set_out_tangents(out_tangents.clone());
            job.run().unwrap();
            assert!(out_positions.borrow()[0].abs_diff_eq(Vec3::new(0.55, 3.2, 3.15), 1e-6));
            assert!(out_normals.borrow()[0].abs_diff_eq(Vec3::new(0.05, -0.06, 0.15), 1e-6));
            assert!(out_tangents.borrow()[0].abs_diff_eq(Vec3::new(0.005, -0.006, 0.015), 1e-6));
            assert!(out_positions.borrow()[1].abs_diff_eq(Vec3::new(3.4, 7.45, 9.0), 1e-5));
            assert!(out_normals.borrow()[1].abs_diff_eq(Vec3::new(0.2, 0.1, 0.3), 1e-6));
            assert!(out_tangents.borrow()[1].abs_diff_eq(Vec3::new(0.02, 0.01, 0.03), 1e-6));
        }
    }
}
