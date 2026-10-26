//! Skinning job golden/deterministic tests.
//!
//! There is no C++ reference data for skinning: the deterministic snapshot below
//! is the cross-platform baseline (generated on the reference platform with
//! `SAVE_TO_EXPECTED=1`, every other platform must reproduce it bit-exactly).

use glam::{Mat4, Quat, Vec3};
use ozz_animation_rs::*;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen_test::*;

mod common;

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct BranchOut {
    influences: usize,
    it: bool,
    normals: bool,
    tangents: bool,
    out_positions: Vec<Vec3>,
    out_normals: Vec<Vec3>,
    out_tangents: Vec<Vec3>,
}

#[derive(Debug, PartialEq)]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
struct TestData {
    branches: Vec<BranchOut>,
}

#[test]
#[wasm_bindgen_test]
fn test_skinning_deterministic() {
    run_skinning(|data| {
        common::compare_with_rkyv("skinning", "skinning", data).unwrap();
    });
}

fn run_skinning<T>(tester: T)
where
    T: Fn(&TestData),
{
    type SkinningJobTest<'t> = SkinningJob<&'t [Mat4], &'t [u16], &'t [f32], &'t [Vec3], Rc<RefCell<Vec<Vec3>>>>;

    let matrices: [Mat4; 4] = [
        Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 2.0, 0.5),
            Quat::from_axis_angle(Vec3::Y, 0.5),
            Vec3::new(-1.0, 0.5, 2.0),
        ),
        Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::Z, -1.0), Vec3::new(0.0, 0.0, 3.0)),
        Mat4::from_cols(
            Vec3::new(2.0, 0.0, 0.0).extend(0.0),
            Vec3::new(0.0, -1.0, 0.0).extend(0.0),
            Vec3::new(0.0, 0.0, 1.0).extend(0.0),
            Vec3::new(0.5, -2.0, 1.5).extend(1.0),
        ),
    ];
    let it_matrices: [Mat4; 4] = [
        Mat4::from_translation(Vec3::new(-1.0, -2.0, -3.0)),
        Mat4::from_scale_rotation_translation(
            Vec3::new(1.0, 0.5, 2.0),
            Quat::from_axis_angle(Vec3::Y, -0.5),
            Vec3::new(1.0, -0.5, -2.0),
        ),
        Mat4::from_rotation_translation(Quat::from_axis_angle(Vec3::X, 0.25), Vec3::new(0.0, 0.0, -3.0)),
        Mat4::IDENTITY,
    ];

    let vertex_count = 3usize;
    let in_positions: [Vec3; 3] = [
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(-4.0, 5.0, -6.0),
        Vec3::new(0.7, -0.8, 0.9),
    ];
    let in_normals: [Vec3; 3] = [
        Vec3::new(0.1, 0.2, 0.3),
        Vec3::new(0.4, -0.5, 0.6),
        Vec3::new(-0.7, 0.8, -0.9),
    ];
    let in_tangents: [Vec3; 3] = [
        Vec3::new(0.01, 0.02, 0.03),
        Vec3::new(0.04, 0.05, 0.06),
        Vec3::new(0.07, 0.08, 0.09),
    ];

    // Indices cycle through the joints, weights stay small enough for the restored
    // last weight to stay positive.
    let indices_for = |n: usize| -> Vec<u16> { (0..vertex_count * n).map(|i| (i % 4) as u16).collect() };
    let weights_for = |n: usize| -> Vec<f32> {
        (0..vertex_count * (n - 1))
            .map(|i| 0.05f32 + 0.1 * ((i % 3) as f32))
            .collect()
    };

    let mut branches = Vec::new();
    for influences in 1..=5usize {
        let indices = indices_for(influences);
        let weights = weights_for(influences);
        for (normals, tangents) in [(false, false), (true, false), (true, true)] {
            for it in [false, true] {
                let out_positions = Rc::new(RefCell::new(vec![Vec3::ZERO; vertex_count]));
                let out_normals = Rc::new(RefCell::new(vec![Vec3::ZERO; vertex_count]));
                let out_tangents = Rc::new(RefCell::new(vec![Vec3::ZERO; vertex_count]));

                let mut job: SkinningJobTest = SkinningJob::default();
                job.set_vertex_count(vertex_count);
                job.set_influences_count(influences);
                job.set_joint_matrices(&matrices);
                job.set_joint_indices(&indices);
                if influences > 1 {
                    job.set_joint_weights(&weights);
                }
                job.set_in_positions(&in_positions);
                job.set_out_positions(out_positions.clone());
                if normals {
                    job.set_in_normals(&in_normals);
                    job.set_out_normals(out_normals.clone());
                }
                if tangents {
                    job.set_in_tangents(&in_tangents);
                    job.set_out_tangents(out_tangents.clone());
                }
                if it {
                    job.set_joint_it_matrices(&it_matrices);
                }
                job.run().unwrap();

                branches.push(BranchOut {
                    influences,
                    it,
                    normals,
                    tangents,
                    out_positions: out_positions.buf().unwrap().to_vec(),
                    out_normals: out_normals.buf().unwrap().to_vec(),
                    out_tangents: out_tangents.buf().unwrap().to_vec(),
                });
            }
        }
    }
    tester(&TestData { branches });
}
