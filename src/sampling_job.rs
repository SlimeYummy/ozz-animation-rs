use crate::animation::Animation;
use crate::math::{ozz_quat_nlerp, ozz_vec3_lerp, OzzNumber, OzzTransform};
use bitvec::prelude::{bitvec, BitVec, LocalBits};
use nalgebra::{self as na, Quaternion, Vector3};
use std::cell::RefCell;
use std::rc::Rc;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct InterpVector3<N: OzzNumber> {
    pub ratio: [N; 2],
    pub value: [Vector3<N>; 2],
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct InterpQuaternion<N: OzzNumber> {
    pub ratio: [N; 2],
    pub value: [Quaternion<N>; 2],
}

pub struct SamplingJob<N: OzzNumber> {
    animation: Rc<Animation<N>>,
    output: Rc<RefCell<Vec<OzzTransform<N>>>>,
    ratio: N,

    translations: Vec<InterpVector3<N>>,
    rotations: Vec<InterpQuaternion<N>>,
    scales: Vec<InterpVector3<N>>,

    translation_keys: Vec<i32>,
    rotation_keys: Vec<i32>,
    scale_keys: Vec<i32>,

    translation_cursor: usize,
    rotation_cursor: usize,
    scale_cursor: usize,

    outdated_translations: BitVec<LocalBits, u32>,
    outdated_rotations: BitVec<LocalBits, u32>,
    outdated_scales: BitVec<LocalBits, u32>,
}

impl<N: OzzNumber> SamplingJob<N> {
    pub fn new(animation: Rc<Animation<N>>) -> SamplingJob<N> {
        let num_tracks = animation.num_tracks();
        let num_aligned_tracks = animation.num_aligned_tracks();
        let mut job = SamplingJob {
            animation: animation.clone(),
            output: Rc::new(RefCell::new(vec![Default::default(); num_tracks])),
            ratio: N::zero(),

            translations: vec![Default::default(); num_tracks],
            rotations: vec![Default::default(); num_tracks],
            scales: vec![Default::default(); num_tracks],

            translation_keys: vec![0; num_aligned_tracks * 2],
            rotation_keys: vec![0; num_aligned_tracks * 2],
            scale_keys: vec![0; num_aligned_tracks * 2],

            translation_cursor: 0,
            rotation_cursor: 0,
            scale_cursor: 0,

            outdated_translations: bitvec![LocalBits, u32; 1; num_tracks],
            outdated_rotations: bitvec![LocalBits, u32; 1; num_tracks],
            outdated_scales: bitvec![LocalBits, u32; 1; num_tracks],
        };

        for idx in 0..num_aligned_tracks {
            job.translation_keys[2 * idx] = idx as i32;
            job.translation_keys[2 * idx + 1] = (idx + num_aligned_tracks) as i32;
            job.translation_cursor = num_aligned_tracks * 2;

            job.rotation_keys[2 * idx] = idx as i32;
            job.rotation_keys[2 * idx + 1] = (idx + num_aligned_tracks) as i32;
            job.rotation_cursor = num_aligned_tracks * 2;

            job.scale_keys[2 * idx] = idx as i32;
            job.scale_keys[2 * idx + 1] = (idx + num_aligned_tracks) as i32;
            job.scale_cursor = num_aligned_tracks * 2;
        }

        return job;
    }

    pub fn output(&self) -> Rc<RefCell<Vec<OzzTransform<N>>>> {
        return self.output.clone();
    }

    pub fn run(&mut self, ratio: N) {
        let num_tracks = self.animation.num_tracks();
        if num_tracks == 0 {
            return;
        }

        self.ratio = na::clamp(ratio, na::zero(), na::one());

        self.update_translation_cursor();
        self.update_translation_key_frames();

        self.update_rotation_cursor();
        self.update_rotation_key_frames();

        self.update_scale_cursor();
        self.update_scale_key_frames();

        self.interpolates();
    }

    fn update_translation_cursor(&mut self) {
        while self.translation_cursor < self.animation.translations().len() {
            let track = self.animation.translations()[self.translation_cursor].track() as usize;
            let key_idx = self.translation_keys[(track * 2 + 1)] as usize;
            let ratio = self.animation.translations()[key_idx].ratio();
            if ratio > self.ratio {
                break;
            }

            *self.outdated_translations.get_mut(track).unwrap() = true;
            let base =
                (self.animation.translations()[self.translation_cursor].track() as usize) * 2;
            self.translation_keys[base] = self.translation_keys[base + 1];
            self.translation_keys[base + 1] = self.translation_cursor as i32;
            self.translation_cursor += 1;
        }
    }

    fn update_translation_key_frames(&mut self) {
        for idx in 0..self.outdated_translations.len() {
            if !self.outdated_translations[idx] {
                continue;
            }
            self.outdated_translations
                .get_mut(idx)
                .map(|mut re| *re = false);

            let key_idx = self.translation_keys[idx * 2];
            let k0 = self.animation.translations()[key_idx as usize];
            self.translations[idx as usize].ratio[0] = k0.ratio();
            self.translations[idx as usize].value[0] = k0.decompress();

            let key_idx = self.translation_keys[idx * 2 + 1];
            let k1 = self.animation.translations()[key_idx as usize];
            self.translations[idx as usize].ratio[1] = k1.ratio();
            self.translations[idx as usize].value[1] = k1.decompress();
        }
    }

    fn update_rotation_cursor(&mut self) {
        while self.rotation_cursor < self.animation.rotations().len() {
            let track = self.animation.rotations()[self.rotation_cursor].track() as usize;
            let key_idx = self.rotation_keys[(track * 2 + 1)] as usize;
            let ratio = self.animation.rotations()[key_idx].ratio();
            if ratio > self.ratio {
                break;
            }

            *self.outdated_rotations.get_mut(track).unwrap() = true;
            let base = (self.animation.rotations()[self.rotation_cursor].track() as usize) * 2;
            self.rotation_keys[base] = self.rotation_keys[base + 1];
            self.rotation_keys[base + 1] = self.rotation_cursor as i32;
            self.rotation_cursor += 1;
        }
    }

    fn update_rotation_key_frames(&mut self) {
        for idx in 0..self.outdated_rotations.len() {
            if !self.outdated_rotations[idx] {
                continue;
            }
            self.outdated_rotations
                .get_mut(idx)
                .map(|mut re| *re = false);

            let key_idx = self.rotation_keys[idx * 2];
            let k0 = self.animation.rotations()[key_idx as usize];
            self.rotations[idx as usize].ratio[0] = k0.ratio();
            self.rotations[idx as usize].value[0] = k0.decompress();

            let key_idx = self.rotation_keys[idx * 2 + 1];
            let k1 = self.animation.rotations()[key_idx as usize];
            self.rotations[idx as usize].ratio[1] = k1.ratio();
            self.rotations[idx as usize].value[1] = k1.decompress();
        }
    }

    fn update_scale_cursor(&mut self) {
        while self.scale_cursor < self.animation.scales().len() {
            let track = self.animation.scales()[self.scale_cursor].track() as usize;
            let key_idx = self.scale_keys[(track * 2 + 1)] as usize;
            let ratio = self.animation.scales()[key_idx].ratio();
            if ratio > self.ratio {
                break;
            }

            *self.outdated_scales.get_mut(track).unwrap() = true;
            let base = (self.animation.scales()[self.scale_cursor].track() as usize) * 2;
            self.scale_keys[base] = self.scale_keys[base + 1];
            self.scale_keys[base + 1] = self.scale_cursor as i32;
            self.scale_cursor += 1;
        }
    }

    fn update_scale_key_frames(&mut self) {
        for idx in 0..self.outdated_scales.len() {
            if !self.outdated_scales[idx] {
                continue;
            }
            self.outdated_scales.get_mut(idx).map(|mut re| *re = false);

            let key_idx = self.scale_keys[idx * 2];
            let k0 = self.animation.scales()[key_idx as usize];
            self.scales[idx as usize].ratio[0] = k0.ratio();
            self.scales[idx as usize].value[0] = k0.decompress();

            let key_idx = self.scale_keys[idx * 2 + 1];
            let k1 = self.animation.scales()[key_idx as usize];
            self.scales[idx as usize].ratio[1] = k1.ratio();
            self.scales[idx as usize].value[1] = k1.decompress();
        }
    }

    fn interpolates(&mut self) {
        let mut output = self.output.borrow_mut();

        for idx in 0..self.animation.num_tracks() {
            let translation = &self.translations[idx];
            let translation_ratio =
                (self.ratio - translation.ratio[0]) / (translation.ratio[1] - translation.ratio[0]);
            output[idx].translation = ozz_vec3_lerp(
                &translation.value[0],
                &translation.value[1],
                translation_ratio,
            );

            let rotation = &self.rotations[idx];
            let rotation_ratio =
                (self.ratio - rotation.ratio[0]) / (rotation.ratio[1] - rotation.ratio[0]);
            output[idx].rotation =
                ozz_quat_nlerp(&rotation.value[0], &rotation.value[1], rotation_ratio);

            let scale = &self.scales[idx];
            let scale_ratio = (self.ratio - scale.ratio[0]) / (scale.ratio[1] - scale.ratio[0]);
            output[idx].scale = ozz_vec3_lerp(&scale.value[0], &scale.value[1], scale_ratio);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::{ArchiveReader, IArchive};
    use crate::test_helper::read_chunk;

    #[test]
    fn test_sampling_job_new() {
        let mut archive = IArchive::new("./test_files/animation-simple.ozz").unwrap();
        let animation = Rc::new(Animation::<f32>::read(&mut archive).unwrap());
        let job = SamplingJob::new(animation);

        assert_eq!(job.translations.len(), 67);
        assert_eq!(job.rotations.len(), 67);
        assert_eq!(job.scales.len(), 67);

        assert_eq!(job.translation_keys.len(), 68 * 2);
        assert_eq!(job.rotation_keys.len(), 68 * 2);
        assert_eq!(job.scale_keys.len(), 68 * 2);

        assert_eq!(job.outdated_translations.len(), 67);
        assert_eq!(job.outdated_rotations.len(), 67);
        assert_eq!(job.outdated_scales.len(), 67);
    }

    #[test]
    fn test_sampling_job_run() {
        let mut archive = IArchive::new("./test_files/animation-simple.ozz").unwrap();
        let animation = Rc::new(Animation::read(&mut archive).unwrap());
        let mut job = SamplingJob::new(animation);

        let mut counter = 0;
        let mut ratio = 0f32;
        while ratio <= 1.0f32 {
            counter += 1;
            ratio += 0.005;
            job.run(ratio);

            let file_no = match counter {
                1 => 1,
                100 => 2,
                200 => 3,
                _ => continue,
            };

            let file = format!("./test_files/sampling/translations_{}", file_no);
            let chunk: Vec<InterpVector3<f32>> = read_chunk(&file).unwrap();
            for idx in 0..job.translations.len() {
                assert_eq!(chunk[idx], job.translations[idx]);
            }

            let file = format!("./test_files/sampling/rotations_{}", file_no);
            let chunk: Vec<InterpQuaternion<f32>> = read_chunk(&file).unwrap();
            for idx in 0..job.rotations.len() {
                assert_eq!(chunk[idx], job.rotations[idx]);
            }

            let file = format!("./test_files/sampling/scales_{}", file_no);
            let chunk: Vec<InterpVector3<f32>> = read_chunk(&file).unwrap();
            for idx in 0..job.scales.len() {
                assert_eq!(chunk[idx], job.scales[idx]);
            }

            let file = format!("./test_files/sampling/output_{}", file_no);
            let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
            for idx in 0..job.output.borrow().len() {
                assert_eq!(chunk[idx], job.output.borrow()[idx]);
            }
        }
    }
}
