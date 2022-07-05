use crate::animation::Animation;
use crate::base::{OzzBuf, OzzBufX, OzzRes, OzzResX};
use crate::math::{ozz_quat_nlerp, ozz_vec3_lerp, OzzNumber, OzzTransform};
use anyhow::{anyhow, Result};
use bitvec::prelude::{bitvec, BitVec, Lsb0};
use nalgebra::{self as na, Quaternion, Vector3};

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

#[derive(Debug)]
pub struct SamplingJob<N: OzzNumber> {
    animation: OzzResX<Animation<N>>,
    output: OzzBufX<OzzTransform<N>>,
    ratio: N,

    max_tracks: usize,

    translations: Vec<InterpVector3<N>>,
    rotations: Vec<InterpQuaternion<N>>,
    scales: Vec<InterpVector3<N>>,

    translation_keys: Vec<i32>,
    rotation_keys: Vec<i32>,
    scale_keys: Vec<i32>,

    translation_cursor: usize,
    rotation_cursor: usize,
    scale_cursor: usize,

    outdated_translations: BitVec<u16>,
    outdated_rotations: BitVec<u16>,
    outdated_scales: BitVec<u16>,
}

impl<N: OzzNumber> Default for SamplingJob<N> {
    fn default() -> SamplingJob<N> {
        return SamplingJob {
            animation: None,
            output: None,

            ratio: N::zero(),
            max_tracks: 0,

            translations: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),

            translation_keys: Vec::new(),
            rotation_keys: Vec::new(),
            scale_keys: Vec::new(),

            translation_cursor: 0,
            rotation_cursor: 0,
            scale_cursor: 0,

            outdated_translations: BitVec::new(),
            outdated_rotations: BitVec::new(),
            outdated_scales: BitVec::new(),
        };
    }
}

impl<N: OzzNumber> SamplingJob<N> {
    pub fn new(max_tracks: usize) -> SamplingJob<N> {
        let mut job = SamplingJob::default();
        job.resize_cache(max_tracks);
        return job;
    }

    pub fn animation(&self) -> OzzResX<Animation<N>> {
        return self.animation.clone();
    }

    pub fn set_animation(&mut self, animation: &OzzRes<Animation<N>>) {
        self.animation = Some(animation.clone());
        self.translation_cursor = 0;
        self.rotation_cursor = 0;
        self.scale_cursor = 0;
    }

    pub fn reset_animation(&mut self) {
        self.animation = None;
        self.translation_cursor = 0;
        self.rotation_cursor = 0;
        self.scale_cursor = 0;
    }

    pub fn output(&self) -> OzzBufX<OzzTransform<N>> {
        return self.output.clone();
    }

    pub fn set_output(&mut self, output: &OzzBuf<OzzTransform<N>>) {
        self.output = Some(output.clone());
    }

    pub fn reset_output(&mut self) {
        self.output = None;
    }

    pub fn ratio(&self) -> N {
        return self.ratio;
    }

    pub fn set_ratio(&mut self, ratio: N) {
        let old_ratio = self.ratio;
        self.ratio = na::clamp(ratio, na::zero(), na::one());
        if self.ratio < old_ratio {
            self.translation_cursor = 0;
            self.rotation_cursor = 0;
            self.scale_cursor = 0;
        }
    }

    pub fn resize_cache(&mut self, num_tracks: usize) {
        self.max_tracks = (num_tracks + 3) & !0x3;

        self.translations = vec![Default::default(); self.max_tracks];
        self.rotations = vec![Default::default(); self.max_tracks];
        self.scales = vec![Default::default(); self.max_tracks];

        self.translation_keys = vec![0; self.max_tracks * 2];
        self.rotation_keys = vec![0; self.max_tracks * 2];
        self.scale_keys = vec![0; self.max_tracks * 2];

        self.translation_cursor = 0;
        self.rotation_cursor = 0;
        self.scale_cursor = 0;

        self.outdated_translations = bitvec![u16, Lsb0; 1; self.max_tracks];
        self.outdated_rotations = bitvec![u16, Lsb0; 1; self.max_tracks];
        self.outdated_scales = bitvec![u16, Lsb0; 1; self.max_tracks];
    }

    pub fn reset_cache(&mut self) {
        self.ratio = N::zero();
        self.translation_cursor = 0;
        self.rotation_cursor = 0;
        self.scale_cursor = 0;
    }

    pub fn max_tracks(&self) -> usize {
        return self.max_tracks;
    }

    pub fn validate(&self) -> bool {
        let animation = match &self.animation {
            Some(animation) => animation,
            None => return false,
        };
        if self.max_tracks < animation.num_aligned_tracks() {
            return false;
        }

        let output = match self.output.as_ref() {
            Some(output) => output.as_ref().borrow(),
            None => return false,
        };
        if output.len() < animation.num_tracks() {
            return false;
        }

        return true;
    }

    pub fn run(&mut self) -> Result<()> {
        if !self.validate() {
            return Err(anyhow!("Invalid SamplingJob"));
        }

        let animation = self.animation.as_ref().unwrap();
        if animation.num_tracks() == 0 {
            return Ok(());
        }

        self.update_translation_cursor();
        self.update_translation_key_frames();

        self.update_rotation_cursor();
        self.update_rotation_key_frames();

        self.update_scale_cursor();
        self.update_scale_key_frames();

        self.interpolates();

        return Ok(());
    }

    fn update_translation_cursor(&mut self) {
        let animation = self.animation.as_ref().unwrap();

        if self.translation_cursor == 0 {
            for idx in 0..animation.num_aligned_tracks() {
                self.translation_keys[2 * idx] = idx as i32;
                self.translation_keys[2 * idx + 1] = (idx + animation.num_aligned_tracks()) as i32;
                self.translation_cursor = animation.num_aligned_tracks() * 2;
            }
            self.outdated_translations.set_elements(0xFFFF);
        }

        while self.translation_cursor < animation.translations().len() {
            let track = animation.translations()[self.translation_cursor].track() as usize;
            let key_idx = self.translation_keys[(track * 2 + 1)] as usize;
            let ratio = animation.translations()[key_idx].ratio();
            if ratio > self.ratio {
                break;
            }

            *self.outdated_translations.get_mut(track).unwrap() = true;
            let base = (animation.translations()[self.translation_cursor].track() as usize) * 2;
            self.translation_keys[base] = self.translation_keys[base + 1];
            self.translation_keys[base + 1] = self.translation_cursor as i32;
            self.translation_cursor += 1;
        }
    }

    fn update_translation_key_frames(&mut self) {
        let animation = self.animation.as_ref().unwrap();

        for idx in 0..self.outdated_translations.len() {
            if !self.outdated_translations[idx] {
                continue;
            }
            self.outdated_translations
                .get_mut(idx)
                .map(|mut re| *re = false);

            let key_idx = self.translation_keys[idx * 2];
            let k0 = &animation.translations()[key_idx as usize];
            self.translations[idx as usize].ratio[0] = k0.ratio();
            self.translations[idx as usize].value[0] = k0.decompress();

            let key_idx = self.translation_keys[idx * 2 + 1];
            let k1 = &animation.translations()[key_idx as usize];
            self.translations[idx as usize].ratio[1] = k1.ratio();
            self.translations[idx as usize].value[1] = k1.decompress();
        }
    }

    fn update_rotation_cursor(&mut self) {
        let animation = self.animation.as_ref().unwrap();

        if self.rotation_cursor == 0 {
            for idx in 0..animation.num_aligned_tracks() {
                self.rotation_keys[2 * idx] = idx as i32;
                self.rotation_keys[2 * idx + 1] = (idx + animation.num_aligned_tracks()) as i32;
                self.rotation_cursor = animation.num_aligned_tracks() * 2;
            }
            self.outdated_rotations.set_elements(0xFFFF);
        }

        while self.rotation_cursor < animation.rotations().len() {
            let track = animation.rotations()[self.rotation_cursor].track() as usize;
            let key_idx = self.rotation_keys[(track * 2 + 1)] as usize;
            let ratio = animation.rotations()[key_idx].ratio();
            if ratio > self.ratio {
                break;
            }

            *self.outdated_rotations.get_mut(track).unwrap() = true;
            let base = (animation.rotations()[self.rotation_cursor].track() as usize) * 2;
            self.rotation_keys[base] = self.rotation_keys[base + 1];
            self.rotation_keys[base + 1] = self.rotation_cursor as i32;
            self.rotation_cursor += 1;
        }
    }

    fn update_rotation_key_frames(&mut self) {
        let animation = self.animation.as_ref().unwrap();

        for idx in 0..self.outdated_rotations.len() {
            if !self.outdated_rotations[idx] {
                continue;
            }
            self.outdated_rotations
                .get_mut(idx)
                .map(|mut re| *re = false);

            let key_idx = self.rotation_keys[idx * 2];
            let k0 = &animation.rotations()[key_idx as usize];
            self.rotations[idx as usize].ratio[0] = k0.ratio();
            self.rotations[idx as usize].value[0] = k0.decompress();

            let key_idx = self.rotation_keys[idx * 2 + 1];
            let k1 = &animation.rotations()[key_idx as usize];
            self.rotations[idx as usize].ratio[1] = k1.ratio();
            self.rotations[idx as usize].value[1] = k1.decompress();
        }
    }

    fn update_scale_cursor(&mut self) {
        let animation = self.animation.as_ref().unwrap();

        if self.scale_cursor == 0 {
            for idx in 0..animation.num_aligned_tracks() {
                self.scale_keys[2 * idx] = idx as i32;
                self.scale_keys[2 * idx + 1] = (idx + animation.num_aligned_tracks()) as i32;
                self.scale_cursor = animation.num_aligned_tracks() * 2;
            }
            self.outdated_scales.set_elements(0xFFFF);
        }

        while self.scale_cursor < animation.scales().len() {
            let track = animation.scales()[self.scale_cursor].track() as usize;
            let key_idx = self.scale_keys[(track * 2 + 1)] as usize;
            let ratio = animation.scales()[key_idx].ratio();
            if ratio > self.ratio {
                break;
            }

            *self.outdated_scales.get_mut(track).unwrap() = true;
            let base = (animation.scales()[self.scale_cursor].track() as usize) * 2;
            self.scale_keys[base] = self.scale_keys[base + 1];
            self.scale_keys[base + 1] = self.scale_cursor as i32;
            self.scale_cursor += 1;
        }
    }

    fn update_scale_key_frames(&mut self) {
        let animation = self.animation.as_ref().unwrap();

        for idx in 0..self.outdated_scales.len() {
            if !self.outdated_scales[idx] {
                continue;
            }
            self.outdated_scales.get_mut(idx).map(|mut re| *re = false);

            let key_idx = self.scale_keys[idx * 2];
            let k0 = &animation.scales()[key_idx as usize];
            self.scales[idx as usize].ratio[0] = k0.ratio();
            self.scales[idx as usize].value[0] = k0.decompress();

            let key_idx = self.scale_keys[idx * 2 + 1];
            let k1 = &animation.scales()[key_idx as usize];
            self.scales[idx as usize].ratio[1] = k1.ratio();
            self.scales[idx as usize].value[1] = k1.decompress();
        }
    }

    fn interpolates(&mut self) {
        let animation = self.animation.as_ref().unwrap();
        let mut output = self.output.as_ref().unwrap().borrow_mut();

        for idx in 0..animation.num_tracks() {
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
mod sampling_tests {
    use super::*;
    use crate::animation::{Float3Key, QuaternionKey};
    use crate::approx::abs_diff_eq;
    use crate::archive::{ArchiveReader, IArchive};
    use crate::base::{ozz_buf, ozz_res};
    use crate::test_helper::f16;

    #[test]
    fn test_validity() {
        let mut archive = IArchive::new("./test_files/animation-blending-1.ozz").unwrap();
        let animation = ozz_res(Animation::<f32>::read(&mut archive).unwrap());
        let aligned_tracks = animation.num_aligned_tracks();

        // invalid output
        let mut job = SamplingJob::<f32>::new(0);
        job.set_animation(&animation);
        assert!(!job.validate());

        // invalid animation
        let mut job = SamplingJob::<f32>::new(0);
        job.set_output(&ozz_buf(vec![OzzTransform::default(); aligned_tracks + 10]));
        assert!(!job.validate());

        // invalid cache size
        let mut job = SamplingJob::<f32>::new(5);
        job.set_animation(&animation);
        job.set_output(&ozz_buf(vec![OzzTransform::default(); aligned_tracks]));
        assert!(!job.validate());

        let mut job = SamplingJob::<f32>::new(aligned_tracks);
        job.set_animation(&animation);
        job.set_output(&ozz_buf(vec![OzzTransform::default(); aligned_tracks]));
        assert!(job.validate());
    }

    fn new_translations() -> Vec<Float3Key<f32>> {
        return vec![
            Float3Key::new(0.0, 0, [f16(0.0); 3]),
            Float3Key::new(0.0, 1, [f16(0.0); 3]),
            Float3Key::new(0.0, 2, [f16(0.0); 3]),
            Float3Key::new(0.0, 3, [f16(0.0); 3]),
            Float3Key::new(1.0, 0, [f16(0.0); 3]),
            Float3Key::new(1.0, 1, [f16(0.0); 3]),
            Float3Key::new(1.0, 2, [f16(0.0); 3]),
            Float3Key::new(1.0, 3, [f16(0.0); 3]),
        ];
    }

    fn new_rotations() -> Vec<QuaternionKey<f32>> {
        return vec![
            QuaternionKey::new(0.0, (0 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(0.0, (1 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(0.0, (2 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(0.0, (3 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (0 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (1 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (2 << 3) + (3 << 1), [0, 0, 0]),
            QuaternionKey::new(1.0, (3 << 3) + (3 << 1), [0, 0, 0]),
        ];
    }

    fn new_scales() -> Vec<Float3Key<f32>> {
        return vec![
            Float3Key::new(0.0, 0, [f16(1.0); 3]),
            Float3Key::new(0.0, 1, [f16(1.0); 3]),
            Float3Key::new(0.0, 2, [f16(1.0); 3]),
            Float3Key::new(0.0, 3, [f16(1.0); 3]),
            Float3Key::new(1.0, 0, [f16(1.0); 3]),
            Float3Key::new(1.0, 1, [f16(1.0); 3]),
            Float3Key::new(1.0, 2, [f16(1.0); 3]),
            Float3Key::new(1.0, 3, [f16(1.0); 3]),
        ];
    }

    const V0: Vector3<f32> = Vector3::new(0.0, 0.0, 0.0);
    const V1: Vector3<f32> = Vector3::new(1.0, 1.0, 1.0);
    const QU: Quaternion<f32> = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    const VX: Vector3<f32> = Vector3::new(1234.5678, 1234.5678, 1234.5678);
    const QX: Quaternion<f32> = Quaternion::new(1234.5678, 1234.5678, 1234.5678, 1234.5678);
    const TX: OzzTransform<f32> = OzzTransform {
        translation: VX,
        rotation: QX,
        scale: VX,
    };

    struct Frame<const S: usize> {
        ratio: f32,
        transform: [(Vector3<f32>, Quaternion<f32>, Vector3<f32>); S],
    }

    fn execute_test<const T: usize>(
        duration: f32,
        translations: Vec<Float3Key<f32>>,
        rotations: Vec<QuaternionKey<f32>>,
        scales: Vec<Float3Key<f32>>,
        frames: Vec<Frame<T>>,
    ) {
        let animation = ozz_res(Animation::<f32> {
            duration,
            num_tracks: T,
            name: String::new(),
            translations,
            rotations,
            scales,
        });
        let mut job = SamplingJob::<f32>::new(T);
        job.set_animation(&animation);

        for frame in &frames {
            let output = ozz_buf(vec![TX; T + 1]);
            job.set_output(&output);
            job.set_ratio(frame.ratio);
            job.run().unwrap();

            if T == 0 {
                assert_eq!(output.borrow()[0], TX);
            }

            for idx in 0..T {
                let out = output.borrow()[idx];
                assert!(
                    abs_diff_eq!(out.translation, frame.transform[idx].0, epsilon = 0.000002),
                    "ratio={} translation idx={}",
                    frame.ratio,
                    idx
                );
                assert!(
                    abs_diff_eq!(out.rotation, frame.transform[idx].1, epsilon = 0.000002),
                    "ratio={} rotation idx={}",
                    frame.ratio,
                    idx
                );
                assert!(
                    abs_diff_eq!(out.scale, frame.transform[idx].2, epsilon = 0.000002),
                    "ratio={} scale idx={}",
                    frame.ratio,
                    idx
                );
            }
        }
    }

    #[test]
    fn test_sampling() {
        fn frame(ratio: f32, t1: f32, t2: f32, t3: f32, t4: f32) -> Frame<4> {
            return Frame {
                ratio,
                transform: [
                    (Vector3::new(t1, 0.0, 0.0), QU, V1),
                    (Vector3::new(t2, 0.0, 0.0), QU, V1),
                    (Vector3::new(t3, 0.0, 0.0), QU, V1),
                    (Vector3::new(t4, 0.0, 0.0), QU, V1),
                ],
            };
        }

        execute_test::<4>(
            1.0,
            vec![
                Float3Key::new(0.0, 0, [f16(-1.0), 0, 0]),
                Float3Key::new(0.0, 1, [f16(0.0), 0, 0]),
                Float3Key::new(0.0, 2, [f16(2.0), 0, 0]),
                Float3Key::new(0.0, 3, [f16(7.0), 0, 0]),
                Float3Key::new(1.0, 0, [f16(-1.0), 0, 0]),
                Float3Key::new(1.0, 1, [f16(0.0), 0, 0]),
                Float3Key::new(0.200000003, 2, [f16(6.0), 0, 0]),
                Float3Key::new(0.200000003, 3, [f16(7.0), 0, 0]),
                Float3Key::new(0.400000006, 2, [f16(8.0), 0, 0]),
                Float3Key::new(0.600000024, 3, [f16(9.0), 0, 0]),
                Float3Key::new(0.600000024, 2, [f16(10.0), 0, 0]),
                Float3Key::new(1.0, 2, [f16(11.0), 0, 0]),
                Float3Key::new(1.0, 3, [f16(9.0), 0, 0]),
            ],
            new_rotations(),
            new_scales(),
            vec![
                frame(-0.2, -1.0, 0.0, 2.0, 7.0),
                frame(0.0, -1.0, 0.0, 2.0, 7.0),
                frame(0.0000001, -1.0, 0.0, 2.0, 7.0),
                frame(0.1, -1.0, 0.0, 4.0, 7.0),
                frame(0.2, -1.0, 0.0, 6.0, 7.0),
                frame(0.3, -1.0, 0.0, 7.0, 7.5),
                frame(0.4, -1.0, 0.0, 8.0, 8.0),
                frame(0.3999999, -1.0, 0.0, 8.0, 8.0),
                frame(0.4000001, -1.0, 0.0, 8.0, 8.0),
                frame(0.5, -1.0, 0.0, 9.0, 8.5),
                frame(0.6, -1.0, 0.0, 10.0, 9.0),
                frame(0.9999999, -1.0, 0.0, 11.0, 9.0),
                frame(1.0, -1.0, 0.0, 11.0, 9.0),
                frame(1.000001, -1.0, 0.0, 11.0, 9.0),
                frame(0.5, -1.0, 0.0, 9.0, 8.5),
                frame(0.9999999, -1.0, 0.0, 11.0, 9.0),
                frame(0.0000001, -1.0, 0.0, 2.0, 7.0),
            ],
        );
    }

    #[test]
    fn test_sampling_no_track() {
        execute_test::<0>(46.0, vec![], vec![], vec![], vec![]);
    }

    #[test]
    fn test_sampling_1_track_0_key() {
        execute_test::<1>(
            46.0,
            new_translations(),
            new_rotations(),
            new_scales(),
            (-2..12)
                .map(|x| Frame {
                    ratio: x as f32 / 10.0,
                    transform: [(V0, QU, V1)],
                })
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_sampling_1_track_1_key() {
        let mut translations = new_translations();
        translations[0] = Float3Key::new(0.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);
        translations[4] = Float3Key::new(1.0 / 46.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);

        execute_test::<1>(
            46.0,
            translations,
            new_rotations(),
            new_scales(),
            (-2..12)
                .map(|x| Frame {
                    ratio: x as f32 / 10.0,
                    transform: [(Vector3::new(1.0, -1.0, 5.0), QU, V1)],
                })
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_sampling_1_track_2_key() {
        fn frame(ratio: f32, v: Vector3<f32>) -> Frame<2> {
            return Frame {
                ratio,
                transform: [(v, QU, V1), (V0, QU, V1)],
            };
        }

        execute_test::<2>(
            46.0,
            vec![
                Float3Key::new(0.0, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(0.0, 1, [f16(0.0); 3]),
                Float3Key::new(0.0, 2, [f16(0.0); 3]),
                Float3Key::new(0.0, 3, [f16(0.0); 3]),
                Float3Key::new(0.5 / 46.0, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(1.0, 1, [f16(0.0); 3]),
                Float3Key::new(1.0, 2, [f16(0.0); 3]),
                Float3Key::new(1.0, 3, [f16(0.0); 3]),
                Float3Key::new(1.0 / 46.0, 0, [f16(2.0), f16(4.0), f16(8.0)]),
                Float3Key::new(1.0, 0, [f16(2.0), f16(4.0), f16(8.0)]),
            ],
            new_rotations(),
            new_scales(),
            vec![
                frame(0.0, Vector3::new(1.0, 2.0, 4.0)),
                frame(0.5 / 46.0, Vector3::new(1.0, 2.0, 4.0)),
                frame(1.0 / 46.0, Vector3::new(2.0, 4.0, 8.0)),
                frame(1.0, Vector3::new(2.0, 4.0, 8.0)),
                frame(0.75 / 46.0, Vector3::new(1.5, 3.0, 6.0)),
            ],
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_sampling_4_track_2_key() {
        execute_test::<4>(
            1.0,
            vec![
                Float3Key::new(0.0, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(0.0, 1, [f16(0.0); 3]),
                Float3Key::new(0.0, 2, [f16(0.0); 3]),
                Float3Key::new(0.0, 3, [f16(-1.0), f16(-2.0), f16(-4.0)]),
                Float3Key::new(0.5, 0, [f16(1.0), f16(2.0), f16(4.0)]),
                Float3Key::new(1.0, 1, [f16(0.0); 3]),
                Float3Key::new(1.0, 2, [f16(0.0); 3]),
                Float3Key::new(1.0, 3, [f16(-2.0), f16(-4.0), f16(-8.0)]),
                Float3Key::new(0.8, 0, [f16(2.0), f16(4.0), f16(8.0)]),
                Float3Key::new(1.0, 0, [f16(2.0), f16(4.0), f16(8.0)]),
            ],
            vec![
                QuaternionKey::new(0.0, (0 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(0.0, (1 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(0.0, (2 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(0.0, (3 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (0 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (1 << 3) + (1 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (2 << 3) + (3 << 1), [0, 0, 0]),
                QuaternionKey::new(1.0, (3 << 3) + (3 << 1), [0, 0, 0]),
            ],
            vec![
                Float3Key::new(0.0, 0, [f16(1.0); 3]),
                Float3Key::new(0.0, 1, [f16(1.0); 3]),
                Float3Key::new(0.0, 2, [f16(0.0); 3]),
                Float3Key::new(0.0, 3, [f16(1.0); 3]),
                Float3Key::new(1.0, 0, [f16(1.0); 3]),
                Float3Key::new(1.0, 1, [f16(1.0); 3]),
                Float3Key::new(0.5, 2, [f16(0.0); 3]),
                Float3Key::new(1.0, 3, [f16(1.0); 3]),
                Float3Key::new(0.8, 2, [f16(-1.0); 3]),
                Float3Key::new(1.0, 2, [f16(-1.0); 3]),
            ],
            vec![
                Frame {ratio: 0.0, transform: [
                    (Vector3::new(1.0, 2.0, 4.0), QU, V1),
                    (V0, QU, V1),
                    (V0, QU, V0),
                    (Vector3::new(-1.0, -2.0, -4.0), QU, V1),
                ]},
                Frame {ratio: 0.5, transform: [
                    (Vector3::new(1.0, 2.0, 4.0), QU, V1),
                    (V0, Quaternion::new(0.7071067, 0.0, 0.7071067, 0.0), V1),
                    (V0, QU, V0),
                    (Vector3::new(-1.5, -3.0, -6.0), QU, V1),
                ]},
                Frame {ratio: 1.0, transform: [
                    (Vector3::new(2.0, 4.0, 8.0), QU, V1),
                    (V0, Quaternion::new(0.0, 0.0, 1.0, 0.0), V1),
                    (V0, QU, -V1),
                    (Vector3::new(-2.0, -4.0, -8.0), QU, V1),
                ]},
            ],
        );
    }

    #[test]
    fn test_cache() {
        let mut translations = new_translations();
        translations[0] = Float3Key::new(0.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);
        translations[4] = Float3Key::new(1.0, 0, [f16(1.0), f16(-1.0), f16(5.0)]);

        let animation1 = ozz_res(Animation::<f32> {
            duration: 46.0,
            num_tracks: 1,
            name: String::new(),
            translations: translations.clone(),
            rotations: new_rotations(),
            scales: new_scales(),
        });

        let animation2 = ozz_res(Animation::<f32> {
            duration: 46.0,
            num_tracks: 1,
            name: String::new(),
            translations: translations.clone(),
            rotations: new_rotations(),
            scales: new_scales(),
        });

        let mut job = SamplingJob::<f32>::new(animation1.num_tracks());
        job.set_animation(&animation1);

        fn run_test(job: &mut SamplingJob<f32>) -> Result<()> {
            let output = ozz_buf(vec![TX; 1]);
            job.set_output(&output);
            job.run()?;
            for item in output.as_ref().borrow().iter() {
                assert_eq!(item.translation, Vector3::new(1.0, -1.0, 5.0));
                assert_eq!(item.rotation, Quaternion::new(1.0, 0.0, 0.0, 0.0));
                assert_eq!(item.scale, Vector3::new(1.0, 1.0, 1.0));
            }
            return Ok(());
        }

        job.set_ratio(0.0);
        run_test(&mut job).unwrap();

        // reuse cache
        run_test(&mut job).unwrap();

        // reset cache
        job.reset_cache();
        run_test(&mut job).unwrap();

        // change animation
        job.set_animation(&animation2);
        run_test(&mut job).unwrap();

        // change animation
        job.set_animation(&animation2);
        run_test(&mut job).unwrap();
    }

    #[test]
    fn test_cache_resize() {
        let animation = ozz_res(Animation::<f32> {
            duration: 46.0,
            num_tracks: 7,
            name: String::new(),
            translations: vec![],
            rotations: vec![],
            scales: vec![],
        });

        let mut job = SamplingJob::<f32>::new(0);
        job.set_animation(&animation);
        let output = ozz_buf(vec![TX; animation.num_tracks()]);
        job.set_output(&output);

        assert!(!job.validate());

        job.resize_cache(7);
        assert!(job.validate());

        job.resize_cache(1);
        assert!(!job.validate());
    }

    // #[test]
    // fn test_sampling_job_run() {
    //     let mut archive = IArchive::new("./test_files/animation-simple.ozz").unwrap();
    //     let animation = Rc::new(Animation::read(&mut archive).unwrap());
    //     let mut job = SamplingJob::new(animation);

    //     let mut counter = 0;
    //     let mut ratio = 0f32;
    //     while ratio <= 1.0f32 {
    //         counter += 1;
    //         ratio += 0.005;
    //         job.run(ratio);

    //         let file_no = match counter {
    //             1 => 1,
    //             100 => 2,
    //             200 => 3,
    //             _ => continue,
    //         };

    //         let file = format!("./test_files/sampling/translations_{}", file_no);
    //         let chunk: Vec<InterpVector3<f32>> = read_chunk(&file).unwrap();
    //         for idx in 0..job.translations.len() {
    //             assert_eq!(chunk[idx], job.translations[idx]);
    //         }

    //         let file = format!("./test_files/sampling/rotations_{}", file_no);
    //         let chunk: Vec<InterpQuaternion<f32>> = read_chunk(&file).unwrap();
    //         for idx in 0..job.rotations.len() {
    //             assert_eq!(chunk[idx], job.rotations[idx]);
    //         }

    //         let file = format!("./test_files/sampling/scales_{}", file_no);
    //         let chunk: Vec<InterpVector3<f32>> = read_chunk(&file).unwrap();
    //         for idx in 0..job.scales.len() {
    //             assert_eq!(chunk[idx], job.scales[idx]);
    //         }

    //         let file = format!("./test_files/sampling/output_{}", file_no);
    //         let chunk: Vec<OzzTransform<f32>> = read_chunk(&file).unwrap();
    //         for idx in 0..job.output.borrow().len() {
    //             assert_eq!(chunk[idx], job.output.borrow()[idx]);
    //         }
    //     }
    // }
}
