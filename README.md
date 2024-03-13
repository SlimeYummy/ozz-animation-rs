[![Release Doc](https://docs.rs/ozz-animation-rs/badge.svg)](https://docs.rs/ozz-animation-rs)
[![Crate](https://img.shields.io/crates/v/ozz-animation-rs.svg)](https://crates.io/crates/ozz-animation-rs)
![github actions](https://github.com/FenQiDian/ozz-animation-rs/actions/workflows/main.yml/badge.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/SlimeYummy/ozz-animation-rs/tree/master.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/SlimeYummy/ozz-animation-rs/tree/master)

# Ozz-animation-rs

Ozz-animation-rs is a rust version skeletal animation library with cross-platform deterministic.

Ozz-animation-rs is based on [ozz-animation](https://github.com/guillaumeblanc/ozz-animation) library, an open source c++ 3d skeletal animation library and toolset. Ozz-animation-rs only implement ozz-animation's runtime part. You should use this library with ozz-animation's toolset.

In order to introduce cross-platform deterministic, ozz-animation-rs does not simply wrap ozz-animation's runtime, but rewrite the full runtime library in rust. So it can be used in network game scenarios, such as lock-step networking synchronize.

### Features

The library supports almost all runtime features supported by C++ version ozz, including:
- Animation playback
- Joint attachment
- Animation blending (partial/additive blending)
- Two bone IK
- Aim (Look-at) IK
- Multi-threading
- SIMD (SSE2 + NEON)
- WASM

The following functions are not supported yet:
- User channels (developing)
- Skinning (developing)
- Baked physic simulation (no plan)

Ozz-animation offline features are not supported, and no plans to support. Please use the original C++ library, which has a many tools and plug-ins.

### Examples

A simple demo is in [./demo](https://github.com/FenQiDian/ozz-animation-rs/tree/master/demo) folder. Enter the folder and execute `cargo run`.

![demo](https://raw.githubusercontent.com/SlimeYummy/ozz-animation-rs/master/demo/demo.jpg)

The test cases under [./tests](https://github.com/FenQiDian/ozz-animation-rs/tree/master/tests) can be viewed as examples.

Ozz-animation-rs keeps the same API styles with original ozz-animation library. Therefore, you can also refer to the ozz-animation [examples](https://github.com/guillaumeblanc/ozz-animation/tree/master/samples).

Here is a very sample example:

```rust
use glam::Mat4;
use ozz_animation_rs::*;
use ozz_animation_rs::math::*;
use std::rc::Rc;

// Load resources
let skeleton = Rc::new(Skeleton::from_path("./resource/skeleton.ozz").unwrap());
let animation = Rc::new(Animation::from_path("./resource/animation.ozz").unwrap());

// Init sample job
let mut sample_job: SamplingJob = SamplingJob::default();
sample_job.set_animation(animation.clone());
sample_job.set_context(SamplingContext::new(animation.num_tracks()));
let sample_out = ozz_buf(vec![SoaTransform::default(); skeleton.num_soa_joints()]);
sample_job.set_output(sample_out.clone());

// Init local to model job
let mut l2m_job: LocalToModelJob = LocalToModelJob::default();
l2m_job.set_skeleton(skeleton.clone());
l2m_job.set_input(sample_out.clone());
let l2m_out = ozz_buf(vec![Mat4::default(); skeleton.num_joints()]);
l2m_job.set_output(l2m_out.clone());

// Run the jobs
let ratio = 0.5;

sample_job.set_ratio(ratio);
sample_job.run().unwrap();

l2m_job.run().unwrap();
l2m_out.vec().unwrap(); // Outputs here, are model-space matrices
```

### Toolchain

Since rust simd features are not stable, you need a nightly version rust to compile this library.

### Platforms

In theory, ozz-animation-rs supports all platforms supported by rust. But I only tested on the following platforms:
- Windows/Ubuntu/Mac x64 (Github actions)
- X64/Arm64 docker ([CircleCI](https://dl.circleci.com/status-badge/redirect/gh/SlimeYummy/ozz-animation-rs/tree/master))

Maybe you can run cross-platform deterministic test cases under [./tests](https://github.com/FenQiDian/ozz-animation-rs/tree/master/tests) on your target platform.

### Why not fixed-point?

Initially, I tried to implement similar functionality using fixed point numbers. But fixed-point performance is worse, and it is difficult to be compatible with other libraries.

With further research, I found that x64/arm64 platforms now have good support for the IEEE floating point standard. So I reimplemented this library based on f32.
