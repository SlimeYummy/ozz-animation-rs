[![Release Doc](https://docs.rs/ozz-animation-rs/badge.svg)](https://docs.rs/ozz-animation-rs)
[![Crate](https://img.shields.io/crates/v/ozz-animation-rs.svg)](https://crates.io/crates/ozz-animation-rs)
![github actions](https://github.com/SlimeYummy/ozz-animation-rs/actions/workflows/main.yml/badge.svg)
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
- User channels
- Skinning
- Multi-threading
- SIMD (SSE2 + NEON)
- WASM
- Serialization (rkyv & serde)

The following functions are not supported yet:
- Baked physic simulation (no plan)
- All offline features (no plan, use C++ library instead)

Ozz-animation offline features are not supported, and no plans to support. Please use the original C++ library, which has a many tools and plug-ins.

### Examples

A simple demo is in [./demo](https://github.com/SlimeYummy/ozz-animation-rs/tree/master/demo) folder. Enter the folder and execute `cargo run`.

![demo](https://raw.githubusercontent.com/SlimeYummy/ozz-animation-rs/master/demo/demo.jpg)

The test cases under [./tests](https://github.com/SlimeYummy/ozz-animation-rs/tree/master/tests) can be viewed as examples.

Ozz-animation-rs keeps the same API styles with original ozz-animation library. Therefore, you can also refer to the ozz-animation [examples](https://github.com/guillaumeblanc/ozz-animation/tree/master/samples).

Here is a very sample example:

```rust
use glam::Mat4;
use ozz_animation_rs::*;
use std::cell::RefCell;
use std::rc::Rc;

// Load resources
let skeleton = Rc::new(Skeleton::from_path("./resource/playback/skeleton.ozz").unwrap());
let animation = Rc::new(Animation::from_path("./resource/playback/animation.ozz").unwrap());

// Init sample job (Rc style)
let mut sample_job: SamplingJobRc = SamplingJob::default();
sample_job.set_animation(animation.clone());
sample_job.set_context(SamplingContext::new(animation.num_tracks()));
let sample_out = Rc::new(RefCell::new(vec![SoaTransform::default(); skeleton.num_soa_joints()]));
sample_job.set_output(sample_out.clone());

// Init local to model job (Ref style)
let mut l2m_job: LocalToModelJobRef = LocalToModelJob::default();
l2m_job.set_skeleton(&skeleton);
let sample_out_ref = sample_out.borrow();
l2m_job.set_input(sample_out_ref.as_ref());
let mut l2m_out = vec![Mat4::default(); skeleton.num_joints()];
l2m_job.set_output(&mut l2m_out);

// Run the jobs
let ratio = 0.5;

sample_job.set_ratio(ratio);
sample_job.run().unwrap();

l2m_job.run().unwrap();
l2m_out.buf().unwrap(); // Outputs here, are model-space matrices
```

### Toolchain

Since rust simd features are not stable, you need a nightly version rust to compile this library.

### Platforms

In theory, ozz-animation-rs supports all platforms supported by rust. But I only tested on the following platforms:
- Windows/Ubuntu/Mac x64 (Github actions)
- X64/Arm64 docker ([CircleCI](https://dl.circleci.com/status-badge/redirect/gh/SlimeYummy/ozz-animation-rs/tree/master))

Maybe you can run cross-platform deterministic test cases under [./tests](https://github.com/SlimeYummy/ozz-animation-rs/tree/master/tests) on your target platform.

### Compatibility

With the release of ozz-animation versions, .ozz files and some APIs will also be upgraded. Therefore ozz-animation-rs remains compatible with the corresponding version of ozz-animation, as shown in the following table:

|ozz-animation-rs|ozz-animation(C++)|
|--|--|
|0.10.x|0.15.x|
|0.9.x|0.14.x|

### Why not fixed-point?

Initially, I tried to implement similar functionality using fixed point numbers. But fixed-point performance is worse, and it is difficult to be compatible with other libraries.

With further research, I found that x64/arm64 platforms now have good support for the IEEE floating point standard. So I reimplemented this library based on f32.
