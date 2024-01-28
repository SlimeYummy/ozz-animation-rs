[![Release Doc](https://docs.rs/ozz-animation-rs/badge.svg)](https://docs.rs/ozz-animation-rs)
[![Crate](https://img.shields.io/crates/v/ozz-animation-rs.svg)](https://crates.io/crates/ozz-animation-rs)
![github actions](https://github.com/FenQiDian/ozz-animation-rs/actions/workflows/main.yml/badge.svg)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/SlimeYummy/ozz-animation-rs/tree/master.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/SlimeYummy/ozz-animation-rs/tree/master)

# Ozz-animation-rs

Ozz-animation-rs is a rust version skeletal animation library with cross-platform deterministic.

Ozz-animation-rs is based on [ozz-animation](https://github.com/guillaumeblanc/ozz-animation) library, an open source c++ 3d skeletal animation library and toolset. Ozz-animation-rs only implement ozz-animation's runtime part. You should use this library with ozz-animation's toolset.

In order to introduce cross-platform deterministic, ozz-animation-rs does not simply wrap ozz-animation's runtime, but rewrite the full runtime library in rust. So it can be used in network game scenarios, such as lock-step networking synchronize.

### Features

| Ozz-animation Features | State |
| ------- | -------- |
| Animation playback | done |
| Joint attachment | done |
| Animation blending | done |
| Partial animations blending | done |
| Additive blending | done |
| Skinning | unsupported |
| User channels | unimplemented |
| Baked physic simulation | unsupported |
| Keyframe reduction | unsupported |
| Offline libraries usage | unsupported |
| Two bone IK | done |
| Look-at | done |
| Foot ik | done |
| Multi-threading | done |
| SIMD | done |

I have no plan to implement "unsupported" features, currently. Or you can try to implement them yourself.

### Toolchain

Since rust simd features are not stable, you need a nightly version rust to compile this library.

### Platforms

In theory, ozz-animation-rs supports all platforms supported by rust. But I only tested on the following platforms:
- Windows/Ubuntu/Mac x64 (Github actions)
- X64/Arm64 docker ([CircleCI](https://dl.circleci.com/status-badge/redirect/gh/SlimeYummy/ozz-animation-rs/tree/master))

Maybe you can run cross-platform deterministic test cases under [./tests](https://github.com/FenQiDian/ozz-animation-rs/tree/master/tests) on your target platform.

### Examples

The test cases under [./tests](https://github.com/FenQiDian/ozz-animation-rs/tree/master/tests) can be viewed as examples.

Ozz-animation-rs keeps the same API styles with original ozz-animation library. Therefore, you can also refer to the ozz-animation [examples](https://github.com/guillaumeblanc/ozz-animation/tree/master/samples).

### Why not fixed-point?

Initially, I tried to implement similar functionality using fixed point numbers. But fixed-point performance is worse, and it is difficult to be compatible with other libraries.

With further research, I found that x64/arm63 platforms now have good support for the IEEE floating point standard. So I reimplemented this library based on f32.
