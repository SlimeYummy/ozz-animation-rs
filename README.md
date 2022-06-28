# ozz-animation-rust
A rust runtime library for ozz-animation with fixed-point support.

[Ozz-animation](https://github.com/guillaumeblanc/ozz-animation) is an open source c++ 3d skeletal animation library and toolset. Ozz-animation-rust only implement ozz-animation's runtime part. You should use ozz-animation-rust with ozz-animation's toolset.

In order to introduce support for fixed-point. Ozz-animation-rust does not simply wrap ozz-animation's runtime, but rewrite the full runtime library in rust. So ozz-animation-rust can be use in network scenarios, such as lock-step networking synchronize.

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
| Two bone IK | unimplemented |
| Look-at | unimplemented |
| Foot ik | unimplemented |
| Multi-threading | unsupported |
