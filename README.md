# ozz-rust
A rust runtime library for ozz-animation with fixed-point support.

[Ozz-animation](https://github.com/guillaumeblanc/ozz-animation) is an open source c++ 3d skeletal animation library and toolset. Ozz-rust only implement ozz-animation's runtime party. You should use ozz-rust with ozz-animation's toolset.

In order to introduce support for fixed-point. Ozz-rust dose not simply wrap ozz-animation's runtime, but rewrite the full runtime library in rust. So ozz-rust can be use in network scenarios, such as lock-step networking synchronize.

Ozz-animation Features:
- Animation playback
- Joint attachment
- Animation blending
- Partial animations blending (unimplemented)
- Additive blending (unimplemented)
- Skinning (unsupported)
- User channels (unimplemented)
- Baked physic simulation (unsupported)
- Keyframe reduction (unsupported)
- Offline libraries usage (unsupported)
- Two bone IK (unsupported)
- Look-at (unsupported)
- Foot ik (unsupported)
- Multi-threading (unsupported)
