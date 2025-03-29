mod base;
mod blend;
mod playback;
mod root_motion;
mod two_bone_ik;

use bevy::pbr::NotShadowCaster;
use bevy::prelude::*;
use bevy::render::mesh::PrimitiveTopology;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{self, AsyncComputeTaskPool, Task};

use crate::base::*;
use crate::blend::OzzBlend;
use crate::playback::OzzPlayback;
use crate::root_motion::OzzRootMotion;
use crate::two_bone_ik::OzzTwoBoneIK;

const BONE_COUNT: usize = 256;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, (setup, setup_ui))
        .add_systems(Update, (update_ozz_animation, update_camera, update_bones, draw_spines))
        .run();
}

#[derive(Default, Component)]
struct OzzComponent {
    task: Option<Task<Box<dyn OzzExample>>>,
    example: Option<Box<dyn OzzExample>>,
    typ: usize,
}

#[derive(Debug, Clone, Copy)]
enum OzzType {
    Playback,
    Blend,
    TwoBoneIK,
    RootMotion,
}

const OZZ_TYPES: [(OzzType, &str); 4] = [
    (OzzType::Playback, "Playback"),
    (OzzType::Blend, "Blend"),
    (OzzType::TwoBoneIK, "TwoBoneIK"),
    (OzzType::RootMotion, "RootMotion"),
];

impl OzzComponent {
    fn load(&mut self) {
        let thread_pool = AsyncComputeTaskPool::get();
        let task = match OZZ_TYPES[self.typ].0 {
            OzzType::Playback => thread_pool.spawn(OzzPlayback::new()),
            OzzType::Blend => thread_pool.spawn(OzzBlend::new()),
            OzzType::TwoBoneIK => thread_pool.spawn(OzzTwoBoneIK::new()),
            OzzType::RootMotion => thread_pool.spawn(OzzRootMotion::new()),
        };
        self.task = Some(task);
        self.example = None;
    }

    fn load_next(&mut self) {
        self.typ = (self.typ + 1) % OZZ_TYPES.len();
        self.load();
    }

    fn poll(&mut self) {
        if let Some(task) = &mut self.task {
            if let Some(example) = tasks::block_on(future::poll_once(task)) {
                self.example = Some(example);
                self.task = None;
            }
        }
    }
}

#[derive(Component)]
struct BoneIndex(usize);

fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<StandardMaterial>>) {
    let mut oc = OzzComponent::default();
    oc.load();
    commands.spawn(oc);

    // ground
    commands.spawn((
        Mesh3d(meshes.add(Rectangle::new(20.0, 30.0))),
        MeshMaterial3d(materials.add(Color::srgb(1.0, 0.96, 0.95))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .with_translation(Vec3::new(-2.0, 0.0, -5.0)),
    ));

    // bones
    let bone_mesh = meshes.add(build_bone_mesh());
    let bone_material = materials.add(Color::srgb(0.68, 0.68, 0.8));
    for i in 0..BONE_COUNT {
        commands.spawn((
            Mesh3d(bone_mesh.clone()),
            MeshMaterial3d(bone_material.clone()),
            Transform::from_xyz(0.0, 0.0, 0.0),
            Visibility::Hidden,
            BoneIndex(i),
        ));
    }

    // camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(1.5, 1.0, 3.0).looking_at(Vec3::new(0.0, 1.0, -0.0), Vec3::Y),
    ));

    // Sky
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.4, 0.61, 0.98),
            unlit: true,
            cull_mode: None,
            ..default()
        })),
        Transform::from_scale(Vec3::splat(30.0)),
        NotShadowCaster,
    ));

    // Sun
    commands.spawn((
        DirectionalLight {
            color: Color::WHITE,
            illuminance: 10000.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.05,
            shadow_normal_bias: 0.9,
        },
        Transform::from_xyz(-3.0, 1.6, -4.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
    ));

    // light
    commands.spawn((
        PointLight {
            shadows_enabled: false,
            color: Color::WHITE,
            ..default()
        },
        Transform::from_xyz(0.0, 5.0, 3.0),
    ));
}

fn update_ozz_animation(
    keycode: Res<ButtonInput<KeyCode>>,
    mut oc_query: Query<&mut OzzComponent>,
    text_query: Query<Entity, With<DemoName>>,
    mut writer: TextUiWriter,
    time: Res<Time>,
) {
    let mut oc = oc_query.iter_mut().last().unwrap(); // only one OzzComponent
    oc.poll();

    if keycode.just_pressed(KeyCode::Space) {
        oc.load_next();

        let text_entity = text_query.iter().last().unwrap();
        *writer.text(text_entity, 0) = format!("Demo: {}", OZZ_TYPES[oc.typ].1);
    }

    if let Some(example) = &mut oc.example {
        example.update(*time);
    }
}

fn update_camera(mut query: Query<&mut Transform, With<Camera3d>>, oc_query: Query<&OzzComponent>) {
    let oc = oc_query.iter().last().unwrap(); // only one OzzComponent
    if let Some(example) = &oc.example {
        // only one OzzComponent
        let root = example.root();
        let target = Vec3::new(root.w_axis.x, 1.0, root.w_axis.z);

        let pos = target + Vec3::new(1.5, 1.0, 3.0);
        for mut transform in query.iter_mut() {
            *transform = Transform::from_translation(pos).looking_at(target, Vec3::Y);
        }
    }
}

fn update_bones(mut query: Query<(&mut Transform, &mut Visibility, &BoneIndex)>, oc: Query<&OzzComponent>) {
    if let Some(example) = &oc.iter().last().unwrap().example {
        // only one OzzComponent
        let bone_trans = example.bone_trans();
        if !bone_trans.is_empty() {
            for (mut transform, mut visibility, idx) in query.iter_mut() {
                if idx.0 < bone_trans.len() {
                    *visibility = Visibility::Visible;
                    transform.translation = bone_trans[idx.0].position;
                    transform.rotation = bone_trans[idx.0].rotation;
                    transform.scale = Vec3::splat(bone_trans[idx.0].scale);
                } else {
                    *visibility = Visibility::Hidden;
                }
            }
        }
    }
}

fn draw_spines(mut gizmos: Gizmos, oc: Query<&OzzComponent>) {
    if let Some(example) = &oc.iter().last().unwrap().example {
        // only one OzzComponent
        let spine_trans = example.spine_trans();
        if !spine_trans.is_empty() {
            for trans in spine_trans {
                draw_gizmos(&mut gizmos, trans);
            }
        }
    }
    draw_ground(&mut gizmos);
}

#[rustfmt::skip]
fn build_bone_mesh() -> Mesh {
    let c = [Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.2, 0.1, 0.1),
        Vec3::new(0.2, 0.1, -0.1),
        Vec3::new(0.2, -0.1, -0.1),
        Vec3::new(0.2, -0.1, 0.1),
        Vec3::new(0.0, 0.0, 0.0)];
    let n = [Vec3::cross(c[2] - c[1], c[2] - c[0]).normalize(),
        Vec3::cross(c[1] - c[2], c[1] - c[5]).normalize(),
        Vec3::cross(c[3] - c[2], c[3] - c[0]).normalize(),
        Vec3::cross(c[2] - c[3], c[2] - c[5]).normalize(),
        Vec3::cross(c[4] - c[3], c[4] - c[0]).normalize(),
        Vec3::cross(c[3] - c[4], c[3] - c[5]).normalize(),
        Vec3::cross(c[1] - c[4], c[1] - c[0]).normalize(),
        Vec3::cross(c[4] - c[1], c[4] - c[5]).normalize()];
    
    
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, vec![
        c[0], c[2], c[1],
        c[5], c[1], c[2],
        c[0], c[3], c[2],
        c[5], c[2], c[3],
        c[0], c[4], c[3],
        c[5], c[3], c[4],
        c[0], c[1], c[4],
        c[5], c[4], c[1],
    ])
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![
        n[0], n[0], n[0],
        n[1], n[1], n[1],
        n[2], n[2], n[2],
        n[3], n[3], n[3],
        n[4], n[4], n[4],
        n[5], n[5], n[5],
        n[6], n[6], n[6],
        n[7], n[7], n[7],
    ])
}

fn draw_ground(gizmos: &mut Gizmos) {
    for i in -12..=8 {
        gizmos.line(
            Vec3::new(i as f32, 0.0, -20.0),
            Vec3::new(i as f32, 0.0, 10.0),
            Color::srgba(0.25, 0.25, 0.25, 0.4),
        );
    }
    for i in -20..=10 {
        gizmos.line(
            Vec3::new(-12.0, 0.0, i as f32),
            Vec3::new(8.0, 0.0, i as f32),
            Color::srgba(0.25, 0.25, 0.25, 0.4),
        );
    }
}

fn draw_gizmos(gizmos: &mut Gizmos, trans: &OzzTransform) {
    let normal_x = trans.rotation.mul_vec3(Vec3::X).normalize();
    let normal_y = trans.rotation.mul_vec3(Vec3::Y).normalize();
    let normal_z = trans.rotation.mul_vec3(Vec3::Z).normalize();
    gizmos.circle(
        Isometry3d::new(trans.position, Quat::from_scaled_axis(normal_x)),
        trans.scale * 0.25,
        Color::srgba(1.0, 0.1, 0.1, 0.5),
    );

    gizmos.circle(
        Isometry3d::new(trans.position, Quat::from_scaled_axis(normal_y)),
        trans.scale * 0.25,
        Color::srgba(0.1, 1.0, 0.1, 0.5),
    );
    gizmos.circle(
        Isometry3d::new(trans.position, Quat::from_scaled_axis(normal_z)),
        trans.scale * 0.25,
        Color::srgba(0.1, 0.1, 1.0, 0.5),
    );
}

#[derive(Component)]
struct DemoName;

fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            Text::new(format!("Demo: {}", OZZ_TYPES[0].1).as_str()),
            TextFont::from_font_size(32.0),
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(5.0),
                top: Val::Px(5.0),
                ..default()
            },
            DemoName,
        ))
        .with_child((
            TextSpan::new("\nPress space to switch demo"),
            TextFont::from_font_size(32.0),
        ));
}
