mod base;
mod blend;
mod playback;
mod two_bone_ik;

use bevy::prelude::*;
use bevy::pbr::NotShadowCaster;
use bevy::render::mesh::PrimitiveTopology;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::tasks::futures_lite::future;
use bevy::tasks::{self, AsyncComputeTaskPool, Task};

use crate::base::*;
use crate::blend::OzzBlend;
use crate::playback::OzzPlayback;
use crate::two_bone_ik::OzzTwoBoneIK;

const BONE_COUNT: usize = 256;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Startup, setup_ui)
        .add_systems(Update, update_ozz_animation)
        .add_systems(Update, update_camera)
        .add_systems(Update, update_bones)
        .add_systems(Update, draw_spines)
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
}

const OZZ_TYPES: [(OzzType, &'static str); 3] = [
    (OzzType::Playback, "Playback"),
    (OzzType::Blend, "Blend"),
    (OzzType::TwoBoneIK, "TwoBoneIK"),
];

impl OzzComponent {
    fn load(&mut self) {
        let thread_pool = AsyncComputeTaskPool::get();
        let task = match OZZ_TYPES[self.typ].0 {
            OzzType::Playback => thread_pool.spawn(OzzPlayback::new()),
            OzzType::Blend => thread_pool.spawn(OzzBlend::new()),
            OzzType::TwoBoneIK => thread_pool.spawn(OzzTwoBoneIK::new()),
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

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut oc = OzzComponent::default();
    oc.load();
    commands.spawn(oc);

    // ground
    commands.spawn(PbrBundle {
        mesh: meshes.add(Rectangle::new(4.0, 16.0)),
        material: materials.add(Color::rgb(0.35, 0.56, 0.45)),
        transform: Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2))
            .with_translation(Vec3::new(0.0, 0.0, 5.0)),
        ..default()
    });

    // bones
    let bone_mesh = meshes.add(build_bone_mesh());
    let bone_material = materials.add(Color::WHITE);
    for i in 0..BONE_COUNT {
        commands.spawn((
            PbrBundle {
                mesh: bone_mesh.clone(),
                material: bone_material.clone(),
                transform: Transform::from_xyz(0.0, 0.0, 0.0),
                visibility: Visibility::Hidden,
                ..default()
            },
            BoneIndex(i),
        ));
    }

    // camera
    commands.spawn((Camera3dBundle {
        transform: Transform::from_xyz(1.5, 1.0, 3.0).looking_at(Vec3::new(0.0, 1.0, -0.0), Vec3::Y),
        ..default()
    }, FogSettings {
        color: Color::rgba(0.35, 0.48, 0.66, 1.0),
        directional_light_color: Color::rgba(1.0, 0.95, 0.85, 0.5),
        directional_light_exponent: 30.0,
        falloff: FogFalloff::from_visibility_colors(
            15.0, // distance in world units up to which objects retain visibility (>= 5% contrast)
            Color::rgb(0.35, 0.5, 0.66), // atmospheric extinction color (after light is lost due to absorption by atmospheric particles)
            Color::rgb(0.8, 0.844, 1.0), // atmospheric inscattering color (light gained due to scattering from the sun)
        ),
    }));

    // Sun
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            color: Color::rgb(0.98, 0.95, 0.82),
            illuminance: 3000.0,
            shadows_enabled: true,
            shadow_depth_bias: 0.05,
            shadow_normal_bias: 0.9,
            ..default()
        },
        transform: Transform::from_xyz(-3.0, 2.0, -4.0)
            .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
        ..default()
    });

    // Sky
    commands.spawn((
        PbrBundle {
            mesh: meshes.add(Cuboid::new(2.0, 1.0, 1.0)),
            material: materials.add(StandardMaterial {
                base_color: Color::hex("888888").unwrap(),
                unlit: true,
                cull_mode: None,
                ..default()
            }),
            transform: Transform::from_scale(Vec3::splat(20.0)),
            ..default()
        },
        NotShadowCaster,
    ));

    // light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            shadows_enabled: false,
            color: Color::WHITE,
            ..default()
        },
        transform: Transform::from_xyz(0.0, 5.0, 3.0),
        ..default()
    });
}

fn update_ozz_animation(keycode: Res<ButtonInput<KeyCode>>, mut oc_query: Query<&mut OzzComponent>, mut text_query: Query<&mut Text, With<DemoName>>, time: Res<Time>) {
    let mut oc = oc_query.iter_mut().last().unwrap(); // only one OzzComponent
    oc.poll();
    
    if keycode.just_pressed(KeyCode::Space) {
        oc.load_next();

        let mut text = text_query.iter_mut().last().unwrap();
        text.sections[0].value = format!("Demo: {}", OZZ_TYPES[oc.typ].1);
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
        if bone_trans.len() > 0 {
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
        if spine_trans.len() > 0 {
            for trans in spine_trans {
                draw_gizmos(&mut gizmos, trans);
            }
        }
    }
}

#[rustfmt::skip]
fn build_bone_mesh() -> Mesh {
    let c = vec![
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.2, 0.1, 0.1),
        Vec3::new(0.2, 0.1, -0.1),
        Vec3::new(0.2, -0.1, -0.1),
        Vec3::new(0.2, -0.1, 0.1),
        Vec3::new(0.0, 0.0, 0.0),
    ];
    let n = vec![
        Vec3::cross(c[2] - c[1], c[2] - c[0]).normalize(),
        Vec3::cross(c[1] - c[2], c[1] - c[5]).normalize(),
        Vec3::cross(c[3] - c[2], c[3] - c[0]).normalize(),
        Vec3::cross(c[2] - c[3], c[2] - c[5]).normalize(),
        Vec3::cross(c[4] - c[3], c[4] - c[0]).normalize(),
        Vec3::cross(c[3] - c[4], c[3] - c[5]).normalize(),
        Vec3::cross(c[1] - c[4], c[1] - c[0]).normalize(),
        Vec3::cross(c[4] - c[1], c[4] - c[5]).normalize(),
    ];
    
    let mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
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
    ]);
    return mesh;
}

fn draw_gizmos(gizmos: &mut Gizmos, trans: &OzzTransform) {
    let normal_x = trans.rotation.mul_vec3(Vec3::X).normalize();
    let normal_y = trans.rotation.mul_vec3(Vec3::Y).normalize();
    let normal_z = trans.rotation.mul_vec3(Vec3::Z).normalize();
    gizmos.circle(
        trans.position,
        Direction3d::new_unchecked(normal_x),
        trans.scale * 0.25,
        Color::rgba(1.0, 0.1, 0.1, 0.4),
    );
    gizmos.circle(
        trans.position,
        Direction3d::new_unchecked(normal_y),
        trans.scale * 0.25,
        Color::rgba(0.1, 1.0, 0.1, 0.4),
    );
    gizmos.circle(
        trans.position,
        Direction3d::new_unchecked(normal_z),
        trans.scale * 0.25,
        Color::rgba(0.1, 0.1, 1.0, 0.4),
    );
}

#[derive(Component)]
struct DemoName;

fn setup_ui(mut commands: Commands) {
    commands.spawn((TextBundle::from_section(
        format!("Demo: {}", OZZ_TYPES[0].1).as_str(),
        TextStyle {
            font_size: 32.0,
            ..default()
        },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        left: Val::Px(5.0),
        top: Val::Px(5.0),
        ..default()
    }), DemoName));

    commands.spawn(TextBundle::from_section(
        "Press space to switch demo",
        TextStyle {
            font_size: 32.0,
            ..default()
        },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        left: Val::Px(5.0),
        top: Val::Px(40.0),
        ..default()
    }));
}
