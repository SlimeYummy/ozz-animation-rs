use bevy::prelude::*;
use ozz_animation_rs::{Archive, OzzError};
use std::io::Cursor;

#[derive(Debug, Clone, Copy)]
pub struct OzzTransform {
    pub scale: f32,
    pub rotation: Quat,
    pub position: Vec3,
}

pub trait OzzExample
where
    Self: Send + Sync,
{
    fn update(&mut self, time: Time);
    fn root(&self) -> Mat4;
    fn bone_trans(&self) -> &[OzzTransform];
    fn spine_trans(&self) -> &[OzzTransform];
}

#[cfg(not(target_arch = "wasm32"))]
pub async fn load_archive(path: &str) -> Result<Archive<Cursor<Vec<u8>>>, OzzError> {
    use std::fs::File;
    use std::io::Read;

    let full_path = format!("../resource/{}", path);
    let mut file = File::open(full_path)?; // Not async! For compatible with reqwest.
    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    return Archive::from_vec(buf);
}

// #[cfg(target_arch = "wasm32")]
// pub async fn load_archive(path: &str) -> Archive<std::io::Cursor<Vec<u8>>> {
//     let url = format!("http://127.0.0.1:8080/{}", path);
//     let buf = reqwest::get(url).await.unwrap().bytes().await.unwrap().to_vec();
//     return Archive::from_vec(buf).unwrap();
// }
