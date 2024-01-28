use std::env::consts::{ARCH, OS};
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::sync::OnceLock;
use std::{env, mem};

// f16 -> f32
// ignore overflow, infinite, NaN
pub fn f16(f: f32) -> u16 {
    let n = unsafe { mem::transmute::<f32, u32>(f) };
    if (n & 0x7FFFFFFF) == 0 {
        return (n >> 16) as u16;
    }
    let sign = (n >> 16) & 0x8000;
    let expo = (((n & 0x7f800000) - 0x38000000) >> 13) & 0x7c00;
    let base = (n >> 13) & 0x03ff;
    return (sign | expo | base) as u16;
}

static FOLDER: OnceLock<()> = OnceLock::new();

#[cfg(feature = "rkyv")]
pub fn compare_with_rkyv<T>(folder: &str, name: &str, data: &T) -> Result<(), Box<dyn Error>>
where
    T: PartialEq + rkyv::Serialize<rkyv::ser::serializers::AllocSerializer<30720>>,
    T::Archived: rkyv::Deserialize<T, rkyv::Infallible>,
{
    use rkyv::ser::Serializer;
    use rkyv::{AlignedVec, Deserialize};

    FOLDER.get_or_init(|| {
        fs::create_dir_all(format!("./expected/{}", folder)).unwrap();
        fs::create_dir_all(format!("./output/{}", folder)).unwrap();
    });

    let to_expected = env::var("SAVE_TO_EXPECTED").is_ok();

    let mut serializer = rkyv::ser::serializers::AllocSerializer::<30720>::default();
    serializer.serialize_value(data)?;
    let buf = serializer.into_serializer().into_inner();
    let path = if to_expected {
        format!("./expected/{0}/{1}.rkyv", folder, name)
    } else {
        format!("./output/{0}/{1}_{2}_{3}.rkyv", folder, name, OS, ARCH)
    };
    let mut file = File::create(path)?;
    file.write_all(&buf)?;

    if !to_expected {
        let path = format!("./expected/{0}/{1}.rkyv", folder, name);
        let mut file = File::open(&path)?;
        let size = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut buf = AlignedVec::with_capacity(size as usize);
        buf.extend_from_reader(&mut file)?;

        let archived = unsafe { rkyv::archived_root::<T>(&buf) };
        let mut deserializer = rkyv::Infallible::default();
        let expected = archived.deserialize(&mut deserializer)?;
        if data != &expected {
            return Err(format!("compare_with_rkyv({})", path).into());
        }
    }
    return Ok(());
}

#[cfg(not(feature = "rkyv"))]
pub fn compare_with_rkyv<T>(_folder: &str, _name: &str, _data: &T) -> Result<(), Box<dyn Error>> {
    return Ok(());
}
