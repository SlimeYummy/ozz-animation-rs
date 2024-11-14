#![doc(hidden)]

// Only for wasm test in NodeJS environment.

use js_sys::{Error, Uint8Array};
use wasm_bindgen::prelude::*;

#[wasm_bindgen(module = "fs")]
extern "C" {
    #[wasm_bindgen(js_name = readFileSync, catch)]
    fn read_file_sync(path: &str) -> Result<JsValue, Error>;
}

pub fn read_file(path: &str) -> Result<Vec<u8>, Error> {
    let buf = read_file_sync(path).unwrap();
    let array = Uint8Array::new(&buf);
    Ok(array.to_vec())
}
