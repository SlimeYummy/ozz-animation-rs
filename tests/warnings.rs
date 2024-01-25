#[test]
fn save_to_expected_warning() {
    assert!(
        std::env::var("SAVE_TO_EXPECTED").is_err(),
        "Running in save to expected mode"
    );
}

#[test]
fn rkyv_deterministic_warning() {
    #[cfg(not(feature = "rkyv"))]
    assert!(false, "Disable rkyv will skip deterministic test cases");
}
