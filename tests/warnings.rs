#[test]
fn save_to_expected_warning() {
    assert!(
        std::env::var("SAVE_TO_EXPECTED").is_err(),
        "Running in save to expected mode"
    );
}

#[test]
fn bincode_deterministic_warning() {
    #[cfg(not(feature = "bincode"))]
    assert!(false, "Disable bincode will skip deterministic test cases");
}
