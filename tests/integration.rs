use std::process::Command;

/// Helper: run metal-cli with given args, capture stdout
fn metal_cli(args: &[&str]) -> (String, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_metal-cli"))
        .args(args)
        .output()
        .expect("Failed to execute metal-cli");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    (stdout, stderr)
}

// ── Spectral graph tests ───────────────────────────────────────────────────

#[test]
fn test_cheeger_path() {
    let (stdout, stderr) = metal_cli(&["cheeger", "path", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("cheeger_constant"), "output should contain 'cheeger_constant', got: {}", stdout);
}

#[test]
fn test_cheeger_path_json() {
    let (stdout, stderr) = metal_cli(&["--json", "cheeger", "path", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "cheeger");
    assert!(parsed["result"]["cheeger_constant"].as_f64().is_some());
}

#[test]
fn test_cheeger_cycle() {
    let (stdout, stderr) = metal_cli(&["cheeger", "cycle", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("cheeger_constant"));
}

#[test]
fn test_cheeger_complete() {
    let (stdout, stderr) = metal_cli(&["cheeger", "complete", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("cheeger_constant"));
}

#[test]
fn test_fiedler_path() {
    let (stdout, stderr) = metal_cli(&["fiedler", "path", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("fiedler_value") || stdout.contains("fiedler"));
}

#[test]
fn test_fiedler_cycle() {
    let (stdout, stderr) = metal_cli(&["fiedler", "cycle", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("fiedler_value"));
}

#[test]
fn test_fiedler_complete() {
    let (stdout, stderr) = metal_cli(&["fiedler", "complete", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("fiedler_value"));
}

#[test]
fn test_fiedler_json() {
    let (stdout, stderr) = metal_cli(&["--json", "fiedler", "cycle", "8"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "fiedler");
    assert!(parsed["result"]["fiedler_value"].as_f64().is_some());
    assert!(parsed["result"]["fiedler_vector"].is_array());
}

// ── Hodge decomposition tests ───────────────────────────────────────────────

#[test]
fn test_hodge_decompose_triangle() {
    let json_data = r#"{
        "n_vertices": 3,
        "edges": [[0,1],[1,2],[0,2]],
        "triangles": [[0,1,2]],
        "edge_values": [1.0, 0.0, -1.0]
    }"#;
    let tmp = std::env::temp_dir().join("hodge_test_triangle.json");
    std::fs::write(&tmp, json_data).unwrap();
    let path = tmp.to_str().unwrap();
    let (stdout, stderr) = metal_cli(&["hodge", path]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("exact_norm") || stdout.contains("harmonic_norm"));
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_hodge_json_output() {
    let json_data = r#"{
        "n_vertices": 3,
        "edges": [[0,1],[1,2],[0,2]],
        "triangles": [],
        "edge_values": [0.5, 0.5, 0.5]
    }"#;
    let tmp = std::env::temp_dir().join("hodge_test_json.json");
    std::fs::write(&tmp, json_data).unwrap();
    let path = tmp.to_str().unwrap();
    let (stdout, stderr) = metal_cli(&["--json", "hodge", path]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "hodge");
    assert!(parsed["result"]["exact_norm"].as_f64().is_some());
    std::fs::remove_file(&tmp).ok();
}

// ── Sheaf cohomology tests ──────────────────────────────────────────────────

#[test]
fn test_sheaf_identity_triangle() {
    let json_data = r#"{
        "stalk_dims": [2, 2, 2],
        "edges": [
            {"v1": 0, "v2": 1, "r1_rows": 2, "r1_cols": 2, "r1": [1,0,0,1], "r2_rows": 2, "r2_cols": 2, "r2": [1,0,0,1]},
            {"v1": 1, "v2": 2, "r1_rows": 2, "r1_cols": 2, "r1": [1,0,0,1], "r2_rows": 2, "r2_cols": 2, "r2": [1,0,0,1]},
            {"v1": 0, "v2": 2, "r1_rows": 2, "r1_cols": 2, "r1": [1,0,0,1], "r2_rows": 2, "r2_cols": 2, "r2": [1,0,0,1]}
        ]
    }"#;
    let tmp = std::env::temp_dir().join("sheaf_test_triangle.json");
    std::fs::write(&tmp, json_data).unwrap();
    let path = tmp.to_str().unwrap();
    let (stdout, stderr) = metal_cli(&["--json", "sheaf", path]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "sheaf");
    assert!(parsed["result"]["h0_dim"].as_u64().is_some());
    assert!(parsed["result"]["h1_dim"].as_u64().is_some());
    std::fs::remove_file(&tmp).ok();
}

// ── Ergodicity tests ────────────────────────────────────────────────────────

#[test]
fn test_ergodic_two_state_chain() {
    let json_data = r#"{
        "n": 2,
        "matrix": [0.9, 0.1, 0.2, 0.8]
    }"#;
    let tmp = std::env::temp_dir().join("ergodic_test_2.json");
    std::fs::write(&tmp, json_data).unwrap();
    let path = tmp.to_str().unwrap();
    let (stdout, stderr) = metal_cli(&["--json", "ergodic", path]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "ergodic");
    assert!(parsed["result"]["ergodic"].is_boolean());
    std::fs::remove_file(&tmp).ok();
}

// ── Evolve sheaf tests ──────────────────────────────────────────────────────

#[test]
fn test_evolve_path() {
    let (stdout, stderr) = metal_cli(&["--json", "evolve", "path", "5"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "evolve");
    assert!(parsed["result"]["min_gap"].as_f64().is_some());
    assert!(parsed["result"]["points"].is_array());
    assert!(parsed["result"]["points"].as_array().unwrap().len() >= 6);
}

#[test]
fn test_evolve_cycle() {
    let (stdout, stderr) = metal_cli(&["--json", "evolve", "cycle", "5"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "evolve");
    assert!(parsed["result"]["min_gap"].as_f64().is_some());
}

// ── RG / Renormalization tests ──────────────────────────────────────────────

#[test]
fn test_renorm_uniform_lattice() {
    // Uniform 8x8 lattice — should be near a fixed point
    let n = 8;
    let data: Vec<f64> = vec![0.5_f64; n * n];
    let json_data = serde_json::json!({
        "L": n,
        "data": data,
        "b": 2,
        "steps": 2
    });
    let tmp = std::env::temp_dir().join("renorm_uniform.json");
    std::fs::write(&tmp, serde_json::to_string(&json_data).unwrap()).unwrap();
    let path = tmp.to_str().unwrap();
    let (stdout, stderr) = metal_cli(&["--json", "renorm", path]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "renorm");
    assert!(parsed["result"]["magnetization_flow"].is_array());
    std::fs::remove_file(&tmp).ok();
}

// ── Free probability tests ──────────────────────────────────────────────────

#[test]
fn test_freeprob_lambda_half() {
    let (stdout, stderr) = metal_cli(&["--json", "freeprob", "0.5"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "freeprob");
    assert!(parsed["result"]["lambda"].as_f64().unwrap() - 0.5 < 1e-10);
    assert!(parsed["result"]["density"].is_array());
    assert!(parsed["result"]["moments"].is_array());
}

#[test]
fn test_freeprob_lambda_one() {
    let (stdout, stderr) = metal_cli(&["--json", "freeprob", "1.0"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "freeprob");
    assert_eq!(parsed["result"]["lambda"].as_f64().unwrap(), 1.0);
}

#[test]
fn test_freeprob_lambda_two() {
    let (stdout, stderr) = metal_cli(&["--json", "freeprob", "2.0"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "freeprob");
    assert_eq!(parsed["result"]["lambda"].as_f64().unwrap(), 2.0);
}

// ── List test ───────────────────────────────────────────────────────────────

#[test]
fn test_list() {
    let (stdout, stderr) = metal_cli(&["list"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    assert!(stdout.contains("cheeger") || stdout.contains("fiedler"));
}

#[test]
fn test_list_json() {
    let (stdout, stderr) = metal_cli(&["--json", "list"]);
    assert!(stderr.is_empty(), "stderr: {}", stderr);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    assert_eq!(parsed["tool"], "list");
    assert!(parsed["result"].is_array());
}
