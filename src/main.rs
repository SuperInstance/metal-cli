//! metal-cli — Unified CLI for all SuperInstance metal library crates.
//!
//! Exposes sheaf-agents, hodge-belief, spectral-graph-agent, ergodic-transport,
//! evolving-sheaf, renormalization-learning, free-probability, and
//! conservation-spectral-topology through a single command-line interface.

#![allow(unused_imports)]

use clap::{Parser, Subcommand};
use serde::Serialize;
use nalgebra;

// ── CLI entry point ─────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name = "metal-cli",
    about = "Unified CLI for SuperInstance metal library crates",
    version
)]
struct Cli {
    /// Output as JSON (for piping/consumption by other tools)
    #[arg(long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compute Cheeger constant on path, cycle, or complete graph
    Cheeger {
        /// Graph type: path, cycle, or complete
        graph_type: String,
        /// Number of vertices
        n: usize,
    },
    /// Compute Fiedler value and vector
    Fiedler {
        /// Graph type: path, cycle, or complete
        graph_type: String,
        /// Number of vertices
        n: usize,
    },
    /// Hodge decomposition of error/belief data from JSON file
    #[command(name = "hodge")]
    HodgeDecompose {
        /// Path to JSON file with edge_values array
        file: String,
    },
    /// Sheaf cohomology (H⁰/H¹) of agent agreement data from JSON
    #[command(name = "sheaf")]
    SheafCohomology {
        /// Path to JSON file with stalk dims and edge restrictions
        file: String,
    },
    /// Forecast the next command in a sequence from JSON
    #[command(name = "forecast")]
    Forecast {
        /// Path to JSON file with command sequence
        file: String,
    },
    /// Check ergodicity of a transition matrix from JSON
    #[command(name = "ergodic")]
    ErgodicCheck {
        /// Path to JSON file with transition matrix
        file: String,
    },
    /// Evolve a sheaf and track spectral gap over time
    #[command(name = "evolve")]
    EvolveSheaf {
        /// Graph topology: path, cycle, or complete
        topology: String,
        /// Number of time steps
        steps: usize,
    },
    /// Coarse-grain a sequence, detect RG fixed points
    #[command(name = "renorm")]
    Renormalize {
        /// Path to JSON file with lattice data
        file: String,
    },
    /// Marchenko–Pastur density at a given λ (aspect ratio)
    #[command(name = "freeprob")]
    FreeProb {
        /// Aspect ratio λ = p/n (features/samples)
        q: f64,
    },
    /// List all available invocations
    List,
}

// ── JSON output envelope ────────────────────────────────────────────────────

#[derive(Serialize)]
struct Output {
    tool: String,
    result: serde_json::Value,
}

fn emit(output: Output, json_flag: bool) {
    if json_flag {
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
    } else {
        // Pretty-print result value
        match &output.result {
            serde_json::Value::String(s) => println!("{}", s),
            serde_json::Value::Array(arr) => {
                for item in arr {
                    match item {
                        serde_json::Value::String(s) => println!("{}", s),
                        serde_json::Value::Number(n) => println!("{}", n),
                        serde_json::Value::Object(o) => println!("{}", serde_json::to_string(o).unwrap()),
                        other => println!("{}", other),
                    }
                }
            }
            serde_json::Value::Object(o) => println!("{}", serde_json::to_string_pretty(o).unwrap()),
            other => println!("{}", other),
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    let json = cli.json;

    match &cli.command {
        Command::Cheeger { graph_type, n } => cmd_cheeger(graph_type, *n, json),
        Command::Fiedler { graph_type, n } => cmd_fiedler(graph_type, *n, json),
        Command::HodgeDecompose { file } => cmd_hodge(file, json),
        Command::SheafCohomology { file } => cmd_sheaf(file, json),
        Command::Forecast { file } => cmd_forecast(file, json),
        Command::ErgodicCheck { file } => cmd_ergodic(file, json),
        Command::EvolveSheaf { topology, steps } => cmd_evolve(topology, *steps, json),
        Command::Renormalize { file } => cmd_renorm(file, json),
        Command::FreeProb { q } => cmd_freeprob(*q, json),
        Command::List => cmd_list(json),
    }
}

// ── Input structure helpers ─────────────────────────────────────────────────

#[derive(serde::Deserialize)]
struct HodgeInput {
    /// Edge values (observations) for the Hodge decomposition
    pub edge_values: Vec<f64>,
    /// Number of vertices in the simplicial complex
    pub n_vertices: usize,
    /// Each element is a 2-element array [u, v]
    pub edges: Vec<[usize; 2]>,
    /// Optional: each element is a 3-element array [u, v, w]
    #[serde(default)]
    pub triangles: Vec<[usize; 3]>,
}

#[derive(serde::Deserialize)]
struct SheafInput {
    /// Stalk dimension at each vertex
    pub stalk_dims: Vec<usize>,
    /// Edge restrictions: each edge has v1, v2, and matrices
    pub edges: Vec<SheafEdgeInput>,
}

#[derive(serde::Deserialize)]
struct SheafEdgeInput {
    pub v1: usize,
    pub v2: usize,
    /// Row-major flattened r1 matrix
    pub r1: Vec<f64>,
    pub r1_rows: usize,
    pub r1_cols: usize,
    /// Row-major flattened r2 matrix
    pub r2: Vec<f64>,
    pub r2_rows: usize,
    pub r2_cols: usize,
}

#[derive(serde::Deserialize)]
struct ErgodicInput {
    /// Transition matrix (row-stochastic), flat row-major array of size n×n
    pub matrix: Vec<f64>,
    pub n: usize,
}

#[derive(serde::Deserialize)]
struct RenormInput {
    /// Linear dimension L (L×L lattice)
    #[allow(non_snake_case)]
    pub L: usize,
    /// Flat row-major array of length L×L
    pub data: Vec<f64>,
    /// Block factor for coarse-graining
    pub b: usize,
    /// Number of RG steps
    pub steps: usize,
}

// ── Graph builder helpers ───────────────────────────────────────────────────

fn build_spectral_graph(graph_type: &str, n: usize) -> spectral_graph_agent::SpectralGraph {
    let mut g = spectral_graph_agent::SpectralGraph::new(n, false);

    match graph_type {
        "path" | "p" => {
            for i in 0..n.saturating_sub(1) {
                g.add_edge(i, i + 1, 1.0).ok();
            }
        }
        "cycle" | "c" => {
            for i in 0..n {
                g.add_edge(i, (i + 1) % n, 1.0).ok();
            }
        }
        "complete" | "k" | "K" => {
            for i in 0..n {
                for j in (i + 1)..n {
                    g.add_edge(i, j, 1.0).ok();
                }
            }
        }
        other => {
            eprintln!("Unknown graph type: {}. Use path, cycle, or complete.", other);
            std::process::exit(1);
        }
    }
    g.finalize();
    g
}

fn build_evolving_graph(topology: &str, n: usize) -> evolving_sheaf::Graph {
    match topology {
        "path" | "p" => evolving_sheaf::Graph::path(n),
        "cycle" | "c" => evolving_sheaf::Graph::cycle(n),
        "complete" | "k" | "K" => evolving_sheaf::Graph::complete(n),
        other => {
            eprintln!("Unknown topology: {}. Use path, cycle, or complete.", other);
            std::process::exit(1);
        }
    }
}

// ── Command implementations ─────────────────────────────────────────────────

fn cmd_cheeger(graph_type: &str, n: usize, json: bool) {
    let g = build_spectral_graph(graph_type, n);
    match g.cheeger_constant() {
        Ok(h) => emit(Output { tool: "cheeger".into(), result: serde_json::json!({"cheeger_constant": h, "graph_type": graph_type, "n": n}) }, json),
        Err(e) => {
            eprintln!("Error computing Cheeger constant: {:?}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_fiedler(graph_type: &str, n: usize, json: bool) {
    let g = build_spectral_graph(graph_type, n);
    match g.fiedler_value() {
        Ok(val) => {
            let vec = g.fiedler_vector().unwrap_or_else(|_| nalgebra::DVector::<f64>::zeros(0));
            let vec_ser: Vec<f64> = vec.iter().copied().collect();
            emit(Output {
                tool: "fiedler".into(),
                result: serde_json::json!({
                    "fiedler_value": val,
                    "fiedler_vector": vec_ser,
                    "graph_type": graph_type,
                    "n": n
                }),
            }, json);
        }
        Err(e) => {
            eprintln!("Error computing Fiedler: {:?}", e);
            std::process::exit(1);
        }
    }
}

fn cmd_hodge(file: &str, json: bool) {
    let content = std::fs::read_to_string(file).unwrap_or_else(|e| {
        eprintln!("Cannot read {}: {}", file, e);
        std::process::exit(1);
    });
    let input: HodgeInput = serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("Invalid JSON in {}: {}", file, e);
        std::process::exit(1);
    });

    let sc = hodge_belief::SimplicialComplex::new(input.n_vertices, input.edges, input.triangles);
    let bs = hodge_belief::BeliefState::from_observations(&sc, &input.edge_values);
    let bh = hodge_belief::BeliefHodge::new(&sc, &bs);

    emit(Output {
        tool: "hodge".into(),
        result: serde_json::json!({
            "exact_norm": bh.hd.exact_norm,
            "coexact_norm": bh.hd.coexact_norm,
            "harmonic_norm": bh.hd.harmonic_norm,
            "exact": bh.evidence(),
            "coexact": bh.coherence(),
            "harmonic": bh.prior(),
        }),
    }, json);
}

fn cmd_sheaf(file: &str, json: bool) {
    let content = std::fs::read_to_string(file).unwrap_or_else(|e| {
        eprintln!("Cannot read {}: {}", file, e);
        std::process::exit(1);
    });
    let input: SheafInput = serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("Invalid JSON in {}: {}", file, e);
        std::process::exit(1);
    });

    let mut sheaf = sheaf_agents::CellularSheaf::new(input.stalk_dims.clone());
    for e in &input.edges {
        let r1 = nalgebra::DMatrix::<f64>::from_row_slice(e.r1_rows, e.r1_cols, &e.r1);
        let r2 = nalgebra::DMatrix::<f64>::from_row_slice(e.r2_rows, e.r2_cols, &e.r2);
        sheaf.add_edge(e.v1, e.v2, r1, r2);
    }

    let tol = 1e-8;
    let h0 = sheaf.h0(tol);
    let h1 = sheaf.h1(tol);
    let gap = sheaf.spectral_gap();
    let h0_basis = sheaf.cohomology_basis(0, tol);
    let h1_basis = sheaf.cohomology_basis(1, tol);

    emit(Output {
        tool: "sheaf".into(),
        result: serde_json::json!({
            "h0_dim": h0,
            "h1_dim": h1,
            "spectral_gap": gap,
            "h0_basis": h0_basis,
            "h1_basis": h1_basis,
        }),
    }, json);
}

fn cmd_forecast(_file: &str, _json: bool) {
    // Placeholder — the renormalization-learning library provides coarse-graining
    // and fixed-point detection for sequences/lattices.
    // A full "forecast" would use RG flow to predict next state.
    emit(Output {
        tool: "forecast".into(),
        result: serde_json::json!({"status": "not_implemented", "note": "Use `renorm` for RG-based sequence analysis"}),
    }, _json);
}

fn cmd_ergodic(file: &str, json: bool) {
    let content = std::fs::read_to_string(file).unwrap_or_else(|e| {
        eprintln!("Cannot read {}: {}", file, e);
        std::process::exit(1);
    });
    let input: ErgodicInput = serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("Invalid JSON in {}: {}", file, e);
        std::process::exit(1);
    });

    let chain = ergodic_transport::MarkovChain::from_flat(input.n, &input.matrix);
    let (ergodic, reason) = chain.is_ergodic();
    let stationary = chain.stationary_distribution();

    emit(Output {
        tool: "ergodic".into(),
        result: serde_json::json!({
            "ergodic": ergodic,
            "reason": reason,
            "stationary_distribution": stationary,
        }),
    }, json);
}

fn cmd_evolve(topology: &str, steps: usize, json: bool) {
    let n = 8; // fixed small graph for quick demo
    let g = build_evolving_graph(topology, n);

    use evolving_sheaf::{SheafConfig, EvolutionMode, NonlinearFn, SpectralGapTracker};

    let cfg = SheafConfig {
        model: EvolutionMode::Linear,
        r0: 1.0,
        alpha: 0.1,
        nonlin: NonlinearFn::Sigmoid,
        nonlin_k: 1.0,
    };

    let tracker = SpectralGapTracker::new(g, cfg, evolving_sheaf::flow_constant);
    let trajectory = tracker.track(0.0, 10.0, steps);

    let points: Vec<serde_json::Value> = trajectory
        .points
        .iter()
        .map(|p| {
            serde_json::json!({
                "t": p.t,
                "gap": p.gap,
                "gap_rate": p.gap_rate,
                "phase_transition": p.phase_transition,
            })
        })
        .collect();

    emit(Output {
        tool: "evolve".into(),
        result: serde_json::json!({
            "topology": topology,
            "min_gap": trajectory.min_gap,
            "max_gap": trajectory.max_gap,
            "n_transitions": trajectory.n_transitions,
            "points": points,
        }),
    }, json);
}

fn cmd_renorm(file: &str, json: bool) {
    let content = std::fs::read_to_string(file).unwrap_or_else(|e| {
        eprintln!("Cannot read {}: {}", file, e);
        std::process::exit(1);
    });
    let input: RenormInput = serde_json::from_str(&content).unwrap_or_else(|e| {
        eprintln!("Invalid JSON in {}: {}", file, e);
        std::process::exit(1);
    });

    use renormalization_learning::{Lattice, CoarseGrainingOperator, RGTransform, FixedPointDetector};

    assert_eq!(input.data.len(), input.L * input.L, "data length must be L×L");

    // Build lattice from input data
    let mut lat = Lattice::new(input.L);
    for y in 0..input.L {
        for x in 0..input.L {
            lat.data[[y, x]] = input.data[y * input.L + x];
        }
    }

    // Run RG flow
    let zeta = 1.0;
    let rgt = RGTransform::new(&lat, input.b, zeta, input.steps);

    let magnetizations = rgt.magnetization_flow();
    let energies = rgt.energy_flow();
    let corr_lengths = rgt.correlation_lengths();

    // Detect fixed point
    let detector = FixedPointDetector::new(1e-6);
    let (converged, fp_steps, fp_mag) = detector.find_fixed_point(&lat, input.b, zeta, input.steps);


    emit(Output {
        tool: "renorm".into(),
        result: serde_json::json!({
            "L": input.L,
            "b": input.b,
            "steps": input.steps,
            "magnetization_flow": magnetizations,
            "energy_flow": energies,
            "correlation_lengths": corr_lengths,
            "fixed_point_converged": converged,
            "fixed_point_steps": fp_steps,
            "fixed_point_magnetization": fp_mag,
        }),
    }, json);
}

fn cmd_freeprob(q: f64, json: bool) {
    // Marchenko-Pastur density evaluation at 100 points
    use free_probability::marchenko_pastur;

    if q <= 0.0 {
        eprintln!("λ must be > 0");
        std::process::exit(1);
    }

    let sigma = 1.0;
    let sqrt_q = q.sqrt();
    let a = sigma * sigma * (1.0 - sqrt_q) * (1.0 - sqrt_q);
    let b = sigma * sigma * (1.0 + sqrt_q) * (1.0 + sqrt_q);

    let n_points = 100;
    let dx = (b - a) / n_points as f64;
    let mut density_points = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let x = a + (i as f64 + 0.5) * dx;
        let rho = marchenko_pastur::density(x, q, sigma);
        density_points.push(serde_json::json!({"x": x, "density": rho}));
    }

    // Compute moments
    let mut moments = vec![0.0_f64; 6];
    marchenko_pastur::moments(q, 6, &mut moments);

    emit(Output {
        tool: "freeprob".into(),
        result: serde_json::json!({
            "lambda": q,
            "support": [a, b],
            "density": density_points,
            "moments": moments,
        }),
    }, json);
}

fn cmd_list(json: bool) {
    let invocations = vec![
        "metal-cli cheeger path 8       ─ Cheeger constant on path graph (n=8)",
        "metal-cli cheeger cycle 8      ─ Cheeger constant on cycle graph (n=8)",
        "metal-cli cheeger complete 8   ─ Cheeger constant on complete graph (n=8)",
        "metal-cli fiedler path 8       ─ Fiedler value & vector on path (n=8)",
        "metal-cli fiedler cycle 8      ─ Fiedler value & vector on cycle (n=8)",
        "metal-cli fiedler complete 8   ─ Fiedler value & vector on complete (n=8)",
        "metal-cli hodge data.json      ─ Hodge decompose [evidence|coherence|prior]",
        "metal-cli sheaf data.json      ─ Sheaf cohomology H⁰/H¹ + spectral gap",
        "metal-cli ergodic data.json    ─ Ergodicity of transition matrix",
        "metal-cli evolve path 100      ─ Evolve sheaf, track spectral gap (100 steps)",
        "metal-cli evolve cycle 100     ─ Evolve sheaf on cycle topology",
        "metal-cli renorm data.json     ─ RG flow: coarse-graining + fixed points",
        "metal-cli freeprob 0.5         ─ Marchenko-Pastur density at λ=0.5",
        "metal-cli freeprob 1.0         ─ Marchenko-Pastur density at λ=1.0",
        "metal-cli freeprob 2.0         ─ Marchenko-Pastur density at λ=2.0",
        "metal-cli list                 ─ This listing",
        "",
        "All commands support --json for machine-readable output.",
        "JSON input files: see https://github.com/SuperInstance/metal-cli#input-formats",
    ];

    emit(Output {
        tool: "list".into(),
        result: serde_json::json!(invocations),
    }, json);
}
