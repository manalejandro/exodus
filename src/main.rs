use std::path::PathBuf;
use std::sync::Arc;

use clap::{Args, Parser, Subcommand};

use exodus::config::ExodusConfig;
use exodus::coordinator::ExodusCoordinator;
use exodus::identity::load_or_create_identity;
use exodus::ledger::ChainStore;
use exodus::network::{LocalTransport, TcpTransport, Transport};

#[derive(Parser)]
#[command(name = "exodus", version = exodus::VERSION, about = "Free distributed compute network")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create your identity + data dir
    Init,
    /// Show the effective runtime configuration
    Config,
    /// Print node status
    Status {
        #[arg(long, value_name = "DIR")]
        data_dir: Option<PathBuf>,
        #[arg(long)]
        json: bool,
    },
    /// Run a 5-node headless simulation
    Simulate {
        #[arg(long, default_value_t = 5)]
        nodes: usize,
        #[arg(long, default_value_t = 40)]
        ticks: usize,
        #[arg(long)]
        seed: Option<u64>,
        #[arg(long, default_value_t = 2)]
        claims_per_tick: usize,
    },
    /// Join the network as a full node
    Run {
        #[command(flatten)]
        node: NodeArgs,
        /// Serve the REST API + dashboard
        #[arg(long)]
        api: bool,
    },
}

#[derive(Args)]
struct NodeArgs {
    #[arg(long, value_name = "DIR")]
    data_dir: Option<PathBuf>,
    #[arg(long, value_name = "HOST")]
    node_host: Option<String>,
    #[arg(long, value_name = "PORT")]
    node_port: Option<u16>,
    #[arg(long, value_name = "HOST:PORT")]
    peer: Vec<String>,
    /// Disable UDP multicast discovery
    #[arg(long)]
    no_discover: bool,
}

fn config_with(data_dir: Option<PathBuf>, node_host: Option<String>, node_port: Option<u16>) -> ExodusConfig {
    let mut cfg = exodus::config::config_from_env();
    if let Some(d) = data_dir {
        cfg.data_dir = d;
    }
    if let Some(h) = node_host {
        cfg.node_host = h;
    }
    if let Some(p) = node_port {
        cfg.node_port = p;
    }
    cfg
}

fn offline_coordinator(cfg: &ExodusConfig) -> Arc<ExodusCoordinator> {
    let identity = load_or_create_identity(&cfg.identity_path()).expect("identity");
    let store = Arc::new(ChainStore::open(&cfg.ledger_path()).expect("ledger"));
    let transport: Arc<dyn Transport> = Arc::new(LocalTransport::new());
    ExodusCoordinator::new(identity, store, transport, cfg.clone(), None)
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Init => {
            let cfg = exodus::config::config_from_env();
            let identity = load_or_create_identity(&cfg.identity_path()).expect("identity");
            println!("Identity ready: {}", identity.node_id);
            println!("  key    : {}", cfg.identity_path().display());
            println!("  ledger : {}", cfg.ledger_path().display());
        }
        Command::Config => {
            let cfg = exodus::config::config_from_env();
            println!("Exodus runtime configuration");
            println!("===========================");
            println!("  data dir                    : {}", cfg.data_dir.display());
            println!("  epoch (checkpoint period)   : {}s", cfg.epoch_seconds);
            println!("  election timeout            : {}s", cfg.election_timeout_seconds);
            println!("  quorum model                : {}", if cfg.byzantine { "byzantine (2f+1)" } else { "majority" });
            println!("  credits per compute unit    : {}", cfg.credits_per_cu);
            println!("  reward curve exponent       : {}", cfg.reward_diminishing);
            println!("  credit half-life            : {}s", cfg.credit_halflife_seconds);
            println!("  free AI-time quota          : {}s/day", cfg.free_quota_seconds);
            println!("  seconds of AI time / credit : {}", cfg.seconds_per_credit);
            println!("  networking                  : tcp {}:{}", cfg.node_host, cfg.node_port);
            println!("  api                         : {}:{}", cfg.api_host, cfg.api_port);
            println!("  distributed inference       : {} (gather timeout {}s)", if cfg.distributed_inference { "enabled" } else { "disabled" }, cfg.distributed_timeout_seconds);
            println!("  model dir                   : {}", cfg.models_dir().display());
            let gpu = exodus::gpu::detect(cfg.gpu_layers);
            println!(
                "  gpu                         : {} ({} device{}, layers {})",
                if gpu.available { "available" } else { "none" },
                gpu.devices.len(),
                if gpu.devices.len() == 1 { "" } else { "s" },
                cfg.gpu_layers.map(|v| v.to_string()).unwrap_or_else(|| "auto".to_string()),
            );
            if gpu.available {
                for d in &gpu.devices {
                    println!("      - {} ({} MB)", d.name, d.memory_total_mb);
                }
            }
        }
        Command::Status { data_dir, json } => {
            let cfg = config_with(data_dir, None, None);
            let coord = offline_coordinator(&cfg);
            let status = coord.status();
            if json {
                println!("{}", serde_json::to_string_pretty(&status).unwrap());
            } else {
                let credits = status.get("credits").cloned().unwrap_or(serde_json::json!({}));
                println!("node     : {}", status["node_id"]);
                println!("ledger   : height={} verified={}", status["ledger_height"], status["verified_chain"]);
                println!("sealer   : {} (view {})", status["sealer"], status["view"]);
                let gpu_devices = status["gpu"]["devices"].as_array().map(|a| a.len()).unwrap_or(0);
                println!("gpu      : available={} devices={}", status["gpu"]["available"], gpu_devices);
                println!("pending  : {} claims", status["pending_claims"]);
                println!(
                    "credits  : {:.4} (tier {}, {:.1}s AI time)",
                    credits["credits"].as_f64().unwrap_or(0.0),
                    credits["priority_tier"].as_i64().unwrap_or(0),
                    credits["ai_time_seconds"].as_f64().unwrap_or(0.0),
                );
            }
            coord.close();
        }
        Command::Simulate { nodes, ticks, seed, claims_per_tick } => {
            let result = exodus::simulation::simulate(nodes, ticks, seed, claims_per_tick, None, None);
            println!("{}", result.summary());
            if !result.consistent {
                println!("{}", result.detail);
                std::process::exit(1);
            }
        }
        Command::Run { node, api } => {
            let cfg = config_with(node.data_dir, node.node_host, node.node_port);
            let identity = load_or_create_identity(&cfg.identity_path()).expect("identity");
            let store = Arc::new(ChainStore::open(&cfg.ledger_path()).expect("ledger"));
            let transport: Arc<dyn Transport> = Arc::new(TcpTransport::new(
                identity.node_id.clone(),
                cfg.node_host.clone(),
                cfg.node_port,
                if node.peer.is_empty() { cfg.peers.clone() } else { node.peer.clone() },
                !node.no_discover && cfg.discover,
            ));
            transport.start().expect("transport start");
            let coord = ExodusCoordinator::new(identity, store, transport, cfg.clone(), None);
            coord.connect();
            println!(
                "listening on {}:{} (api on {}:{})",
                cfg.node_host, cfg.node_port, cfg.api_host, cfg.api_port
            );
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            rt.block_on(async move {
                coord.run(api).await;
            });
        }
    }
}