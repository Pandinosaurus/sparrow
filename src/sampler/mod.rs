/// Syncing model to S3
pub mod model_sync;
/// A stratified storage structor that organize examples on disk according to their weights.
pub mod stratified_storage;


use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::sleep;
use std::thread::spawn;
use std::time::Duration;

use commons::channel;
use commons::io::raw_read_all;
use commons::Model;
use commons::bins::Bins;
use commons::INIT_MODEL_PREFIX;
use config::Config;
use config::SampleMode;

use self::model_sync::ModelStats;
use self::model_sync::ModelSync;
use self::model_sync::gamma::Gamma;
use self::stratified_storage::StratifiedStorage;


pub fn start(config: &Config, sample_mode: &SampleMode, bins: &Vec<Bins>, init_tree: &Model) {
    debug!("Starting Sampler");
    let sampler_state = Arc::new(RwLock::new(true));
    // Pass the models between the network to the Strata
    let (next_model_s, next_model_r) = channel::bounded(config.channel_size, "updated-models");
    let gen_sample_version = Arc::new(RwLock::new(0));
    debug!("Starting the stratified structure.");
    let init_model_name = INIT_MODEL_PREFIX.to_string();
    let stratified_structure = StratifiedStorage::new(
        init_tree.clone(),
        init_model_name.clone(),
        config.num_examples,
        config.buffer_size,
        config.num_features,
        config.positive.clone(),
        config.num_examples_per_block,
        config.disk_buffer_filename.as_ref(),
        gen_sample_version.clone(),
        sample_mode.clone(),
        config.num_assigners,
        config.num_samplers,
        next_model_r,
        config.channel_size,
        sampler_state.clone(),
        config.debug_mode,
        config.resume_training,
        config.exp_name.clone(),
    );

    debug!("Initializing the stratified structure.");
    stratified_structure.init_stratified_from_file(
        config.training_filename.clone(),
        config.num_examples,
        config.batch_size,
        config.num_features,
        bins.clone(),
        init_tree.clone(),
    );

    debug!("Starting the model sync.");
    let model_stats = ModelStats::new(init_tree.clone(), config.num_trees);
    let gamma = Gamma::new(config.default_gamma, config.min_gamma);
    let mut model_sync = ModelSync::new(
        model_stats,
        config.num_trees,
        &config.exp_name,
        config.min_ess,
        gamma,
        sampler_state.clone(),
        next_model_s,
        gen_sample_version.clone(),
        stratified_structure.node_counts.clone(),
    );
    model_sync.start_network(config.local_name.clone(), config.network.clone(), config.port);
    spawn(move || {
        model_sync.run_with_network();
    });

    // Monitor running state
    let mut state = true;
    while state {
        // Check if termination is manually requested
        let filename = "status.txt".to_string();
        if Path::new(&filename).exists() && raw_read_all(&filename).trim() == "0".to_string() {
            debug!("sampler state, false, change in the status.txt has been detected");
            *(sampler_state.write().unwrap()) = false;
        }
        state = {
            let t = sampler_state.read().unwrap();
            *t
        };
        sleep(Duration::from_secs(20));
    }
    debug!("State has been set to false. Main process to exit in 120 seconds.");
    sleep(Duration::from_secs(120));
    if std::fs::remove_file("status.txt").is_ok() {
        debug!("removed `status.txt`");
    }
}