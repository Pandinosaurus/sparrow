pub mod learner;
pub mod learner_helpers;

use rand;
use rand::Rng;

use std::sync::Arc;
use std::sync::RwLock;
use std::fmt::Display;

use config::Config;
use commons::model::Model;
use commons::tree::Tree;
use scanner::buffer_loader::BufferLoader;
use commons::bins::Bins;
use commons::packet::TaskPacket;
use commons::performance_monitor::PerformanceMonitor;
use self::learner::Learner;
use super::BoosterState;


pub enum BoostingResult {
    Succeed,
    FailedToTrigger,
    LowESS,
}


/// The boosting algorithm. It contains two functions, one for starting
/// the network communication, the other for starting the training procedure.
pub struct Boosting {
    booster_state: Arc<RwLock<BoosterState>>,
    training_loader: BufferLoader,
    learner: Learner,
    num_splits: usize,

    init_packet: TaskPacket,
    curr_model: Model,

    // save_process: bool,
    verbose: bool,
}

impl Boosting {
    /// Create a boosting training class.
    ///
    /// * `training_loader`: the double-buffered data loader that provides examples to the algorithm.
    /// over multiple workers, it might be a subset of the full feature set.
    /// * `max_sample_size`: the number of examples to scan for determining the percentiles for the features.
    /// * `default_gamma`: the initial value of the edge `gamma` of the candidate valid weak rules.
    pub fn new(
        init_packet: TaskPacket,
        booster_state: Arc<RwLock<BoosterState>>,
        training_loader: BufferLoader,
        bins: Vec<Bins>,
        config: Config,
    ) -> Boosting {
        // TODO: make num_cadid a paramter
        let packet = init_packet.clone();
        let (mut model, gamma, expand_node) = (
            packet.model.unwrap(), packet.gamma.unwrap(), packet.expand_node.unwrap(),
        );
        let mut learner = Learner::new(gamma, bins, config.num_features, config.num_splits);

        model.set_base_size();
        learner.set_expand_node(expand_node);
        Boosting {
            booster_state: booster_state,
            training_loader: training_loader,
            learner: learner,
            num_splits: config.num_splits,

            init_packet: init_packet,
            curr_model: model,

            // save_process: config.save_process,
            verbose: false,
        }
    }

    pub fn destroy(self) -> (TaskPacket, Model, BufferLoader) {
        (self.init_packet, self.curr_model, self.training_loader)
    }

    /// Start training the boosting algorithm.
    pub fn training(&mut self) -> BoostingResult {
        info!("Start training.");

        let mut global_timer = PerformanceMonitor::new();
        let mut learner_timer = PerformanceMonitor::new();
        global_timer.start();
        // let mut last_logging_ts = global_timer.get_duration();

        // split the root of the tree on the median of a randomly-selected feature dimension
        let tree = self.get_root_node();
        if tree.is_none() {
            info!("Training is stopped because ess is too small, {:?}",
                self.training_loader.ess);
            return BoostingResult::LowESS;
        }

        let mut tree = tree.unwrap();
        let mut is_booster_running = true;
        self.verbose = false;
        while is_booster_running && !tree.is_full_tree() {
            let mut new_rule = None;
            self.learner.reset();
            while is_booster_running && new_rule.is_none() && self.training_loader.is_ess_valid() &&
                    self.learner.total_count < self.training_loader.size {
                // Logging for the status check
                /*
                if global_timer.get_duration() - last_logging_ts >= 10.0 {
                    self.print_log();
                    last_logging_ts = global_timer.get_duration();
                }
                */

                let (rule, batch_size, _switched) = {
                    let (data, switched) =
                        self.training_loader.get_next_batch_and_update(true, &self.curr_model);

                    learner_timer.resume();
                    let new_rule = self.learner.update(&tree, &data);
                    learner_timer.update(data.len());
                    learner_timer.pause();

                    (new_rule, data.len(), switched)
                };
                new_rule = rule;
                global_timer.update(batch_size);

                // global_timer.write_log("boosting-overall");
                // learner_timer.write_log("boosting-learning");

                let booster_state = self.booster_state.read().unwrap();
                is_booster_running = (*booster_state) == BoosterState::RUNNING;
                drop(booster_state);
            }
            if !self.training_loader.is_ess_valid() {
                info!("Training is stopped because ess is too small, {:?}",
                    self.training_loader.ess);
                return BoostingResult::LowESS;
            }
            if new_rule.is_none() {
                info!("Training is stopped because stopping rule is failed to trigger.");
                return BoostingResult::FailedToTrigger;
            }
            let rule = new_rule.unwrap();
            rule.write_log();
            let (left_index, right_index) = tree.split(
                rule.prt_index,
                rule.feature,
                rule.threshold,
                rule.predict.0,
                rule.predict.1,
            );
            info!("scanner, added new rule, {}, {}, {}, {}, {}", self.curr_model.size(),
                rule.num_scanned, self.learner.total_count, left_index, right_index);
        }
        self.curr_model.append(tree);
        // write_model(&self.curr_model, global_timer.get_duration(), self.save_process);
        info!("Training is finished. Model length: {}.", self.curr_model.size());
        BoostingResult::Succeed
    }

    pub fn get_root_node(&mut self) -> Option<Tree> {
        let root_index = 0;
        let mut rng = rand::thread_rng();
        let selected_feature: usize = rng.gen_range(0, self.learner.num_features);

        let mut feature_vals = Vec::with_capacity(self.training_loader.size);
        let mut count = 0;
        while self.training_loader.is_ess_valid() && count < self.training_loader.size {
            let (data, _) = self.training_loader.get_next_batch(false);
            count += data.len();
            let mut vals = data.iter()
                               .map(|(example, _)| example.feature[selected_feature])
                               .collect();
            feature_vals.append(&mut vals);
        }

        if !self.training_loader.is_ess_large() {  // ESS must be updated after a full scan above
            None
        } else {
            feature_vals.sort();
            let median = feature_vals[feature_vals.len() / 2];
            let mut tree = Tree::new(self.num_splits);
            let (left_index, right_index) = tree.split(
                root_index,
                selected_feature,
                median,
                0.0,
                0.0,
            );
            info!("scanner, added new rule, {}, {}, {}, {}, {}", self.curr_model.size(), count,
                count, left_index, right_index);
            Some(tree)
        }
    }

    #[allow(dead_code)]
    fn print_log(&self) {
        debug!("booster, status, {}",
                vec![
                    self.curr_model.size().to_string(),
                    self.learner.total_count.to_string(),
                    self.learner.rho_gamma.to_string(),
                ].join(", ")
        );
    }

    #[allow(dead_code)]
    fn print_verbose_log<T>(&self, message: T) where T: Display {
        if self.verbose {
            debug!("booster, verbose, {}", message);
        }
    }
}