use super::packet_stats::PacketStats;
use super::packet_stats::UpdateSpeed;

pub struct Gamma {
    pub gamma: f32,
    pub root_gamma: f32,
    shrink_factor: f32,
    pub gamma_version: usize,
    min_gamma: f32,
}


impl Gamma {
    pub fn new(default_gamma: f32, min_gamma: f32) -> Gamma {
        Gamma {
            gamma: default_gamma.clone(),
            root_gamma: default_gamma,
            gamma_version: 0,
            shrink_factor: 0.9,
            min_gamma: min_gamma,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.gamma >= self.min_gamma
    }

    pub fn adjust(&mut self, packet_stats: &PacketStats, model_size: usize) -> bool {
        if packet_stats.is_nonroot_same_trend() {
            self.shrink_factor = (0.8 + self.shrink_factor) / 2.0;
        } else if packet_stats.is_nonroot_opposite_trend() {
            self.shrink_factor = (1.0 + self.shrink_factor) / 2.0;
        }
        self.gamma = match packet_stats.curr_nonroot_condition {
            UpdateSpeed::TooFast => self.gamma / self.shrink_factor,  // increase gamma
            UpdateSpeed::TooSlow => self.gamma * self.shrink_factor,  // decrease gamma
            UpdateSpeed::Okay    => self.gamma,
        };
        if packet_stats.curr_nonroot_condition != UpdateSpeed::Okay {
            // gamma is changed
            self.gamma_version += 1;
            debug!("model_mamanger, gamma update, non-root, {}, {}, {}, {}, {}, {}, {}",
                    self.gamma_version, self.gamma, self.root_gamma, self.shrink_factor, model_size,
                    packet_stats.avg_accept_nonroot_rate, packet_stats.last_accept_nonroot_rate);
            true
        } else {
            false
        }
    }

    pub fn decrease_gamma(&mut self, model_size: usize) {
        self.gamma *= self.shrink_factor;
        self.gamma_version += 1;
        debug!("model_manager, gamma update, forced non-root, {}, {}, {}, {}, {}",
                self.gamma_version, self.gamma, self.root_gamma, self.shrink_factor, model_size);
    }

    pub fn decrease_root_gamma(&mut self, model_size: usize) {
        self.root_gamma *= 0.8;
        self.gamma_version += 1;
        debug!("model_manager, gamma update, root, {}, {}, {}, {}, {}",
                self.gamma_version, self.gamma, self.root_gamma, self.shrink_factor, model_size);
    }
}