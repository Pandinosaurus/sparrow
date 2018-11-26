use rayon::prelude::*;

use std::f32::INFINITY;
use std::cmp::Ordering;
use std::ops::Range;

use commons::ExampleInSampleSet;
use tree::Tree;
use super::bins::Bins;

use buffer_loader::BufferLoader;
use commons::max;
use commons::min;
use commons::get_bound;
use commons::get_relative_weights;
use commons::get_symmetric_label;

/*
TODO: extend support to regression tasks

Each split corresponds to 4 types of predictions,
    1. Left +1, Right +1;
    2. Left +1, Right -1;
    3. Left -1, Right +1;
    4. Left -1, Right -1.
*/

// TODO: extend learner to support multi-level trees
/// A weak rule with an edge larger or equal to the targetting value of `gamma`
pub struct WeakRule {
    feature: usize,
    threshold: f32,
    left_predict: f32,
    right_predict: f32,

    pub gamma: f32,
    raw_martingale: f32,
    sum_c: f32,
    sum_c_squared: f32,
    bound: f32,
    num_scanned: usize,
}

impl WeakRule {
    /// Return a decision tree (or decision stump) according to the valid weak rule
    pub fn to_tree(self) -> Tree {
        let mut tree = Tree::new(2);
        tree.split(0, self.feature, self.threshold, self.left_predict, self.right_predict);
        tree.release();
        tree
    }

    pub fn write_log(&self, model_len: usize, curr_sum_gamma: f32) {
        info!(
            "new-tree-info, {}, {}, {}, {}, {}, {}, {}, {}",
            model_len + 1,
            self.num_scanned,
            self.gamma,
            self.raw_martingale,
            self.sum_c,
            self.sum_c_squared,
            self.bound,
            curr_sum_gamma + self.gamma,
        );
    }
}


/// Statisitics of all weak rules that are being evaluated.
/// The objective of `Learner` is to find a weak rule that satisfies the condition of
/// the stopping rule.
pub struct Learner {
    bins: Vec<Bins>,
    range_start: usize,
    range_end: usize,
    cur_rho_gamma: f32,
    num_examples_before_shrink: usize,

    weak_rules_score: Vec<Vec<f32>>,
    sum_c:            Vec<Vec<f32>>,
    sum_c_squared:    Vec<Vec<f32>>,

    pub count: usize,
    sum_weights: f32,
    sum_weights_squared: f32,

}

impl Learner {
    /// Create a `Learner` that search for valid weak rules.
    /// `default_gamma` is the initial value of the edge `gamma`.
    /// `bins` is vectors of the all thresholds on all candidate features for generating weak rules.
    ///
    /// `range` is the range of candidate features for generating weak rules. In most cases,
    /// if the algorithm is running on a single worker, `range` is 0..`num_of_features`;
    /// if the algorithm is running on multiple workers, `range` is a subset of the feature set.
    pub fn new(default_gamma: f32, num_examples_before_shrink: u32, bins: Vec<Bins>, range: &Range<usize>) -> Learner {
        Learner {
            range_start: range.start,
            range_end: range.end,
            cur_rho_gamma: default_gamma,
            num_examples_before_shrink: num_examples_before_shrink as usize,

            weak_rules_score: bins.iter().map(|bin| vec![0.0; 2 * bin.len()]).collect(),
            sum_c:            bins.iter().map(|bin| vec![0.0; 2 * bin.len()]).collect(),
            sum_c_squared:    bins.iter().map(|bin| vec![0.0; 2 * bin.len()]).collect(),

            count: 0,
            sum_weights: 0.0,
            sum_weights_squared: 0.0,
            bins: bins,
        }
    }

    /// Reset the statistics of all candidate weak rules,
    /// but leave the targetting `gamma` unchanged.
    pub fn reset(&mut self) {
        for i in 0..self.weak_rules_score.len() {
            for j in 0..self.weak_rules_score[i].len() {
                self.weak_rules_score[i][j] = 0.0;
                self.sum_c[i][j]            = 0.0;
                self.sum_c_squared[i][j]    = 0.0;
            }
        }

        self.count = 0;
        self.sum_weights = 0.0;
        self.sum_weights_squared = 0.0;
    }

    fn get_max_empirical_ratio(&self) -> f32 {
        self.weak_rules_score.iter().flat_map(|rules| {
            rules.iter().map(|scores| {
                scores / self.sum_weights
            })
        }).fold(0.0, max)
    }

    /// Update the statistics of all candidate weak rules using current batch of
    /// training examples. 
    pub fn update(&mut self, data: &[ExampleInSampleSet]) -> Option<WeakRule> {
        // Shrinking the value of the targetting edge `gamma` if it was too high
        if self.count >= self.num_examples_before_shrink {
            let old_rho_gamma = self.cur_rho_gamma;
            let max_empirical_gamma = self.get_max_empirical_ratio() / 2.0;
            self.cur_rho_gamma = 0.9 * min(self.cur_rho_gamma, max_empirical_gamma);
            self.reset();
            debug!("shrink-gamma, {}, {}, {}",
                   old_rho_gamma, max_empirical_gamma, self.cur_rho_gamma);
        }

        // update global stats
        let weights = get_relative_weights(data);
        // TODO: sum_w is not considered in calculation of weak rules
        let (sum_w, sum_w_squared, sum_labeled_weight) =
            data.par_iter()
                .zip(weights.par_iter())
                .map(|(example, weight)| (
                    weight.clone(),
                    weight * weight,
                    get_symmetric_label(&(example.0)) * weight,
                )).reduce(
                    || (0.0, 0.0, 0.0),
                    |(a1, a2, a3), (b1, b2, b3)| (a1 + b1, a2 + b2, a3 + b3)
                );
        self.sum_weights += sum_w;
        self.sum_weights_squared += sum_w_squared;
        self.count += data.len();

        // preprocess examples
        let range_start = self.range_start;
        let range_end = self.range_end;
        let mut data: Vec<(usize, f32, f32, (f32, f32), (f32, f32))> =
            data.par_iter().zip(weights.par_iter()).flat_map(|(example, weight)| {
                let score = weight * get_symmetric_label(&(example.0));
                let mut i = (example.0).get_position(range_start);
                let mut ret = vec![];
                while let Some((index, value)) = (example.0).get_value_at(i) {
                    if index >= range_end {
                        break;
                    }
                    ret.push(
                        (index - range_start, value as f32, weight.powi(2),
                         (*weight, score), (*weight, score))
                    );
                    i += 1;
                }
                ret
        }).collect();  // rel_index, feature_val, weight^2, left, right
        data.sort_by(|a, b| {
            if a.0 <= b.0 && a.1 <= b.1 {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });
        for i in 0..data.len() {
            if i > 0 {
                let (index, value, wsq, left, right) = data[i];
                let left_weight = left.0 + (data[i - 1].3).0;
                let left_sum    = left.1 + (data[i - 1].3).1;
                data[i] = (index, value, wsq, (left_weight, left_sum), right);
            }
            let j = data.len() - 1 - i;
            if j + 1 < data.len() {
                let (index, value, wsq, left, right) = data[j];
                let right_weight = right.0 + (data[j + 1].4).0;
                let right_sum    = right.1 + (data[j + 1].4).1;
                data[j] = (index, value, wsq, left, (right_weight, right_sum));
            }
        }

        // update each weak rule
        let mut valid_weak_rule = None;
        let num_scanned = self.count;
        let gamma = self.cur_rho_gamma;
        let mut i = 0;
        while i < data.len() {
            let index = data[i].0;
            let bins             = &self.bins[index];
            let weak_rules_score = &mut self.weak_rules_score[index];
            let sum_c            = &mut self.sum_c[index];
            let sum_c_squared    = &mut self.sum_c_squared[index];
            for j in 0..bins.len() {
                let threshold = bins.get_vals()[j];
                while i + 1 < data.len() && data[i + 1].0 == index && data[i].1 <= threshold {
                    i += 1;
                }
                // Update left branch (3)
                let left_entry = {
                    if i > 0 && data[i - 1].0 == index && data[i - 1].1 <= threshold {
                        Some((data[i - 1].2, data[i - 1].3))
                    } else if data[i].0 == index && data[i].1 <= threshold {
                        Some((data[i].2, data[i].3))
                    } else {
                        None
                    }
                };
                if left_entry.is_some() {
                    let (wsq, (weight, score)) = left_entry.unwrap();
                    sum_c[2 * j]            += score - 2.0 * gamma * weight;
                    sum_c_squared[2 * j]    += (1.0 + 2.0 * gamma).powi(2) * wsq;
                    weak_rules_score[2 * j] += score;
                }
                // Update right branch (4)
                let right_entry = {
                    if data[i].0 == index && data[i].1 > threshold {
                        Some((data[i].2, data[i].4))
                    } else {
                        None
                    }
                };
                if right_entry.is_some() {
                    let (wsq, (weight, score)) = right_entry.unwrap();
                    sum_c[2 * j + 1]            += score - 2.0 * gamma * weight;
                    sum_c_squared[2 * j + 1]    += (1.0 + 2.0 * gamma).powi(2) * wsq;
                    weak_rules_score[2 * j + 1] += score;
                }

                for k in [2 * j as usize, 2 * j + 1 as usize].iter() {
                    let weak_rules_score = weak_rules_score[*k];
                    let sum_c            = sum_c[*k];
                    let sum_c_squared    = sum_c_squared[*k];
                    let bound = get_bound(sum_c, sum_c_squared).unwrap_or(INFINITY);
                    if sum_c > bound {
                        let base_pred = 0.5 * ((0.5 + gamma) / (0.5 - gamma)).ln();
                        let left_predict  = if *k == 2 * j     { base_pred } else { 0.0 };
                        let right_predict = if *k == 2 * j + 1 { base_pred } else { 0.0 };
                        valid_weak_rule = Some(
                            WeakRule {
                                feature:        index + range_start,
                                threshold:      threshold,
                                left_predict:   left_predict,
                                right_predict:  right_predict,

                                gamma:          gamma,
                                raw_martingale: weak_rules_score,
                                sum_c:          sum_c,
                                sum_c_squared:  sum_c_squared,
                                bound:          bound,
                                num_scanned:    num_scanned,
                            }
                        );
                    }
                }
            }

            // jump to the region with the next index
            while i < data.len() && data[i].0 == index {
                i += 1;
            }
        }
        valid_weak_rule
    }
}


pub fn get_base_tree(max_sample_size: usize, data_loader: &mut BufferLoader) -> (Tree, f32) {
    let mut sample_size = max_sample_size;
    let mut n_pos = 0;
    let mut n_neg = 0;
    while sample_size > 0 {
        let data = data_loader.get_next_batch(true);
        let (num_pos, num_neg) =
            data.par_iter().fold(
                || (0, 0),
                |(num_pos, num_neg), (example, _, _)| {
                    if example.get_label() > 0 {
                        (num_pos + 1, num_neg)
                    } else {
                        (num_pos, num_neg + 1)
                    }
                }
            ).reduce(|| (0, 0), |(a1, a2), (b1, b2)| (a1 + b1, a2 + b2));
        n_pos += num_pos;
        n_neg += num_neg;
        sample_size -= data.len();
    }

    let gamma = (0.5 - n_pos as f32 / (n_pos + n_neg) as f32).abs();
    let prediction = 0.5 * (n_pos as f32 / n_neg as f32).ln();
    let mut tree = Tree::new(2);
    tree.split(0, 0, 0.0, prediction, prediction);
    tree.release();

    info!("root-tree-info, {}, {}, {}, {}", 1, max_sample_size, gamma, gamma * gamma);
    (tree, gamma)
}
