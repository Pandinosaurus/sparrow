pub mod kdtree;

use rand::Rng;
use rand::thread_rng;

use commons::bins::Bins;
use commons::packet::Packet;
use commons::persistent_io::VersionedSampleModel;
use commons::persistent_io::load_sample_s3;
use commons::persistent_io::upload_assignments;
use Example;
use super::gamma::Gamma;
use super::model_with_version::ModelWithVersion;
use self::kdtree::Grid;
use self::kdtree::Grids;
use self::kdtree::KdTree;

pub struct Scheduler {
    exp_name: String,
    num_machines: usize,

    scanner_task: Vec<Option<(usize, usize)>>,  // (key, node_id)
    availability: Vec<Option<usize>>,
    last_gamma: Vec<f32>,

    pub grids_version: usize,
    curr_grids: Grids,
    next_grids: Option<Grids>,
}


impl Scheduler {
    pub fn new(
        num_machines: usize, exp_name: &String, _bins: &Vec<Bins>, model: &mut ModelWithVersion,
    ) -> Scheduler {
        let mut scheduler = Scheduler {
            exp_name: exp_name.clone(),
            num_machines: num_machines.clone(),
            scanner_task: vec![None; num_machines],
            grids_version: 0,
            curr_grids: vec![vec![]],  // by default, all scanners are assigned to root
            availability: vec![None],  // ditto
            last_gamma: vec![1.0],     // ditto
            next_grids: None,
        };
        scheduler.set_assignments(model, 1.0);
        scheduler
    }

    pub fn set_assignments(&mut self, model: &mut ModelWithVersion, gamma: f32) -> usize {
        let idle_scanners: Vec<usize> =
            self.scanner_task.iter()
                .enumerate()
                .filter(|(_, assignment)| assignment.is_none())
                .map(|(scanner_index, _)| scanner_index)
                .collect();
        let num_idle_scanners = idle_scanners.len();
        let num_updates = self.assign(idle_scanners, model, gamma);
        if num_idle_scanners > 0 {
            debug!("model-manager, assign updates, {}, {}", num_idle_scanners, num_updates);
        }
        if num_updates > 0 {
            let assignment: Vec<Option<usize>> = self.get_assignment();
            upload_assignments(&assignment, &self.exp_name);
        }
        num_updates
    }

    pub fn handle_accept(&mut self, packet: &Packet) -> bool {
        self.get_grid_node_ids(packet).is_some()
    }

    pub fn handle_empty(&mut self, packet: &Packet) -> bool {
        let grid_node_ids = self.get_grid_node_ids(packet);
        if grid_node_ids.is_none() {
            return false;
        }
        let (grid_index, node_id) = grid_node_ids.unwrap();
        debug!("model_manager, scheduler, handle empty, {}, {}, {}",
                packet.source_machine_id, node_id, packet.gamma);
        self.release_grid(grid_index);
        self.last_gamma[grid_index] = packet.gamma;
        true
    }

    pub fn refresh_grid(&mut self, min_size: usize) {
        let new_grid = get_new_grids(min_size, self.exp_name.as_ref());
        if new_grid.is_some() {
            self.next_grids = new_grid;
        }
        self.reset_assign();
    }

    fn reset_assign(&mut self) -> bool {
        if self.next_grids.is_some() {
            self.curr_grids = self.next_grids.take().unwrap();
            self.availability = vec![None; self.curr_grids.len()];
            self.last_gamma = vec![1.0; self.curr_grids.len()];
            self.scanner_task = vec![None; self.num_machines];
            self.grids_version += 1;
            true
        } else {
            false
        }
    }

    fn get_assignment(&self) -> Vec<Option<usize>> {
        self.scanner_task
            .iter().map(|t| {
                if t.is_some() {
                    let (_grid_index, node_index) = t.unwrap();
                    Some(node_index)
                } else {
                    None
                }
            }).collect()
    }

    // assign a non-taken grid to each idle scanner
    fn assign(
        &mut self, idle_scanners: Vec<usize>, model: &mut ModelWithVersion, gamma: f32,
    ) -> usize {
        let assignment: Vec<(usize, (usize, Grid))> =
            idle_scanners.into_iter()
                         .map(|scanner_id| (scanner_id, self.get_new_grid(scanner_id, gamma)))
                         .filter(|(_, grid)| grid.is_some())
                         .map(|(scanner_id, grid)| (scanner_id, grid.unwrap()))
                         .collect();
        let update_size = assignment.len();
        assignment.into_iter().for_each(|(scanner_id, (grid_index, grid))| {
            let node_index = model.add_grid(grid);
            self.scanner_task[scanner_id] = Some((grid_index, node_index));
            debug!("model-manager, assign, {}, {}, {}", scanner_id, grid_index, node_index);
        });
        update_size
    }

    fn get_new_grid(&mut self, scanner_id: usize, gamma: f32) -> Option<(usize, Grid)> {
        let mut grid_index = 0;
        while grid_index < self.curr_grids.len() {
            if self.availability[grid_index].is_none() && self.last_gamma[grid_index] > gamma {
                break;
            }
            grid_index += 1;
        }
        if grid_index >= self.curr_grids.len() {
            return None;
        }
        let grid = self.curr_grids[grid_index].clone();
        self.availability[grid_index] = Some(scanner_id);
        Some((grid_index, grid))
    }

    fn release_grid(&mut self, grid_index: usize) {
        let machine_id = self.availability[grid_index];
        if machine_id.is_none() {
            return;
        }
        let machine_id = machine_id.unwrap();
        self.availability[grid_index] = None;
        self.scanner_task[machine_id] = None;
    }

    fn get_grid_node_ids(&self, packet: &Packet) -> Option<(usize, usize)> {
        if self.scanner_task[packet.source_machine_id].is_none() {
            debug!("model_manager, scheduler, no assignment, {}, {}, {}",
                    packet.packet_signature, packet.source_machine, packet.source_machine_id);
            return None;
        }
        let (grid_index, node_id) = self.scanner_task[packet.source_machine_id].unwrap();
        if node_id != packet.node_id {
            debug!("model_manager, scheduler, node_id mismatch, {}, {}, {}, {}, {}",
                    packet.packet_signature, packet.source_machine, packet.source_machine_id,
                    node_id, packet.node_id);
            return None;
        }
        Some((grid_index, node_id))
    }

    pub fn print_log(&self, num_consecutive_err: usize, gamma: &Gamma) {
        let num_working_scanners = self.scanner_task.iter().filter(|t| t.is_some()).count();
        debug!("model_manager, scheduler, status, {}, {}, {}, {}",
                num_consecutive_err, gamma.gamma,
                num_working_scanners, self.scanner_task.len() - num_working_scanners);
    }
}

// TODO: support loading from the local disk
fn get_new_grids(min_size: usize, exp_name: &str) -> Option<Grids> {
    let ret = load_sample_s3(0, exp_name);
    if ret.is_none() {
        return None;
    }

    let (version, new_examples, _, _): VersionedSampleModel = ret.unwrap();
    let examples: Vec<Example> = new_examples.into_iter().map(|(e, _)| e).collect();
    debug!("scheduler, received new sample, {}, {}", version, examples.len());

    let mut kd_tree = KdTree::new(examples, min_size);
    let mut grids = kd_tree.get_leaves();
    thread_rng().shuffle(&mut grids);

    Some(grids)
}


#[cfg(test)]
mod tests {
    use super::Scheduler;
    use super::ModelWithVersion;

    use commons::Model;
    use commons::test_helper::get_mock_packet;

    #[test]
    fn test_scheduler() {
        let num_machines = 5;
        let test_machine_id = 0;

        let mut model = ModelWithVersion::new(Model::new(1));
        model.model.add_root(0.0, 0.0);
        let mut scheduler = Scheduler::new(num_machines, &"test".to_string(), &vec![], &mut model);

        // initial phase
        scheduler.set_assignments(&mut model, 0.5);
        let assignment = scheduler.get_assignment();
        assert_eq!(assignment.len(), num_machines);
        assert_eq!(assignment[0], Some(0));
        for i in 1..num_machines {
            assert_eq!(assignment[i], None);
        }

        let packet = get_mock_packet(test_machine_id, 0, 0.5, 0);
        scheduler.handle_accept(&packet);
        assert_eq!(scheduler.get_assignment()[test_machine_id], Some(0));
        scheduler.handle_empty(&packet);
        assert_eq!(scheduler.get_assignment()[test_machine_id], None);
        scheduler.set_assignments(&mut model, 0.5);
        assert_eq!(scheduler.get_assignment()[test_machine_id], None);  // because \gamma didn't change
        scheduler.set_assignments(&mut model, 0.4);
        assert_eq!(scheduler.get_assignment()[test_machine_id], Some(0));

        // refresh grid
        scheduler.refresh_grid(10);
        scheduler.set_assignments(&mut model, 0.5);
        let assignment = scheduler.get_assignment();
        assert_eq!(assignment.len(), num_machines);
        for i in 0..num_machines {
            assert!(assignment[i].is_some());
        }
        let mut assigns: Vec<usize> = assignment.iter().map(|t| t.unwrap()).collect();
        let assign0 = assigns[test_machine_id];
        // all assignments are unique
        assigns.sort();
        for i in 1..num_machines {
            assert!(assigns[i] != assigns[i - 1])
        }

        let packet = get_mock_packet(test_machine_id, assign0, 0.5, 0);
        scheduler.handle_accept(&packet);
        assert_eq!(scheduler.get_assignment()[test_machine_id], Some(assign0));
        scheduler.handle_empty(&packet);
        let assignment = scheduler.get_assignment();
        assert_eq!(assignment[test_machine_id], None);
        for i in (test_machine_id + 1)..num_machines {
            assert!(assignment[i].is_some());
        }
        // now we have enough grids, so no need to set lower gamma (yet)
        scheduler.set_assignments(&mut model, 0.5);
        assert!(scheduler.get_assignment()[test_machine_id].is_some());
    }
}