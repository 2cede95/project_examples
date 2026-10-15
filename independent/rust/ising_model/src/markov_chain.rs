use crate::spin_lattice::*;
use rand::{rng, RngExt};
use crate::observable::Observable;

pub struct MarkovChain {}

impl MarkovChain {

    fn metropolis_check(energy_change: f64, beta: f64) -> bool {
        if energy_change < 0. || (-energy_change/beta).exp() > rng().random::<f64>() {true} else {false}
    }

    pub fn glauber_step(spin_lattice: &mut SpinLattice, beta: f64) -> () {

        let random_row: usize = rng().random_range(0..spin_lattice.dimension);
        let random_col: usize = rng().random_range(0..spin_lattice.dimension);
        let energy_change = -2.*Observable::site_energy_from(spin_lattice, random_row, random_col);
        
        if MarkovChain::metropolis_check(energy_change, beta) {
            spin_lattice.flip_spin(random_row, random_col);
            spin_lattice.energy += energy_change;
            spin_lattice.magnetisation += 2.*spin_lattice.get_spin(random_row, random_col);
        } 
    }
}

