use rand::{RngExt, rng};
use crate::observable::Observable;

pub struct SpinLattice {
    pub spins: Vec<f64>,
    pub dimension: usize,
    pub coupling_constant: f64,
    pub external_field: f64,
    pub energy: f64,
    pub magnetisation: f64,
}

impl SpinLattice {

    pub fn new(dimension: usize, coupling_constant: f64, external_field: f64) -> Self {

        let mut temp = Self {
            spins: SpinLattice::disordered_spins(dimension),
            dimension,
            coupling_constant,
            external_field,
            energy: 0f64,
            magnetisation: 0f64,
        };

        temp.energy = Observable::total_energy_from(&temp, coupling_constant, external_field);
        temp.magnetisation = Observable::total_magnetisation_from(&temp);

        temp
    }

    fn disordered_spins(dimension: usize) -> Vec<f64> {

        let mut spins = vec![0f64; dimension * dimension];
        for i in &mut spins {
            if rng().random::<f64>() < 0.5 {*i = -1.;} else {*i = 1.};
        }
        spins

    }   

    pub fn flip_spin(&mut self, row: usize, col: usize) -> () {
        let d = self.dimension;
        self.spins[row * d + col] *= -1.;
    }   

    pub fn get_spin(&mut self, row: usize, col: usize) -> f64 {
        let d = self.dimension;
        self.spins[row * d + col]
    }

}