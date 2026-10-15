use crate::{CORRELATION_GAP, COUPLING_CONSTANT, DIMENSION, EXTERNAL_FIELD, MAX_BETA, MIN_BETA, NUMBER_OF_SIMULATIONS, SWEEPS_PER_SIMULATION, WARMUP_SWEEPS};
use crate::{markov_chain::MarkovChain, spin_lattice::SpinLattice, observable::Observable, plot::Plot};

pub struct Simulation{}

impl Simulation {

    pub fn to_equilibrium(spin_lattice: &mut SpinLattice, sweeps: usize, beta: f64) -> (Vec<f64>, Vec<f64>) {

        let measurements = sweeps/CORRELATION_GAP;

        let mut energy = vec![0.; measurements]; energy[0] = spin_lattice.energy;
        let mut magnetisation = vec![0.; measurements]; magnetisation[0] = spin_lattice.magnetisation;

        for i in 1..measurements {

            Simulation::sweep_for(spin_lattice, beta, CORRELATION_GAP);
            energy[i] = spin_lattice.energy;
            magnetisation[i] = spin_lattice.magnetisation;

        }

        (energy, magnetisation)
    }

    fn sweep_for(spin_lattice: &mut SpinLattice, beta: f64, number_of_sweeps: usize) {

        for _ in 0..spin_lattice.dimension*spin_lattice.dimension*number_of_sweeps {
            MarkovChain::glauber_step(spin_lattice, beta);
        }

    }

    fn linspace(start: f64, end: f64, length: usize) -> Vec<f64> {
        let range = end - start;
        let step = range.abs()/(length as f64);
        let vals = vec![0.;length];

        if range > 0. {
            vals.iter().enumerate().map(|(i, _)| start + (i as f64)*step).collect()
        } else {
            vals.iter().enumerate().map(|(i, _)| start - (i as f64)*step).collect()
        }
    }

    fn mean(values: &Vec<f64>) -> f64 {
        values.iter().sum::<f64>()/(values.len() as f64)
    }

    pub fn phase_diagram() {

        let mut spin_lattice = SpinLattice::new(DIMENSION, COUPLING_CONSTANT, EXTERNAL_FIELD);
        let mut beta = Simulation::linspace(MAX_BETA, MIN_BETA, NUMBER_OF_SIMULATIONS); //hot -> cold
        
        let mut mean_energy = vec![0f64; NUMBER_OF_SIMULATIONS];
        let mut mean_magnetisation = vec![0f64; NUMBER_OF_SIMULATIONS];
        let mut susceptibility = vec![0f64; NUMBER_OF_SIMULATIONS];
        let mut heat_capacity = vec![0f64; NUMBER_OF_SIMULATIONS];

        Simulation::sweep_for(&mut spin_lattice, beta[0], WARMUP_SWEEPS); //warmup

        for (i, b) in beta.iter().enumerate() {

            let (energy, magnetisation) = Simulation::to_equilibrium(&mut spin_lattice, SWEEPS_PER_SIMULATION, *b);
            mean_energy[i] = Simulation::mean(&energy);
            mean_magnetisation[i] = Simulation::mean(&magnetisation);
            susceptibility[i] = Observable::susceptibility_from(&magnetisation, *b, spin_lattice.dimension);
            heat_capacity[i] = Observable::heat_capacity_from(&energy, *b, spin_lattice.dimension);
        }

        for i in [&mut beta, &mut mean_energy, &mut mean_magnetisation, &mut heat_capacity, &mut susceptibility] {i.reverse()};

        Plot::plot_phase_diagram(&beta, &mean_energy, &mean_magnetisation, &heat_capacity, &susceptibility);

    }
}