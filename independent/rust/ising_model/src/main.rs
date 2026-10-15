mod spin_lattice;
mod markov_chain;
mod simulation;
mod observable;
mod plot;

const CORRELATION_GAP: usize = 10;
const DIMENSION: usize = 50;
const COUPLING_CONSTANT: f64 = 1.;
const EXTERNAL_FIELD: f64 = 0.;
const NUMBER_OF_SIMULATIONS: usize = 100;
const SWEEPS_PER_SIMULATION: usize = 10000;
const WARMUP_SWEEPS: usize = 100;
const MAX_BETA: f64 = 3.; // 1/kbT = beta
const MIN_BETA: f64 = 1.;

fn main() {
    simulation::Simulation::phase_diagram();
}
