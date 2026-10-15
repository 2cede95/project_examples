mod game_board;
mod evolve;
mod simulation;
mod plot;

const MAX_ITERATIONS: usize = 15000;

fn main() {
    // plot::Plot::gol_animation(500, 2, 1);
    simulation::Simulation::equilibriation_times(50, 10000);
}
