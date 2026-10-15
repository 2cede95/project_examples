use crate::{MAX_ITERATIONS, evolve::Evolve, game_board::GameBoard, plot::Plot};
use std::collections::HashSet;

pub struct Simulation {}

impl Simulation {

    pub fn to_equilibrium(dimension: usize) -> u32 {

        let mut game_board = GameBoard::new(dimension);
        let mut board_history: HashSet<GameBoard> = HashSet::new();
        let mut iterations: u32 = 0;

        while !board_history.contains(&game_board) && iterations <= MAX_ITERATIONS as u32 {

            board_history.insert(game_board.clone());
            Evolve::sweep_board(&mut game_board);
            iterations += 1;

        };
        
        iterations
    }

    pub fn equilibriation_times(dimension: usize, simulation_count: usize) -> () {

        let mut iterations = vec![0u32; simulation_count];
        for i in 0..simulation_count {

            iterations[i] = Simulation::to_equilibrium(dimension);

        }

        Plot::histogram(&iterations);
    }

}