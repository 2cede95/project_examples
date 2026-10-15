use crate::game_board::GameBoard;

pub struct Evolve {}

impl Evolve {

    pub fn sweep_board(game_board: &mut GameBoard) {
        let d = game_board.dimension;
        let board_buffer = game_board.clone();
        for i in 0..d {
            for j in 0..d {
                game_board.cells[i*d + j] = 
                Evolve::rules_to_cell(board_buffer.cells[i*d + j],
                Evolve::nearest_neighbours(&board_buffer, i, j))
            }
        }
    }

    fn rules_to_cell(state: u32, alive_neighbours: u32) -> u32 {
        match (state, alive_neighbours) {
            (1, 2) | (1, 3) => 1,
            (0, 3) => 1,
            _ => 0
        }
    }

    fn nearest_neighbours(game_board: &GameBoard, row: usize, col: usize) -> u32 {
        let d = game_board.dimension;
        game_board.cells[((row + 1) % d) * d + col]
        + game_board.cells[((row + d - 1) % d) * d + col]
        + game_board.cells[row * d + (col + 1) % d]
        + game_board.cells[row * d + (col + d - 1) % d]
        + game_board.cells[((row + 1) % d) * d + (col + 1) % d]
        + game_board.cells[((row + 1) % d) * d + (col - 1) % d]
        + game_board.cells[((row - 1) % d) * d + (col + 1) % d]
        + game_board.cells[((row - 1) % d) * d + (col - 1) % d]
    }

}