use rand::{rng, RngExt};

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct GameBoard {
    pub dimension: usize,
    pub cells: Vec<u32>
}

impl GameBoard {

    pub fn new(dimension: usize) -> Self {
        Self{
            dimension,
            cells: GameBoard::random_board(dimension)
        }
    }

    fn random_board(dimension: usize) -> Vec<u32> {
        let cells = vec![0i32; dimension*dimension];
        cells.iter().map(|_| {if rng().random::<f64>() < 0.5 {0u32} else {1u32}}).collect() 
    }
    
}