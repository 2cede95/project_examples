use ndarray::Array2;
use rand::{rng, RngExt};

pub struct Lattice2 {

    pub vals: Array2<f64>

}

impl Lattice2 {

    pub fn new_boundary(initial_value: f64, noise: f64, boundary_value: f64, nx: usize, ny: usize) -> Self {

        let mut buffer = Lattice2::initial_values_with_ghost(initial_value, noise, nx, ny);
        Lattice2::apply_dirichlett(&mut buffer, boundary_value);

        Self {
            vals: buffer
        }

    }

    pub fn new(initial_value: f64, noise: f64, nx: usize, ny: usize) -> Self {

        let mut buffer = Lattice2::initial_values(initial_value, noise, nx, ny);
        
        Self {
            vals: buffer
        }

    }

    fn initial_values_with_ghost(initial_value: f64, noise: f64, nx: usize, ny: usize) -> Array2<f64> {
        let one = Array2::<f64>::ones([nx + 2, ny + 2]); // 0 and ni + 1 are ghost cells https://link.springer.com/chapter/10.1007/978-3-319-49583-5_31
        one.clone()*initial_value + one.map(|x| x*noise*rng().random::<f64>())
    }

    fn initial_values(initial_value: f64, noise: f64, nx: usize, ny: usize) -> Array2<f64> {
        let one = Array2::<f64>::ones([nx, ny]); // 0 and ni + 1 are ghost cells https://link.springer.com/chapter/10.1007/978-3-319-49583-5_31
        one.clone()*initial_value + one.map(|x| x*noise*rng().random::<f64>())
    }

    pub fn apply_dirichlett(values: &mut Array2<f64>, boundary_value: f64) {

        let &[n, l] = values.shape() else { panic!("Expected 2D Array.")}; 
        for i in 0..n {
            values[[0, i]] = boundary_value; // left
            values[[n - 1, i]] = boundary_value; // right
        }  

        for j in 0..l {
            values[[j, 0]] = boundary_value; // top
            values[[j, l - 1]] = boundary_value; // bottom
        }    

    }

    pub fn update(&mut self, arr: Array2<f64>) {
        self.vals = arr;
    }


}