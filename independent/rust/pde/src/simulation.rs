use crate::lattice::Lattice2;
use crate::vector_calculus::CenteredDifference2D;
use ndarray::{Array, Array1, Array2};

pub struct Simulation {}
impl Simulation {

    // pub fn cahn_hilliard(iterations: usize, nx: usize, ny: usize, dx: f64, dy: f64) {

    //     // constants
    //     let a = 1.;
    //     let k = 1.;
    //     let initial_value = 0.;
    //     let noise = 0.1;
    //     let boundary_value = 0.;

    //     let chemical_potential = |op, dx, dy| -a*op*(1. - op*op) - k*CenteredDifference2D::laplacian_on(op, dx, dy);
    //     let free_energy_density = |op, dx, dy| - 0.5*a*op*op + 0.25*a*op*op*op*op + 0.5*k*CenteredDifference2D::laplacian_on(op, dx, dy);

    //     let mut free_energy = Array1::<f64>::zeros([iterations]);
    //     let mut order_parameter = Lattice2::new(initial_value, noise, boundary_value, nx, ny); 





    // }


}