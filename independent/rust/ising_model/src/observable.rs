use crate::spin_lattice::SpinLattice;

pub struct Observable{}

impl Observable {
    
    pub fn total_energy_from(spin_lattice: &SpinLattice, coupling_constant: f64, external_field: f64) -> f64 {

        let mut energy: f64 = 0.;
        let d = spin_lattice.dimension;
        for row in 0..d {
            for col in 0..d {
                let neighbour_sum = (spin_lattice.spins[((row+1) % d)*d  + col]
                                  + spin_lattice.spins[((row + d -1) % d) * d + col]
                                  + spin_lattice.spins[row * d + (col + 1) % d]
                                  + spin_lattice.spins[row * d + (col + d - 1) % d])
                                  as f64;
                energy -= (0.5*neighbour_sum*coupling_constant + external_field)*(spin_lattice.spins[row*d + col] as f64);
            }
        }

        energy
    }

    pub fn total_magnetisation_from(spin_lattice: &SpinLattice) -> f64 {
        spin_lattice.spins.iter().sum::<f64>()
    }

    pub fn site_energy_from(spin_lattice: &SpinLattice, row: usize, col: usize,) -> f64 {

        let d = spin_lattice.dimension;

        let neighbour_sum = (spin_lattice.spins[((row+1) % d)*d + col]
                                  + spin_lattice.spins[((row + d -1) % d) * d + col]
                                  + spin_lattice.spins[row * d + (col + 1) % d]
                                  + spin_lattice.spins[row * d + (col + d - 1) % d])
                                  as f64;
        -(neighbour_sum*spin_lattice.coupling_constant + spin_lattice.external_field)*(spin_lattice.spins[row*d + col] as f64)
    }

    fn variance(values: &Vec<f64>) -> f64 {
        let mean_squared = f64::powi(values.iter().sum::<f64>()/(values.len() as f64), 2);
        let squared_mean = values.iter().map(|x| x*x).sum::<f64>()/(values.len() as f64);
        squared_mean - mean_squared
    }

    pub fn susceptibility_from(magnetisation: &Vec<f64>, beta: f64, dimension: usize) -> f64 {
        Observable::variance(magnetisation)/(((dimension*dimension) as f64)*beta)
    }

    pub fn heat_capacity_from(energy: &Vec<f64>, beta: f64, dimension: usize) -> f64 {
        Observable::variance(energy)/(((dimension*dimension) as f64)*beta*beta)
    }

}