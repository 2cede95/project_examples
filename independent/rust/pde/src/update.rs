use crate::vector_calculus::{CenteredDifference2D, CenteredDifference3D};
use ndarray::{Array2, Array3};

pub struct Update2D {}
/// scalar -> scalar
impl Update2D {

    pub fn forward_euler_on(
        values: &Array2<f64>,
        values_2: &Array2<f64>,
        dx: f64,
        dy: f64,
        dt: f64) -> Array2<f64> {
        values + dt * CenteredDifference2D::laplacian_on(&values_2, dx, dy)
    }

}

pub struct Update3D {}
impl Update3D {

    // scalar -> scalar
    pub fn jacobi_on(values: &Array3<f64>,
        values_2: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64) -> Array3<f64> {
        values + dt*(CenteredDifference3D::laplacian_on(values, dx, dy, dz) + values_2) 
    }

    // scalar -> scalar
    pub fn gauss_seidel_on(values: &Array3<f64>, values_2: &Array3<f64>, dx: f64, dy: f64, dz: f64, dt: f64) -> Array3<f64> {

        let &[n, m, l] = values.shape() else { panic!("3D Array expected.") };
        let mut gauss_seidel = values.clone();

        // even values
        for mut i in 0..=n/2 {
            for mut j in 0..=m/2 {
                for mut k in 0..=l/2 {
                    i*=2; j*=2; k*=2;
                    let mut lap = 0.;
                    lap += (values[[(i + 1) % n, j, k]] + values[[(i + n - 1) % n, j, k]] - 2.*values[[i, j, k]])/(dx*dx);
                    lap += (values[[i, (j + 1) % m, k]] + values[[i, (j + m - 1) % m, k]] - 2.*values[[i, j, k]])/(dy*dy);
                    lap += (values[[i, j, (k + 1) % l]] + values[[i, j , (k + l - 1) % l]] - 2.*values[[i, j, k]])/(dz*dz);
                    gauss_seidel[[i, j, k]] += dt*(lap + values_2[[i, j, k]]); 
                }
            }
        }

        // odd values
        for mut i in 0..n/2 {
            for mut j in 0..m/2 {
                for mut k in 0..l/2 {
                    i += i + 1; j += j + 1; k += k + 1;
                    let mut lap = 0.;
                    lap += (values[[(i + 1) % n, j, k]] + values[[(i + n - 1) % n, j, k]] - 2.*values[[i, j, k]])/(dx*dx);
                    lap += (values[[i, (j + 1) % m, k]] + values[[i, (j + m - 1) % m, k]] - 2.*values[[i, j, k]])/(dy*dy);
                    lap += (values[[i, j, (k + 1) % l]] + values[[i, j , (k + l - 1) % l]] - 2.*values[[i, j, k]])/(dz*dz);
                    gauss_seidel[[i, j, k]] += dt*(lap + values_2[[i, j, k]]); 
                }
            }
        }

        gauss_seidel
    }

    /// successive over relax
    /// scalar -> scalar
    pub fn sor_on(values: &Array3<f64>,
        values_2: &Array3<f64>,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        relaxation_parameter: f64) -> Array3<f64> {

        let gauss_seidel = Update3D::gauss_seidel_on(values, values_2, dx, dy, dz, dt);
        (1. - relaxation_parameter)*values + relaxation_parameter*gauss_seidel

    }

}