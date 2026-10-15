use ndarray::{Array2, Array3, Array4};

pub struct CenteredDifference2D{}

impl CenteredDifference2D {

    // scalar -> vector
    pub fn grad_on(values: &Array2<f64>, dx: f64, dy: f64) -> (Array2<f64>, Array2<f64>) {

        let &[n, m] = values.shape() else { panic!("2D Array expected.") };
        let mut grad_0 = Array2::<f64>::zeros([n, m]);
        let mut grad_1 = Array2::<f64>::zeros([n, m]);
        
        for i in 0..n {
            for j in 0..m {
                grad_0[[i, j]] = (values[[(i + 1) % n, j]] - values[[(i + n - 1) % n, j]])/(2. * dx);
                grad_1[[i, j]] = (values[[i, (j + 1) % m]] - values[[i, (j + m - 1) % m]])/(2. * dy);
            }
        }

        (grad_0, grad_1)
    }

    // scalar -> scalar
    pub fn laplacian_on(values: &Array2<f64>, dx: f64, dy: f64) -> Array2<f64> {

        let &[n, m] = values.shape() else { panic!("2D Array expected.") };
        let mut lap = Array2::zeros([n, m]);

        for i in 0..n {
            for j in 0..m {
                lap[[i, j]] += (values[[(i + 1) % n, j]] + values[[(i + n - 1) % n, j]] - 2.*values[[i, j]])/(dx*dx);
                lap[[i, j]] += (values[[i, (j + 1) % m]] + values[[i, (j + m - 1) % m]] - 2.*values[[i, j]])/(dy*dy);
            }
        }

        lap
    }   

}

pub struct CenteredDifference3D {}
impl CenteredDifference3D {

    // scalar -> vector
    pub fn grad_on(values: &Array3<f64>, dx: f64, dy: f64, dz: f64) -> (Array3<f64>, Array3<f64>, Array3<f64>) {

        let &[n, m, l] = values.shape() else { panic!("3D Array expected.") };
        let mut grad_0 = Array3::<f64>::zeros([n, m, l]);
        let mut grad_1 = Array3::<f64>::zeros([n, m, l]);
        let mut grad_2 = Array3::<f64>::zeros([n, m, l]);
        
        for i in 0..n {
            for j in 0..m {
                for k in 0..l {
                    grad_0[[i, j, k]] = (values[[(i + 1) % n, j, k]] - values[[(i + n - 1) % n, j, k]])/(2. * dx);
                    grad_1[[i, j, k]] = (values[[i, (j + 1) % m, k]] - values[[i, (j + m - 1) % m, k]])/(2. * dy);
                    grad_2[[i, j, k]] = (values[[i, j, (k + 1) % l]] - values[[i, j , (k + l - 1) % l]])/(2. * dz);
                }
            }
        }

        (grad_0, grad_1, grad_2)
    }

    // scalar -> scalar
    pub fn laplacian_on(values: &Array3<f64>, dx: f64, dy: f64, dz: f64) -> Array3<f64> {

        let &[n, m, l] = values.shape() else { panic!("3D Array expected.") };
        let mut lap = Array3::zeros([n, m, l]);
        for i in 0..n {
            for j in 0..m {
                for k in 0..l {
                    lap[[i, j, k]] += (values[[(i + 1) % n, j, k]] + values[[(i + n - 1) % n, j, k]] - 2.*values[[i, j, k]])/(dx*dx);
                    lap[[i, j, k]] += (values[[i, (j + 1) % m, k]] + values[[i, (j + m - 1) % m, k]] - 2.*values[[i, j, k]])/(dy*dy);
                    lap[[i, j, k]] += (values[[i, j, (k + 1) % l]] + values[[i, j , (k + l - 1) % l]] - 2.*values[[i, j, k]])/(dz*dz);
                }
            }
        }

        lap
    }

    /// from vector decomposition; vector -> vector
    /// nabul cross A = (d/dy * A_z - d/dz * A_y, d/dz * A_x - d/dx * A_z, d/dy * A_x - d/dx * A_y)
    pub fn curl_on(values: &Array4<f64>, dx: f64, dy: f64, dz: f64) -> (Array3<f64>, Array3<f64>, Array3<f64>) {

        // shape is (leading, x, y, z)
        let &[_, n, m, l] = values.shape() else { panic!("4D Array expected.") };
        let mut curl_0 = Array3::<f64>::zeros([n, m, l]);
        let mut curl_1 = Array3::<f64>::zeros([n, m, l]);
        let mut curl_2 = Array3::<f64>::zeros([n, m, l]);

        for i in 0..n {
            for j in 0..m {
                for k in 0..l {
                    curl_0[[i, j, k]] += (values[[3, i, (j + 1) % m, k]] - values[[3, i, (j + m - 1) % m, k]])/(2. * dy);
                    curl_0[[i, j, k]] -= (values[[2, i, j, (k + 1) % l]] - values[[2, i, j, (k + l - 1) % l]])/(2. * dz);

                    curl_1[[i, j, k]] += (values[[1, i, j, (k + 1) % l]] - values[[1, i, j, (k + l - 1) % l]])/(2. * dz);
                    curl_1[[i, j, k]] -= (values[[3, (i + 1) % n, j, k]] - values[[3, (i + n - 1) % n, j, k]])/(2. * dx);

                    curl_2[[i, j, k]] += (values[[1, i, (j + 1) % m, k]] - values[[1, i, (j + m - 1) % m, k]])/(2. * dy);
                    curl_2[[i, j, k]] -= (values[[2, (i + 1) % n, j, k]] - values[[2, (i + n - 1) % n, j, k]])/(2. * dx);
                }
            }
        }

        (curl_0, curl_1, curl_2)
    }   

}