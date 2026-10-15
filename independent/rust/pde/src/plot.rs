use minifb::{Key, Window, WindowOptions};
use rand::{RngExt, rng};
use core::time;
use std::io::read_to_string;
use std::{cell, collections::HashSet, iter::StepBy};
use plotters::prelude::*;
use ndarray::{Array1, Array2};

use crate::lattice::Lattice2;
use crate::vector_calculus::CenteredDifference2D;
use crate::update::Update2D;

pub struct Plot2D {}

impl Plot2D {

    pub fn cahn_hilliard_animation(max_iterations: usize,
        nx: usize,
        ny: usize,
        initial_value: f64,
        noise: f64,
        dx: f64,
        dy: f64,
        dt : f64,
        cell_size: usize,
        frame_period_nanoseconds: u64
        ) {

        // constants
        let a = 1.;
        let k = 1.;

        let get_chemical_potential = |op: &Array2<f64>, dx, dy| -a*op*(1. - op*op) - k*CenteredDifference2D::laplacian_on(op, dx, dy);
        let get_free_energy_density = |op: &Array2<f64>, dx, dy| - 0.5*a*op*op + 0.25*a*op*op*op*op + 0.5*k*CenteredDifference2D::laplacian_on(op, dx, dy);

        let mut free_energy = Array1::<f64>::zeros([max_iterations]);
        let mut order_parameter = Lattice2::new(initial_value, noise, nx, ny); 

        let &[nx, ny] = order_parameter.vals.shape() else { panic!("Expected 2D array.") };
        let cx = nx * cell_size; let cy = ny * cell_size; // redefinition to account for ghost variables (if present)
        let mut window = Window::new("Cahn Hilliard",
                                    cx, cy,
                                    WindowOptions::default()
                                    ).unwrap();


        let mut pixel_buffer = Array2::<u32>::zeros([cx, cy]);
        let mut iterations = 0;

        for i in 0..max_iterations {
        
            if !window.is_open() || window.is_key_down(Key::Escape) { break; } else {

            let chemical_ponential = get_chemical_potential(&order_parameter.vals, dx, dy);
            order_parameter.update(Update2D::forward_euler_on(&order_parameter.vals, &chemical_ponential, dx, dy, dt));

            free_energy[i] = get_free_energy_density(&order_parameter.vals, dx, dy).sum();

            iterations += 1;

            if iterations % 100 == 0 {
                let max_order_parameter = order_parameter.vals.iter().max_by(|a, b| a.total_cmp(b)).unwrap();
                for i in 0..nx {
                    for j in 0..ny {

                        let colour = Plot2D::float_to_hex(order_parameter.vals[[i, j]]/max_order_parameter);

                        for icell in 0..cell_size { 
                            for jcell in 0..cell_size {
                                pixel_buffer[[i*cell_size + icell, j*cell_size + jcell]] = colour;
                            }
                        }
                    }
                }

                let pixels: Vec<u32> = pixel_buffer.iter().copied().collect();
                window.update_with_buffer(&pixels, cx, cy).unwrap();
                std::thread::sleep(time::Duration::from_nanos(frame_period_nanoseconds));
            }
            }
        }

    }

    pub fn float_to_hex(value: f64) -> u32 {
        let v = value.clamp(-1.0, 1.0);

        let red = if v > 0.0 { v } else { 0.0 };
        let blue = if v < 0.0 { -v } else { 0.0 };

        let r = 255 - (red * 255.0).round() as u8;
        let b = 255 - (blue * 255.0).round() as u8;
        let g = (255. * 0.8) as u8; 

        ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
    }


}
