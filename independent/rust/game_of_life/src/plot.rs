use minifb::{Key, Window, WindowOptions};
use rand::{RngExt, rng};
use crate::{MAX_ITERATIONS, game_board::GameBoard, evolve::Evolve, simulation::Simulation};
use core::time;
use std::{collections::HashSet, iter::StepBy};
use plotters::prelude::*;


pub struct Plot {}

impl Plot {

    pub fn gol_animation(dimension: usize, cell_size: usize, frame_period_nanoseconds: u64) {

        let mut game_board = GameBoard::new(dimension);
        let d = dimension;
        let canvas_d = d*cell_size;
        let white = 0xffffff;
        let black = 0x000000;

        let mut window = Window::new("John Conway's Game of Life",
                                    canvas_d, canvas_d,
                                    WindowOptions::default()
                                    ).unwrap();


        let mut pixel_buffer = vec![0u32; canvas_d*canvas_d];
        let mut iterations = 0;
        let mut board_history: HashSet<GameBoard> = HashSet::new();

        while window.is_open()
           && !window.is_key_down(Key::Escape)
           && iterations <= MAX_ITERATIONS 
           && !board_history.contains(&mut game_board)
            {

            board_history.insert(game_board.clone());
            Evolve::sweep_board(&mut game_board);
            iterations += 1;

            for i in 0..d {
                for j in 0..d {

                    let colour = if game_board.cells[i*d + j] == 0u32 {white} else {black};

                    for icell in 0..cell_size {
                        for jcell in 0..cell_size {
                            pixel_buffer[(i*cell_size + icell)*canvas_d + (j*cell_size + jcell)] = colour;
                        }
                    }
                }
            }

            window.update_with_buffer(&pixel_buffer, canvas_d, canvas_d).unwrap();
            std::thread::sleep(time::Duration::from_nanos(frame_period_nanoseconds));

        }

    }

    fn data_range(data: &Vec<u32>) -> (u32, u32) {
        let mut sorted = data.clone();
        sorted.sort();
        (sorted[0], sorted[sorted.len() - 1])
        
    }

    fn bin_data(data: &Vec<u32>, bins: usize) -> Vec<u32> {
        
        let (min, max) = Plot::data_range(&data);
        let bin_step = (max-min)/(bins as u32) + 1;
        let mut binned = vec![0u32; data.len()];
        for i in 0..data.len() {
            binned[i] = min + ((data[i] - min) / bin_step) * bin_step;
        }
        binned
    }

    pub fn histogram(data: &Vec<u32>) -> () {
        
        let label_size = 35;
        let margin = 35;
        let bins = 20;
        let binned = Plot::bin_data(data, bins);
        let (_, max) = Plot::data_range(&binned);
        let caption = "Game of Life Equilibriation Time";

        let canvas = BitMapBackend::new("histogram.png", (1000, 1000)).into_drawing_area(); // NEVER USE SVG BACKEND FOR HISTOGRAMS
        canvas.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&canvas)
            .set_left_and_bottom_label_area_size(label_size)
            .margin(margin)
            .caption(caption, ("sans-serif", 15))
            .build_cartesian_2d((0..max).into_segmented(), 0u32..((max + max/4) as u32))
            .unwrap();

        chart.configure_mesh()
            .disable_mesh()
            .draw()
            .unwrap();
        
        chart.draw_series(Histogram::vertical(&chart)
            .style(RED.filled())
            .margin(15)
            .data(binned.iter().map(|x|(*x, 1))))
            .unwrap();

        canvas.present().unwrap();
    }

}