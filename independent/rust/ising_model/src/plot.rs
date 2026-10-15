use plotters::prelude::*;

pub struct Plot{}

impl Plot {

    fn y_axis_range(values: &Vec<f64>) -> std::ops::Range<f64> {
        let mut sorted = values.clone(); 
        sorted.sort_by(f64::total_cmp);
        sorted[0]..sorted[sorted.len()-1]
    }

    pub fn plot_phase_diagram (
        beta: &Vec<f64>,
        energy: &Vec<f64>,
        magnetisation: &Vec<f64>,
        heat_capacity: &Vec<f64>,
        susceptibility: &Vec<f64>,
        ) -> ()
        {

        let margin = 35;
        let label_size = 50;
        let x_ticks = 15;
        let y_ticks = 12;
        let error_message = "Plotting error.";
        let x_range: std::ops::Range<f64> = beta[0]..beta[beta.len()-1];
        let x_label = "Thermal Energy [J/kbT]";

        let canvas = SVGBackend::new("plot.svg", (750, 750)).into_drawing_area();
        canvas.fill(&WHITE).ok();

        let split_canvas = canvas.split_evenly((2, 2));
        let top_left = &split_canvas[0]; let top_right = &split_canvas[1];
        let bottom_left = &split_canvas[2]; let bottom_right = &split_canvas[3];

        let chart_prep =
            |sub_canvas: &DrawingArea<SVGBackend, plotters::coord::Shift>,
             y_range: std::ops::Range<f64>,
             y_points: &Vec<f64>,
             y_label| {
            let mut chart = ChartBuilder::on(sub_canvas)
                .margin(margin)
                .set_left_and_bottom_label_area_size(label_size)
                .build_cartesian_2d(x_range.clone(), y_range)
                .expect(error_message);

            chart.configure_mesh()
                .x_labels(x_ticks)
                .y_labels(y_ticks)
                .x_desc(x_label)
                .y_desc(y_label)
                .y_label_formatter(&|x| if (*x).abs() > 10. {format!("{:.0}", x)} else {format!("{:.1}", x)})
                .disable_mesh()
                .draw()
                .expect(error_message);
            
            chart.draw_series(LineSeries::new(beta.iter().zip(y_points.iter()).map(|(&i, &j)| (i, j)), &RED)).expect(error_message);
        };

        chart_prep(top_left, Plot::y_axis_range(energy), energy, "Mean Energy [J]");
        chart_prep(top_right, Plot::y_axis_range(heat_capacity), heat_capacity, "Heat Capacity [kb]");
        chart_prep(bottom_left, Plot::y_axis_range(magnetisation), magnetisation, "Mean Magnetisation [unitless]");
        chart_prep(bottom_right, Plot::y_axis_range(susceptibility), susceptibility, "Susceptibility [unitless]");

        canvas.present().expect(error_message);

    }






}

