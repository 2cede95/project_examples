use crate::arguments::parse_args;

mod vector_calculus;
mod update;
mod lattice;
mod simulation;
mod plot;
mod arguments;

fn main() {
    
    let args = parse_args();

    plot::Plot2D::cahn_hilliard_animation(
        args.iterations,
        args.nx,
        args.ny,
        args.initial_value,
        args.noise,
        args.dx,
        args.dy,
        args.dt,
        args.cell_size,
        args.refresh_period_nano
    );
}
