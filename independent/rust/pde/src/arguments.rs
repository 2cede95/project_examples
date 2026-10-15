use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(short = 'N',
    long = "iterations",
    default_value_t = 1_000_000usize,
    help = "Number of iterations to run the solver for.")]
    pub iterations: usize,

    #[arg(long = "width",
    default_value_t = 100usize,
    help = "Width of the spacial grid. Where boundary values are present, ghost cells are added on to this value.")]
    pub nx: usize,

    #[arg(long = "height",
    default_value_t = 100usize,
    help = "Height of the spacial grid. Where boundary values are present, ghost cells are added on to this value.")]
    pub ny: usize,

    #[arg(long = "depth",
    default_value_t = 100usize,
    help = "Depth of the spacial grid. Where boundary values are present, ghost cells are added on to this value.")]
    pub nz: usize,

    #[arg(short = 'i',
    long = "initial_value",
    default_value_t = 0f64,
    help = "Inital value for the order parameter.")]
    pub initial_value: f64,

    #[arg(long = "noise",
    default_value_t = 0.01f64,
    help = "Random noise added to the order parameter on initialisation (from RngExt distribution via rand crate).")]
    pub noise: f64,

    #[arg(short = 'x',
    long = "x_step",
    default_value_t = 1f64,
    help = "Discretisation length in the first axis.")]
    pub dx: f64,

    #[arg(short = 'y',
    long = "y_step",
    default_value_t = 1f64,
    help = "Discretisation length in the second axis.")]
    pub dy: f64,

    #[arg(short = 'z',
    long = "z_step",
    default_value_t = 1f64,
    help = "Discretisation length in the third axis.")]
    pub dz: f64,

    #[arg(short = 't',
    long = "t_step",
    default_value_t = 0.01f64,
    help = "Discretisation length in the time axis.")]
    pub dt: f64,

    #[arg(short = 's',
    long = "scale",
    default_value_t = 4usize,
    help = "Scale of pixels in the visualisation. i.e. scale of 1 means each spacial grid point is represented by 1 pixel.")]
    pub cell_size: usize,

    #[arg(short = 'r',
    long = "refresh",
    default_value_t = 1u64,
    help = "How often to draw new frames in nanoseconds.")]
    pub refresh_period_nano: u64,
}

pub fn parse_args() -> Args {
    Args::parse()
}