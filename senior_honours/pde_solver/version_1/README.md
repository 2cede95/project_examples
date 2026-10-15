# Differential Equation Solver
Python program which uses a range of solvers based on the centered difference scheme to solve discretised differential equations. 

## solver_library.py
Contains the centered difference operations in the integrator_tools class and physics equations/ preset solver functions in the fluid_equations class.

## solver.py
Contains user interface logic.

## Requirements
Both solver.py and solver_library.py must be in the same directory. 

Dependencies:
    - python 3.14.2
    - numpy 2.4.2
    - matplotlib 3.10.8

## Output
For the preset simulation options the following plots are presented:
    maxwell, poisson:
        - A slice in the i-j plane of the order parameter through the centre of the k axis (2D contour plot);
        - (optional, --visual) Live 2D heatmap plot of the above as the equation is being solved;
        - A plot of the vector field using matplotlib's 3D quiver plot (indexed with 'ij');
        - A 1D plot of the strength of the potential as a function of distance from the centre;
        - A 1D plot fo the strength of the vector field as a function of distance from the centre.
    cahn_hilliard:
        - a 1D plot of the free energy of the state as a function of iterations;
        - (optional, --visual) Live 2D heatmap plot of the order parameter as the equation is being solved.
    optimal_w:
        - a 1D plot of the iterations to converge as a function of relaxation parameter. 

The above plots are all saved at ./runs/{timestamp}/figures in a .png format.

The maxwell and poisson solvers save the vector field numpy arrays and order parameter numpy arrays in the .npy numpy binary file format at ./runs/{timestamp}/data

A log file containing a dictionary of arguments is saved at ./runs/{timestamp} in a txt format.

## Usage
    solver.py [-h] [-v] -e {cahn_hilliard,maxwell,poisson} [--w_range W_RANGE W_RANGE] [--w_res W_RES]
                 [--optimal_w] [-s {jacobi,gauss_seidel,sor}] [-d DIMENSION] [-x X_STEP] [-t T_STEP] [-w RELAXATION]
                 [-iv INITIAL_VALUE] [-n NOISE]
                 [-bv BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE]
                 [-tol TOLERANCE] [--discard] [-N N_MAX]

Library for solving differential equation using the centered difference discretisation schemes. Comes with preset simulations displaying physical solutions to electromagnetism and fluid dynamics equations.

## Arguments
  -h, --help            show this help message and exit
  -v, --visual          Show a visualisation of the order parameter evolution whilst a simulation is running.
                        (default: False)
  -e {cahn_hilliard,maxwell,poisson}, --equation {cahn_hilliard,maxwell,poisson}
                        Type of equation to solve. (default: None)
  --w_range W_RANGE W_RANGE
                        Range of relaxation parameters to sample. (default: [0.0, 2.0])
  --w_res W_RES         Number of points to sample in relexation parameter space. (default: 100)
  --optimal_w           Determine optimal relaxation parameter for a given equation. (default: False)
  -s {jacobi,gauss_seidel,sor}, --solver {jacobi,gauss_seidel,sor}
                        Solver to use for the simulation = 'maxwell' and simulation = 'poisson' arguments. (default:
                        jacobi)
  -d DIMENSION, --dimension DIMENSION
                        Dimension of simulation. (default: 100)
  -x X_STEP, --x_step X_STEP
                        Length scale to discretise space with. t_step/(x_step)**2 must exceed 0.25 for stable
                        execution. (default: 1.0)
  -t T_STEP, --t_step T_STEP
                        Time scale to discretise with. t_step/(x_step)**2 must exceed 0.25 for stable execution in 2D.
                        (default: 0.01)
  -w RELAXATION, --relaxation RELAXATION
                        Relaxation parameter for the solver = 'SOR' argument. (default: 1.89)
  -iv INITIAL_VALUE, --initial_value INITIAL_VALUE
                        Initialisation of order parameter. A small amount of noise is added to this value - noise can
                        be changed in fluid_equations.init_ivp. If no value is set then the order parameter is
                        initialized with noise only. (default: 0.0)
  -n NOISE, --noise NOISE
                        Multiplier added to noise on initial values. (default: 0.001)
  -bv BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE, --boundary_value BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE BOUNDARY_VALUE
                        Initialisation of order parameter boundary values. Can be changed in fluid_equations.init_bvp.
                        (default: [None, None, None, None, None, None])
  -tol TOLERANCE, --tolerance TOLERANCE
                        Minimum value between successive time steps for convergence condition. (default: 0.0001)
  --discard             Discard data from the run. (default: False)
  -N N_MAX, --N_max N_MAX
                        Maximum number of iterations to perform. Note: simulation = 'cahn_hilliard' is likely to hit
                        this number before convergence. (default: 10000)
