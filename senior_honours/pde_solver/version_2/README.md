# Partial Differential Equations Solver
## Description
Python library which can compute numerical solutions to N-dimensional partial differential equations using centered difference methods.

## Examples
Example outputs have been organised in ./examples. The cmd prompt used to produce the examples are detailed below:

DISCLAIMER: Some of the simulations will take 30+ minutes to run.

examples/cahn_hilliard_01/* : "python3 .\pde.py -e cahn_hilliard -N 250000 -iv 0.5"
examples/cahn_hilliard_02/* : "python3 .\pde.py -e cahn_hilliard -N 250000 -iv 0."
examples/poisson/* : "python3 .\pde.py -e poisson -N 10000 -iv 0.0"
examples/maxwell/* : "python3 .\pde.py -e maxwell -N 250000 -iv 0. -bv 0 0 0 0 0 0"
examples/optimal_w/* : "python3 .\pde.py -e optimal_w -N 500 --w_range 0.1 1.999 --w_res 25"

NOTE: "python3 ..." is the method used to call python on some windows machine. Other common aliases are "python ..." or "py ..." please consult your python installation for more information.

## Implementation
Two files are provided; pde_lib.py and pde.py.
    - pde_lib.py contains the functional code and can be imported and used in other projects (difference_schemes class). The simulation class must be used with pde.py. For information on what classes/functions this file contains please see the documentation in the file itself.
    - pde.py contains the user interface logic for producing simulations for the Cahn Hilliard, Poisson and Maxwell equations. The purpose of this file is to perform the checkpoint tasks. 

Where possible, vectorisation was introduced using the Numpy package. 

Any data related to the simulation will be output into the ./runs directory. A unique directory is created with a timestamp corresponding to the time of program execution. Figures are saved .png images and numerical data are saved as .npy files. These files are not human readable, they must be accessed via Numpy's np.load() function for processing in Python. This is intentional as it has not been specified that the files must be human readable and this file format takes up less space, avoids complications with multiple dimensional Numpy arrays and is easy to use in Python.

## Requirements
Both pde.py and pde_lib.py must be present in the same directory for the code in pde.py to run. Graphs are produced by following the syntax in the provided examples. The pde.py file comes with a help menu
which can be accessed by executing "python3 pde.py -h" (or one of the other python aliases). The help menu provides information on all of the command line arguments and what they do. A copy of the help menu is pasted at the bottom of this file.

### Dependencies
Below is a list of dependencies (including version numbers) for which this code is known to operate correctly:
    - Python 3.11.9
    - Numpy 2.2.3
    - Matplotlib 3.10.1
    - scipy 1.15.2

## Other Information
### Purpose
Code produced for the third checkpoint hand-in as part of the Modelling and Visualisation course at the University of Edinburgh School of Physics and Astronomy in the 2025/26 academic year.

### -h output
usage: pde.py [-h] [-v] [-s {jacobi,gauss_seidel,sor}] -e {cahn_hilliard,maxwell,poisson,optimal_w}
              [--w_range W_RANGE W_RANGE] [--w_sim {poisson,maxwell}] [--w_res W_RES] [-l LENGTH] [-x X_STEP]
              [-t T_STEP] [-w RELAXATION] [-iv INITIAL_VALUE] [-n NOISE] [-bv BOUNDARY_VALUES [BOUNDARY_VALUES ...]]
              [-tol TOLERANCE] [--discard] [-N N_MAX]

Library for solving differential equation using the centered difference discretisation schemes. Comes with preset
simulations displaying physical solutions to electromagnetism and fluid dynamics equations.

options:
  -h, --help            show this help message and exit
  -v, --visual          Show a visualisation of the order parameter evolution whilst a simulation is running.
                        (default: False)
  -s {jacobi,gauss_seidel,sor}, --solver {jacobi,gauss_seidel,sor}
                        Solver to use for maxwell and poisson equations. (default: jacobi)
  -e {cahn_hilliard,maxwell,poisson,optimal_w}, --equation {cahn_hilliard,maxwell,poisson,optimal_w}
                        Equation to solve. (default: None)
  --w_range W_RANGE W_RANGE
                        Range of relaxation parameters to sample. (default: [0.1, 2.0])
  --w_sim {poisson,maxwell}
                        Equation to determine the optimal relaxation parameter for. (default: poisson)
  --w_res W_RES         Number of points to sample in relexation parameter space. (default: 100)
  -l LENGTH, --length LENGTH
                        Length of the space grid. Each dimension has the same length. (default: 100)
  -x X_STEP, --x_step X_STEP
                        Space discretisation. (default: 1.0)
  -t T_STEP, --t_step T_STEP
                        Time discretisation. (default: 0.01)
  -w RELAXATION, --relaxation RELAXATION
                        Relaxation parameter for the SOR solver. (default: 1.97)
  -iv INITIAL_VALUE, --initial_value INITIAL_VALUE
                        Initial value of order parameter grid. (default: 0.0)
  -n NOISE, --noise NOISE
                        Multiplier added to noise on initial values for the order parameter. 0 = no noise. (default:
                        0.001)
  -bv BOUNDARY_VALUES [BOUNDARY_VALUES ...], --boundary_values BOUNDARY_VALUES [BOUNDARY_VALUES ...]
                        Order parameter boundary values for maxwell and poisson equations. (default: None)
  -tol TOLERANCE, --tolerance TOLERANCE
                        Minimum value between successive time steps for convergence condition. (default: 0.01)
  --discard             Discard data from the run. (default: False)
  -N N_MAX, --N_max N_MAX
                        Maximum number of iterations to perform. (default: 10000)