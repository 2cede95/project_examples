# Ising Model Simulation 
## Description
Stochastic Ising model Markov Chain simulation incorporating both Glauber and Kawasaki dynamics (in Python).

## Examples
Example outputs have been organised in ./examples. The cmd prompt used to produce the examples are detailed below:

DISCLAIMER: Some of the simulations will take 30+ minutes to run without tweaking the command line arguments. 

examples/glauber : "python3 ising.py -a glauber -bsim"
examples/kawasaki : "python3 ising.py -a kawasaki -bsim"

NOTE: "python3 ..." is the method used to call python on some windows machine. Other common aliases are "python ..." or "py ..." please consult your python installation for more information.

## Implementation
Two files are provided; ising_lib.py and ising.py.
    - ising_lib.py contains the functional code for simulations, markov chain logic and kawasaki/glauber dynamics logic.
    - ising.py contains the user interface logic for running simulations.

Where possible, vectorisation was introduced using the Numpy package.

The code has an implementation for coupling constants which are not constant (in space). A. 4d array stores the coupling constants of each sites nearest neighbours. The update rules account for this coupling constant and also a constant external field. By default the coupling constants are all 1 and the external field is 0. There are no user arguments to change this, the code must be modified manually. See markov_chain.init_jarray and markov_chain.init_harray in ising_lib.py.

The Kawasaki dynamics implementation is significantly slower (in time) than the Glauber implementation because there exists a convenient shortcut for computing the energy change when a single spin is flipped. Glauber requires just one markov_chain.energy() call but Kawasaki requires either ten or eight depending on the location of the two random sites.

For in-depth documentation on function/classes please see their docstrings inside the respective files.

Any data related to the simulation will be output into the ./runs directory. A unique directory is created with a timestamp corresponding to the time of program execution. Figures are saved .png images and numerical data are saved as .npy files. These files are not human readable, they must be accessed via Numpy's np.load() function for processing in Python. This is intentional as it has not been specified that the files must be human readable and this file format takes up less space, avoids complications with multiple dimensional Numpy arrays and is easy to use in Python.

## Requirements
Both ising.py and ising_lib.py must be present in the same directory for the code in gol.py to run. 

Graphs are produced by following the syntax in the provided examples. The spin.py file comes with a help menu which can be accessed by executing "python3 sirs.py -h" (or one of the other python aliases). The help menu provides information on all of the command line arguments and what they do. A copy of the help menu is pasted at the bottom of this file.

### Dependencies
Below is a list of dependencies (including version numbers) for which this code is known to operate correctly:
    - Python 3.11.9
    - Numpy 2.2.3
    - Matplotlib 3.10.1

## Other Information
### Purpose
Code produced for the first checkpoint hand-in as part of the Modelling and Visualisation course at the University of Edinburgh School of Physics and Astronomy in the 2025/26 academic year.

### -h output
usage: ising.py [-h] [--discard] [-v] [-bsim] [-br B_RANGE B_RANGE] [-bres B_RESOLUTION] -a {glauber,kawasaki}
                [-e {bootstrap,jackknife}] [-l LENGTH] [-ic {random,up,down}] [-c CORRELATION_GAP] [-N SWEEPS]
                [--warmup WARMUP] [-b BETA]

Ising model simulation with Glauber and Kawasaki dynamics options. System evolves stochastically in a Markov Chain.

options:
  -h, --help            show this help message and exit
  --discard             Discard data from the simulation. (default: False)
  -v, --visual          Show a visualisation of the spin grid evolution when simulation running. (default: False)
  -bsim, --b_simulation
                        Perform b_res simulations from b_min to b_max and plot the heat capacity.If Glauber dynamics
                        is selected the magnetic susceptibility is also plotted. (default: False)
  -br B_RANGE B_RANGE, --b_range B_RANGE B_RANGE
                        Range of beta values to sample (for use with --beta_simulation; beta is the thermal energy
                        kbT). (default: [1.0, 3.0])
  -bres B_RESOLUTION, --b_resolution B_RESOLUTION
                        Number of beta values to sample (for use with --beta_simulation). (default: 100)
  -a {glauber,kawasaki}, --algorithm {glauber,kawasaki}
                        Dynamics algorithm to use to evolve spin grid. (default: None)
  -e {bootstrap,jackknife}, --error_method {bootstrap,jackknife}
                        Method to calculate observable errors with. (default: bootstrap)
  -l LENGTH, --length LENGTH
                        Length of the square spin grid. (default: 50)
  -ic {random,up,down}, --initial_conditions {random,up,down}
                        Intialise the spin grid with aligned spins (up or down) or distributed randomly. (default:
                        random)
  -c CORRELATION_GAP, --correlation_gap CORRELATION_GAP
                        Number of spins in between observable measurements. (default: 10)
  -N SWEEPS, --sweeps SWEEPS
                        Number of sweeps to perform (per simulation). (default: 1000)
  --warmup WARMUP       Number of warmup sweeps to perform (per simulation). These sweeps are perform before the -N
                        sweeps argument. (default: 100)
  -b BETA, --beta BETA  Inverse of the thermal energy. (default: 1)