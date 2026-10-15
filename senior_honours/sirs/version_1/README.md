# SIRS Model Simulation
Python program to simulate the evolution of a pandemic using stochastic nearest neighbour update rules with periodic boundary conditions. The simulation has a mutlticore processing option which significantly speeds up the code. This is optional: if pathos is not installed the code will run as long as the --multicore argument is False (default). The pathos library is only imported once --multicore is True.

## sirs_library.py
Contains all of the simulation classes and logic.

## sirs.py
Contains user interface logic.

## Requirements
Both sirs.py and sirs_library.py must be in the same directory. 

Dependencies:
    - python 3.14.2
    - numpy 2.4.2
    - matplotlib 3.10.8
    - pathos 0.3.5 (optional: for multicore processing)

## Output
For the visual_only, absorbing, dynamic_equilibrium and oscillating simulations no data is written to disk. However, a matplotlib plot will be drawn on the display. For the rest of the --sim-type arguments a directory will be created called at ./runs (if not already there) and and inside this directory each run has its own sub-directory which is named after the timestamp at which the program was executed. The sub-directory houses all of the data/figures produced by the program. Numerical data is stored as a .txt file using numpy's np.savetxt for easy importing. Figures are stored as .png files with a dpi of 400. 

-t visual_only, absorbing, dynamic_equilibrium, oscillating 
    -> .txt log file in ./runs/{timestamp}

-t full_phase_diagram -> .txt log file in ./runs/{timestamp}
    -> .png contour plot of mean fraction of infected sites at a dpi of 400
    -> .png heatmap plot of variance of infected sites at a dpi of 400
    -> .txt files with mean infected sites; variance of infected sites; all three probabilities and the mean infected fraction of sites.

-t slice_phase_diagram 
    -> .txt log file in ./runs/{timestamp}
    -> .png 1D plot (with error bars) of mean fraction of infected sites and. 1D plot of variance of infected sites for the probability slice (both at a dpi of 400)
    -> .txt files with mean infected sites; variance of infected sites; all three probabilities; the mean infected fraction of sites and the errors on both the aforementioned variance and means

-t immunity 
    -> .txt log file in ./runs/{timestamp}
    -> .png 1D plot of mean infected fraction of sites against immunity fraction at a dpi of 400
    -> .txt files containing mean infected sites; all three probabilities; mean infected fraction of sites and the immunity fractions

Default behaviour with no arguments is to run a simulation with p(r->s) = 0.3, p(s->i) = 0.3, p(i->r) = 0.4 and immune_frac = 0 for --sweeps_per_sim iterations. A log file is created at ./runs.

The error bars presented on the sliced variance plots are calculated via bootstrapping. 

## Usage
     sirs.py [-h] [-d DIMENSION] [-N SWEEPS_PER_SIM] [-w WARMUP_SWEEPS] [-r RESOLUTION]
              [--meas_freq MEAS_FREQ] [-v VISUAL] [--multicore MULTICORE]
              [-t {visual_only,absorbing,dynamic_equilibrium,oscillating,full_phase_diagram,slice_phase_diagram,immunity}]
              [--p_init_s P_INIT_S] [--p_init_i P_INIT_I] [--p_init_r P_INIT_R] [--p_update_s P_UPDATE_S]
              [--p_update_i P_UPDATE_I] [--p_update_r P_UPDATE_R] [-f IMMUNE_FRAC] [--plot_freq PLOT_FREQ]

## Arguments
     -h, --help            show this help message and exit
     -d, --dimension DIMENSION
                           Length of the square board to simulate. (default: 50)
     -N, --sweeps_per_sim SWEEPS_PER_SIM
                           Number of sweeps to conduct per simulation. (default: 1000)
     -w, --warmup_sweeps WARMUP_SWEEPS
                           Number of sweeps to conduct before each simulation. (default: 100)
     -r, --resolution RESOLUTION
                           Resolution of probabilities/immune_fraction for their respective simulations. (default:
                           0.05)
     --meas_freq MEAS_FREQ
                           How often (in sweeps) to take a mesaurement. (default: 1)
     -v, --visual VISUAL   Whether or not to show a visual representation of the board updating. No visual runs
                           much faster. (default: False)
     --multicore MULTICORE
                           Use the pathos.multiprocessing library to utilize hardware fully. (default: False)
     -t, --sim_type {visual_only,absorbing,dynamic_equilibrium,oscillating,full_phase_diagram,slice_phase_diagram,immunity}
                           Which simulation to produce -> See README for details. (default: visual_only)
     --p_init_s P_INIT_S   Probaility that a site will initially be susceptible. (default: 0.3333333333333333)
     --p_init_i P_INIT_I   Probaility that a site will initially be infecfed. (default: 0.3333333333333333)
     --p_init_r P_INIT_R   Probaility that a site will initially be Recovered. (default: 0.3333333333333333)
     --p_update_s P_UPDATE_S
                           Probaility that a site will update from recovered->suscept. (default: 0.3)
     --p_update_i P_UPDATE_I
                           Probaility that a site will update from suscept->infected. (default: 0.3)
     --p_update_r P_UPDATE_R
                           Probaility that a site will update from infected->recovered. (default: 0.4)
     -f, --immune_frac IMMUNE_FRAC
                           Fraction of sites to initialize as immune. (i.e. infinitely effective vaccine)
                           (default: 0)
     --plot_freq PLOT_FREQ
                        Number of sweeps between plot updates. (default: 5)                    
                    
### Further Argument Info                   
                    
    Description of what the various sim_type arguments do:

    visual_only: random initialisation, system evolves for --sweeps_per_spin iterations

    dynamic_equilibrium: preset initial condition and update probabilities which result in dynamic equilibrium at a system size of (50x50)

    oscillating: preset initial condition and update probabilities which result in an oscillating state with a system size of (50x50) 

    absorbing: preset initial condition and update probabilities which result in an absorbing state with a system size of (50x50) 

    full_phase_diagram: iterate through two sets probability value pairs (recovered -> susceptible and susceptible -> infected) and produce a contour plot of the average infected site fraction as well as a heat map for the variance of infected sites. Both of these probabilities sets range from 0 -> 1 with a step size of --resolution

    slice_phase_diagram: iterate through one set of probability values (susceptible -> infected) and plot the mean infected sites with error bars computed using bootstrapping. Value range from 0.2 -> 0.5. This can be changed in the sirs.py file where the arguments to simulation.phase_slice_multi and phase_slice_single include min_p and max_p floats for a range of min_p -> max_p. The nomenclature x_single / x_multi has been used throughout to distinguish multicore computations from single core computations.

    immunity: fix all update probabilities at 0.5 and change the degree of freedom to be the fraction of total immune sites. Immune sites are immutable for the entirety of the simulation. They are initialised randomly after the distribution of susceptible, infected and recovered sites in the lattice object __init__.