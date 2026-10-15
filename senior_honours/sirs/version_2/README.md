# SIRS Model Simulation
## Description
Python code which simulates the spread of disease in humans using the Susceptible Infected Recovered Susceptible (SIRS) model.

## Examples
Example outputs have been organised in ./examples. The cmd prompt used to produce the examples are detailed below:

DISCLAIMER: Some of the simulations will take 30+ minutes to run.

./phase_diagram/* : "python3 sirs.py -s phase_diagram --multicore" (Warning! This will probably take ~30+ minutes to run, hardware/OS dependent. Recommended to change parameters, see -h flag for more information.)
./variance_slice : "python3 sirs.py -s variance_slice --multicore --resolution 50"
./immune_fraction : "python3 sirs.py -s immunity_fraction --multicore --resolution 50"
s
NOTE: "python3 ..." is the method used to call python on some windows machine. Other common aliases are "python ..." or "py ..." please consult your python installation for more information.

## Implementation
Three files are provided; sirs_lib.py, sirs_lib_multicore.py and sirs.py.
    - sirs_lib.py contains the functional code for simulations and SIRS logic. This code runs on a single thread.
    - sirs_lib_multicore.py contains the same functionality as sirs_lib.py with the code rewritten to make use of multiple cores.
    - sirs.py contains the user interface logic for running simulations.

Where possible, vectorisation was introduced using the Numpy package. Multicore implementations take advantage of the python standard installation's multiprocessing library.

For in-depth documentation on function/classes please see their docstrings inside the respective files.

Any data related to the simulation will be output into the ./runs directory. A unique directory is created with a timestamp corresponding to the time of program execution. Figures are saved .png images and numerical data are saved as .npy files. These files are not human readable, they must be accessed via Numpy's np.load() function for processing in Python. This is intentional as it has not been specified that the files must be human readable and this file format takes up less space, avoids complications with multiple dimensional Numpy arrays and is easy to use in Python.

## Requirements
Both sirs.py and sirs_lib.py/sirs_lib_multicore.py must be present in the same directory for the code in sirs.py to run. If sirs_lib.py is not present and --multicore is not used then an error will be thrown and the program will not work. Similarly, if sirs_lib_multicore.py is not present but --multicore is used then an error will be thrown and the program will not work.

Graphs are produced by following the syntax in the provided examples. The sirs.py file comes with a help menu which can be accessed by executing "python3 sirs.py -h" (or one of the other python aliases). The help menu provides information on all of the command line arguments and what they do. A copy of the help menu is pasted at the bottom of this file.

### Dependencies
Below is a list of dependencies (including version numbers) for which this code is known to operate correctly:
    - Python 3.11.9
    - Numpy 2.2.3
    - Matplotlib 3.10.1

## Other Information
### Purpose
Code produced for the second checkpoint hand-in as part of the Modelling and Visualisation course at the University of Edinburgh School of Physics and Astronomy in the 2025/26 academic year.

### -h output
usage: sirs.py [-h] [--discard] [--multicore] [-v] [-p {absorbing,dynamic_equilibrium,cyclic_wave}]
               [-s {phase_diagram,variance_slice,immunity_fraction}] [-r RESOLUTION] [-l LENGTH] [-c CORRELATION_GAP]
               [-N SWEEPS] [--warmup WARMUP] [-p_s PROBABILITY_SUSCEPTIBLE] [-p_i PROBABILITY_INFECTED]
               [-p_r PROBABILITY_RECOVERED] [-ip_s INITIAL_PROBABILITY_SUSCEPTIBLE]
               [-ip_i INITIAL_PROBABILITY_INFECTED] [-ip_r INITIAL_PROBABILITY_RECOVERED] [-if IMMUNE_FRACTION]

Susceptible, Infected, Recovered, Susceptible (SIRS) simulation tool.

options:
  -h, --help            show this help message and exit
  --discard             Discard data from the simulation. (default: False)
  --multicore           Faster computation but uses more compute. Leaves one cpu core free for the OS. (default:
                        False)
  -v, --visual          Visualiser for SIRS system evolution. (default: False)
  -p {absorbing,dynamic_equilibrium,cyclic_wave}, --preset {absorbing,dynamic_equilibrium,cyclic_wave}
                        Preset SIRS simulations. Recommend length of >=100 for cyclic_wave. (default: None)
  -s {phase_diagram,variance_slice,immunity_fraction}, --simulation {phase_diagram,variance_slice,immunity_fraction}
                        Three preset experiments/simulations. (default: None)
  -r RESOLUTION, --resolution RESOLUTION
                        Resolution for probability/immune fraction grid. For use with preset simulations (--simulation
                        argument). (default: 100)
  -l LENGTH, --length LENGTH
                        Length of the square SIRS grid. (default: 50)
  -c CORRELATION_GAP, --correlation_gap CORRELATION_GAP
                        Number of sweeps in between observable measurements. (default: 10)
  -N SWEEPS, --sweeps SWEEPS
                        Number of sweeps to perform (per simulation). (default: 1000)
  --warmup WARMUP       Number of warmup sweeps to perform (per simulation). (default: 100)
  -p_s PROBABILITY_SUSCEPTIBLE, --probability_susceptible PROBABILITY_SUSCEPTIBLE
                        Probability for a state to go from recovered to susceptible. (default: 0.5)
  -p_i PROBABILITY_INFECTED, --probability_infected PROBABILITY_INFECTED
                        Probability for a state to go from susceptible to infected. (default: 0.5)
  -p_r PROBABILITY_RECOVERED, --probability_recovered PROBABILITY_RECOVERED
                        Probability for a state to go from susceptible to recovered. (default: 0.5)
  -ip_s INITIAL_PROBABILITY_SUSCEPTIBLE, --initial_probability_susceptible INITIAL_PROBABILITY_SUSCEPTIBLE
                        Probability for a site to be in the susceptible state in the initial SIRS grid. (default:
                        0.3333333333333333)
  -ip_i INITIAL_PROBABILITY_INFECTED, --initial_probability_infected INITIAL_PROBABILITY_INFECTED
                        Probability for a site to be in the infected state in the initial SIRS grid. (default:
                        0.3333333333333333)
  -ip_r INITIAL_PROBABILITY_RECOVERED, --initial_probability_recovered INITIAL_PROBABILITY_RECOVERED
                        Probability for a site to be in the recovered state in the initial SIRS grid. (default:
                        0.3333333333333333)
  -if IMMUNE_FRACTION, --immune_fraction IMMUNE_FRACTION
                        Probability for a site to be immune in the SIRS grid. Applied after the susceptible, infected
                        and recovered sites are distributed. (default: 0)