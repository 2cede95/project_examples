# John Conway's Game of Life Simulation
## Description
Python code which executes John Conway's Game of Life update rules on a grid.

## Examples
Example outputs have been organised in ./examples. The cmd prompt used to produce the examples are detailed below:

DISCLAIMER: Some of the simulations will take 30+ minutes to run.

Arbitrary game of life run : "python3 gol.py --visual -l 50 -p random --discard"
    - a board will be presented and the game will evolve according to the update rules. The game will stop automatically when a steady or oscillating state is reached OR if the --simulations integer count of iterations is reached.

examples/histogram/* : "python3 gol.py -s histogram"
examples/glider_velocity/* : "python3 gol.py -s glider_velocity"

NOTE: "python3 ..." is the method used to call python on some windows machine. Other common aliases are "python ..." or "py ..." please consult your python installation for more information.

## Implementation
Two files are provided; gol_lib.py and gol.py.
    - gol_lib.py contains the functional code for simulations and Game of Life logic.
    - gol.py contains the user interface logic for running simulations.

Where possible, vectorisation was introduced using the Numpy package.

For in-depth documentation on function/classes please see their docstrings inside the respective files.

Any data related to the simulation will be output into the ./runs directory. A unique directory is created with a timestamp corresponding to the time of program execution. Figures are saved .png images and numerical data are saved as .npy files. These files are not human readable, they must be accessed via Numpy's np.load() function for processing in Python. This is intentional as it has not been specified that the files must be human readable and this file format takes up less space, avoids complications with multiple dimensional Numpy arrays and is easy to use in Python.

## Requirements
Both gol.py and gol_lib.py must be present in the same directory for the code in gol.py to run. 

Graphs are produced by following the syntax in the provided examples. The gol.py file comes with a help menu which can be accessed by executing "python3 sirs.py -h" (or one of the other python aliases). The help menu provides information on all of the command line arguments and what they do. A copy of the help menu is pasted at the bottom of this file.

### Dependencies
Below is a list of dependencies (including version numbers) for which this code is known to operate correctly:
    - Python 3.11.9
    - Numpy 2.2.3
    - Matplotlib 3.10.1

## Other Information
### Purpose
Code produced for the second checkpoint hand-in as part of the Modelling and Visualisation course at the University of Edinburgh School of Physics and Astronomy in the 2025/26 academic year.

### -h output
usage: gol.py [-h] [--discard] [-v] [-p {random,blinker,beehive,glider,mf_glider}] [-s {histogram,glider_velocity}]
              [-l LENGTH] [-N_max MAX_ITERATIONS] [-N SIMULATIONS]

Game of Life simulation tool.

options:
  -h, --help            show this help message and exit
  --discard             Discard data from the simulation. (default: False)
  -v, --visual          Visualiser for game of life system evolution. (default: False)
  -p {random,blinker,beehive,glider,mf_glider}, --preset {random,blinker,beehive,glider,mf_glider}
                        Preset Game of Life initial conditions. (default: random)
  -s {histogram,glider_velocity}, --simulation {histogram,glider_velocity}
                        Two preset experiments/simulations. (default: None)
  -l LENGTH, --length LENGTH
                        Length of the square Game of Life grid. (default: 50)
  -N_max MAX_ITERATIONS, --max_iterations MAX_ITERATIONS
                        Maximum number of iterations to perform on the Game of Life board. (default: 15000)
  -N SIMULATIONS, --simulations SIMULATIONS
                        Number of GOL simulations to conduct for the histogram --simulation option. (default: 1000)