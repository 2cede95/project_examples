# Game of Life
Python program to simulate the game of life according to John Conway's update rules (1970). The rules are as follows:
    #1: Any alive cell with less than 2 alive neighbours dies.
    #2: Any alive cell with 2 or 3 alive neighbours lives on to the next step.
    #3: Any alive cell with more than 3 alive neighbours dies.
    #4: Any dead cell with exactly 3 alive neighbours becomes alive.

## gol_library.py
Contains all of the simulation classes and logic.

## gol.py
Contains user interface logic.

## Requirements
Both gol.py and gol_library.py must be in the same directory. 

Dependencies:
    - python 3.14.2
    - numpy 2.4.2
    - matplotlib 3.10.8

## Output
For the simulation types glider_velocity and equilibriation_time, data and figures are written to disk in the directory ./run (which is created if it does not already exist). Inside this directory a sub-directory is created with a timestamp of when the simulation was run. Inside this sub-directory the data, saved as .txt files by numpy.savetxt, and the figures, saved as .png with a dpi of 400, is saved. 

For ALL runs of the program a log file is created as a .txt file inside ./runs/{timestamp}. This file contains all of the information that is printed to stdout through the run of the program.

-t equilibriation_times 
    -> .png histogram of equilibriation time at a dpi of 400
    -> a .txt file for equilibriation times and a log file

-t glider_velocity 
    -> .txt files for velocities, positions, distances and times
    -> .png subplot containing glider centre of mass distance vs time and glider centre of mass displacement vs time (time in iterations)

WARNING: if the program is terminated before the number of iterations in the --N_max argument has been completed no data is stored.

Default behaviour with no arguments is to run until equilibrium with a random uniform initialization.

## Usage
     gol.py [-h] [-d DIMENSION] [-i {uniform,void,glider,blinker,beehive}] [-p {glider,blinker,beehive}]
              [-t {equilibriation_times,glider_velocity}] [--history_length HISTORY_LENGTH] [-u UPDATE_PERIOD]
              [-m {visual,fast}] [-n N_SIMULATIONS] [--N_max N_MAX]
## Arguments

     -h, --help            show this help message and exit
     -d, --dimension DIMENSION
                           Length of the square board to simulate. (default: 50)
     -i, --initialization {uniform,void,glider,blinker,beehive}
                           Initial conditions for the game. (default: uniform)
     -p, --preset {glider,blinker,beehive,uniform}
                           Preset system evolutions for checkpoint. These simulations are visual onyl so record no
                           data. (default: None)
     -t, --sim_type {equilibriation_times,glider_velocity}
                           Available observable measurement simulations (default: None)
     --history_length HISTORY_LENGTH
                           Number of boards to be kept in memory for pattern recognition to determine if the board
                           is in equilibrium. Higher = better but uses more system memory. (default: 100)
     -u, --update_period UPDATE_PERIOD
                           How often to update the plot. (default: 0.001)
     -m, --update_method {visual,fast}
                           Whether or not to show a visual representation of the board updating. (default: visual)
     -n, --n_simulations N_SIMULATIONS
                           Number of simulations to complete for the histogram of equilibriation times. (default:
                           100)
     --N_max N_MAX         Maximum simulation length (default: 15000)
