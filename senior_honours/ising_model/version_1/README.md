# Ising Model Simulation
Simulates a 2D lattice of spins using the Ising model with periodic boundary conditions. Spin states are updated using a Markov chain with either the Glauber or Kawasaki algorithms (Metropolis update rules).

## dynamics.py
Python file containing all of the classes used in the model. All model related code is inside this file.

## run.py 
Python file containing the user interface logic. Imports dynamics.py and runs the simulation based on user arguments (or default settings).

### Usage
  run.py [-h] [--visual_only {True,False}] [-t TEMPERATURE] [-f PLOT_FREQ] [--visualiser VISUALISER] [-a {glauber,kawasaki}]
              [--temp_range TEMP_RANGE TEMP_RANGE] [--temp_step TEMP_STEP] [-d DIMENSION] [--sweep_depth SWEEP_DEPTH]
              [--sweep_warmup SWEEP_WARMUP] [--correlation_gap CORRELATION_GAP] [-s {aligned,random}] [-e {bootstrap,jackknife}]


### Arguments
  -h, --help            show this help message and exit
  --visual_only {True,False}
                        View the Ising lattice update as the Markov Chain progresses but do no data collection.
                        (default: False)
  -t TEMPERATURE, --temperature TEMPERATURE
                        For user with --visual_only flag. Temperature to visualise the Markov Chain evolution at.
                        (default: 3.0)
  -f PLOT_FREQ, --plot_freq PLOT_FREQ
                        How often the visualiser draws a new frame. For use with --visual_only and --visualiser.
                        (default: 10)
  --visualiser VISUALISER
                        Display a plot of the spins in the Ising lattice as the simulation compute the Markov Chain.
                        (default: False)
  -a {glauber,kawasaki}, --algorithm {glauber,kawasaki}
                        Step algorithm for the Markov Chain. Example: run.py glauber (default: glauber)
  --temp_range TEMP_RANGE TEMP_RANGE
                        Temperature range to iterate through in the simulation. Example: run.py --temp_range min max
                        (default: (1.0, 3.0))
  --temp_step TEMP_STEP
                        Stepsize to iterate through the temp_range. (default: 0.1)
  -d DIMENSION, --dimension DIMENSION
                        Side length of the square Ising lattice. (default: 50)
  --sweep_depth SWEEP_DEPTH
                        Number of sweeps to compute at each temperature value in temp_range. (default: 1000)
  --sweep_warmup SWEEP_WARMUP
                        Number of sweeps to compute before temperature iterations and data collection begins.
                        (default: 100)
  --correlation_gap CORRELATION_GAP
                        Number of sweeps to perform between observable measurements to avoid correlation. (default:
                        10)
  -s {aligned,random}, --start {aligned,random}
                        Start the simulation with aligned or random spins. Example: run.py --start aligned (default:
                        random)
  -e {bootstrap,jackknife}, --error_method {bootstrap,jackknife}
                        Method used to estimate observable measurement errors. Example: run.py --error_method
                        bootstrap (default: bootstrap)

### Output
- if --visual_only is True (default)
    - log file saved at ./{timestamp}
    - NOTE: the log file displays values to 2 decimal places only; for small temp_steps the log file will not report changes to the thermal energy any smaller than 0.01 kbT/J

- if --visual_only is False
    - Data saved as individual .txt files at ./{timestamp}/data
    - Figure file saved at ./{timestamp}
    - log file saved at ./{timestamp}

- progress updates printed to console 

 
### Dependencies
 - python libs (3.14.2)
    - run.py
    - dynamics.py
    - argparse
    - sys
    - datetime
    - pathlib
    - time
    - os
 - numpy 2.2.3
 - matplotlib 3.10.1

 ### Units
 - The magnetic coupling constant J (units of joules) is set to 1 in this simulation so that units are:
    - Energy: [no units] 
    - Thermal Energy: [kbT/J]
    - Heat capacity: [kb]
    - Susceptibility: [no units]
    - Magnetisation: [no units]

### Physical Constants
 - kb = Boltzmann Constant [Joules/Kelvin]
 - J = Coupling Energy [Joules] 

### Errors
 - Errors for observables mean energy, mean absolute magnetisation, susceptibility and heat capacity computed using statistical resampling methods (Bootstrap or Jackknife, see Arguments)
 - Error for the critical temperature is quoted as the reading error of the matplotlib plot (to 2 decimal places) i.e. ± temp_step