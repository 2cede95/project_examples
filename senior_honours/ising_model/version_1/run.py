from dynamics import *
import argparse
from sys import exit
import datetime
from pathlib import Path
from sys import stdout

# handle system arguments using argparse library
parser = argparse.ArgumentParser(
    formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    prog = "run.py",
    description = "Simulates a 2D lattice of spins using the Ising model with periodic boundary conditions. Spin states are updated using a Markov chain with either the Glauber or Kawasaki algorithms (Metropolis update rules)."
)

# plot related
parser.add_argument('--visual_only', type = bool, default = False, choices = (True, False), help = "View the Ising lattice update as the Markov Chain progresses but do no data collection.")
parser.add_argument('-t', '--temperature', type = float, default = 3.0, help = "For user with --visual_only flag. Temperature to visualise the Markov Chain evolution at.")
parser.add_argument('-f', '--plot_freq', type = float, default = 10, help = "How often the visualiser draws a new frame. For use with --visual_only and --visualiser.")
parser.add_argument('--visualiser', type = bool, default = False, help = "Display a plot of the spins in the Ising lattice as the simulation compute the Markov Chain.")

# simulation related
parser.add_argument('-a', '--algorithm', type = str, default = 'glauber', choices = ('glauber', 'kawasaki'), help = "Step algorithm for the Markov Chain. Example: run.py glauber")
parser.add_argument('--temp_range', type = float, nargs = 2, default = (1.0, 3.0), help = "Temperature range to iterate through in the simulation. Example: run.py --temp_range min max")
parser.add_argument('--temp_step', type = float, default = 0.1, help = "Stepsize to iterate through the temp_range.")
parser.add_argument('-d', '--dimension', type = int, default = 50, help = "Side length of the square Ising lattice.")
parser.add_argument('--sweep_depth', type = int, default = 1000, help = "Number of sweeps to compute at each temperature value in temp_range.")
parser.add_argument('--sweep_warmup', type = int, default = 100, help = "Number of sweeps to compute before temperature iterations and data collection begins.")
parser.add_argument('--correlation_gap', type = int, default = 10, help = "Number of sweeps to perform between observable measurements to avoid correlation.")
parser.add_argument('-s', '--start', type = str, default = 'random', choices = ('aligned', 'random'), help = "Start the simulation with aligned or random spins. Example: run.py --start aligned")
parser.add_argument('-e', '--error_method', type = str, default = 'bootstrap', choices = ('bootstrap', 'jackknife'), help = "Method used to estimate observable measurement errors. Example: run.py --error_method bootstrap")

# handle user arguments
args = parser.parse_args()

# timestamp for saving data
# sleep to ensure unique
sleep(1)
timestamp = datetime.datetime.now().strftime("%d-%m-%Y@%H-%M-%S")

# create log directory (if doesnt exist)
log_dir = Path(f"{timestamp}")
log_dir.mkdir(exist_ok=True)
log_path = log_dir / f"logs.txt"
log = open(log_path, "a")
print("\n" + "*"*10 + "\n" + f"Arguments: {vars(args)}" + "\n" + "*"*10 + "\n")
print("\n" + "*"*10 + "\n" + f"Arguments: {vars(args)}" + "\n" + "*"*10 + "\n", file = log)
print(f"start: {timestamp}")
print(f"start: {timestamp}", file=log)
log.close()

# main user interface logic
        
if args.visual_only:
    markov_chain = markov(dimension=args.dimension,
                          algorithm=args.algorithm,
                          thermal_energy=args.temperature,
                          start=args.start,
                          plot_freq=args.plot_freq,
                          visualise=False)
    
    # start visualisation
    start_time = time.time()
    markov_chain.visualisation()
    
    # when closed by user
    log = open(log_path, "a")
    elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
    print(f"{markov_chain.sweeps} sweeps elapsed in {elapsed_time}")
    print(f"{markov_chain.sweeps} sweeps elapsed in {elapsed_time}", file=log)

    
    # exit the program
    finish_timestamp = datetime.datetime.now().strftime("%d-%m-%Y@%H-%M-%S")
    print(f"finish: {finish_timestamp}")
    print(f"finish: {finish_timestamp}", file=log)  

    # cleanup
    log.close()
    exit()

else:
    sim = simulation(dimension=args.dimension,
                    algorithm=args.algorithm,
                    thermal_energy_range=args.temp_range,
                    thermal_energy_gap=args.temp_step,
                    start=args.start,
                    plot_freq=args.plot_freq,
                    visualise=args.visualiser,
                    error_method=args.error_method,
                    log_path=log_path
    )

    # do the simulation and save
    sim.run(sweeps_per_temperature=args.sweep_depth, warmup_sweeps=args.sweep_warmup, correlation_gap=args.correlation_gap)
    data = sim.data
    sim.save_data(timestamp)
    
    # open log file again
    log = open(log_path, "a")

    # cleanup log file
    finish_timestamp = datetime.datetime.now().strftime("%d-%m-%Y@%H-%M-%S")
    print(f"finish: {finish_timestamp}")
    print(f"finish: {finish_timestamp}", file=log) 
    log.close() 

    # plot data and save
    params = sim.simulation_params()
    plot_object = plotter(data, args.algorithm, params = params)
    plot_object.show_plots(timestamp,) 


    # exit  
    exit()

# if visual_only is somehow not boolean
print("Invalid Arguments. Please use run.py --help to see the help menu.")
log.close()
