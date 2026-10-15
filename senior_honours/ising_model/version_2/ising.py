from ising_lib import *
import argparse
import datetime
from os import devnull
from pathlib import Path

def main() -> None:

    # user arguments
    args = arguments()
        
    # file path handling
    if args.discard != True:
        time.sleep(1)
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y@%H-%M-%S")
        run_dir = Path("runs")
        run_dir.mkdir(exist_ok=True)
        save_dir = run_dir / timestamp
        save_dir.mkdir(exist_ok=True)

        # save arguments to a .txt file
        log = open(save_dir / "logs.txt", "a")
        print(f"{vars(args)}")
        print(f"{vars(args)}", file = log)
        log.close()

    else:
        save_dir = devnull
        print(f"Arguments: {vars(args)}" + "\n")
        warning = "\n!!!!! Data will not be saved. !!!!!\n"
        print("!"*(len(warning) - 2) + "\n!!!!! Data will not be saved. !!!!!\n" + "!"*(len(warning) - 2))

    # run program
    im = ising_model(args.length, args.initial_conditions, args.correlation_gap, args.warmup, args.sweeps, save_dir, args.visual, args.error_method) #type: ignore
    
    if True == args.b_simulation:
        results = im.beta_sim(args.b_range[0], args.b_range[1], args.b_resolution, args.algorithm)
        
        if 'glauber' == args.algorithm:
            im.plot_glauber(*results)
        else:
            im.plot_kawasaki(*results)

    else:
        mc = markov_chain(args.length, args.beta, args.initial_conditions)
        
        if True == args.visual:
            im.vsimulation(mc, args.sweeps, args.beta, mc.glauber if "glauber" == args.algorithm else mc.kawasaki)
        else:
            im.simulation(mc, args.sweeps, args.beta, mc.glauber if "glauber" == args.algorithm else mc.kawasaki)

def arguments() -> argparse.Namespace:
    """
    Parses user arguments using the argparse library and returns a argparse.ArgumentParser object containing the arguments.

    Returns:
        command line arguments (argparse.Namespace)
    """

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        prog = "ising.py",
        description = "Ising model simulation with Glauber and Kawasaki dynamics options. System evolves stochastically in a Markov Chain."
    )

    parser.add_argument("--discard", 
                    action="store_true",
                    help = "Discard data from the simulation.",
                    )

    parser.add_argument("-v",
                    "--visual", 
                    action="store_true",
                    help = "Show a visualisation of the spin grid evolution when simulation running.",
                    )   
    
    parser.add_argument("-bsim",
                    "--b_simulation", 
                    action="store_true",
                    help = "Perform b_res simulations from b_min to b_max and plot the heat capacity." + 
                    "If Glauber dynamics is selected the magnetic susceptibility is also plotted.",
                    )  
    
    parser.add_argument("-br",
                    "--b_range", 
                    type = float, default = [1.0, 3.0], nargs = 2,
                    help = "Range of beta values to sample (for use with --beta_simulation; beta is the thermal energy kbT).",
                    )
    
    parser.add_argument("-bres",
                    "--b_resolution", 
                    type = int, default = 50,
                    help = "Number of beta values to sample (for use with --beta_simulation).",
                    )

    parser.add_argument("-a",
                    "--algorithm",
                    required = True, 
                    type = str, choices = ['glauber', 'kawasaki'],
                    help = "Dynamics algorithm to use to evolve spin grid.",
                    )

    parser.add_argument("-e",
                    "--error_method", 
                    default = 'bootstrap',
                    type = str, choices = ['bootstrap', 'jackknife'],
                    help = "Method to calculate observable errors with.",
                    )

    parser.add_argument("-l",
                    "--length", 
                    type = int, default = 50,
                    help = "Length of the square spin grid.",
                    )
    
    parser.add_argument("-ic",
                    "--initial_conditions", 
                    type = str, choices = ['random', 'up', 'down'], default = 'random',
                    help = "Intialise the spin grid with aligned spins (up or down) or distributed randomly.",
                    )
    
    parser.add_argument("-c",
                    "--correlation_gap", 
                    type = int, default = 10,
                    help = "Number of spins in between observable measurements.",
                    )
    
    parser.add_argument("-N",
                    "--sweeps", 
                    type = int, default = 1000,
                    help = "Number of sweeps to perform (per simulation).",
                    )    
    
    parser.add_argument("--warmup", 
                    type = int, default = 100,
                    help = "Number of warmup sweeps to perform (per simulation). These sweeps are perform before the -N sweeps argument.",
                    )    
    
    parser.add_argument("-b",
                    "--beta", 
                    type = float, default = 1.,
                    help = "Inverse of the thermal energy.",
                    )    

    
    return parser.parse_args()

if __name__ == '__main__':
    main()