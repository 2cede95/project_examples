import argparse
from time import sleep
from datetime import datetime
from os import devnull
from pathlib import Path

def main() -> None:

    # user arguments
    args = arguments()

    # handle multicore case 
    if args.multicore:
        from sirs_lib_multicore import SIRS, simulation
    else:
        from sirs_lib import SIRS, simulation

    # file path handling
    if args.discard != True:
        sleep(1)
        timestamp = datetime.now().strftime("%d-%m-%Y@%H-%M-%S")
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
        warning = "\n!!!!! Data and figures will not be saved. !!!!!\n"
        print("!"*(len(warning) - 2) + warning + "!"*(len(warning) - 2))

    # run program
    sim = simulation(correlation_gap=args.correlation_gap, save_dir=save_dir) # type: ignore
    sirs = SIRS(args.length,
                args.probability_susceptible,
                args.probability_infected,
                args.probability_recovered,
                args.immune_fraction,
                (args.initial_probability_susceptible, args.initial_probability_infected, args.initial_probability_recovered))

    if args.preset is not None:

        if "absorbing" == args.preset:
            sim.absorbing(args.sweeps, args.length)
        
        elif "dynamic_equilibrium" == args.preset:
            sim.dyanmic_equilibrium(args.sweeps, args.length)
    
        elif "cyclic_wave" == args.preset:
            sim.cyclic_wave(args.sweeps, args.length)

    elif args.simulation is not None:

        if "phase_diagram" == args.simulation:
            sim.SI_RS_phase_diagram(args.resolution, args.warmup, args.sweeps, args.length)

        elif "variance_slice" == args.simulation:
            sim.SI_variance(args.resolution, args.warmup, args.sweeps, args.length)

        elif "immunity_fraction":
            sim.immunity(args.resolution, args.warmup, args.sweeps, args.length)

    else:
        sirs.run_visual(args.sweeps)
        
def arguments() -> argparse.Namespace:
    """
    Parses user arguments into an argparse.ArgumentParser object.

    Returns:
        command line arguments (argparse.Namespace)
    """

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        prog = "sirs.py",
        description = "Susceptible, Infected, Recovered, Susceptible (SIRS) simulation tool."
    )

    parser.add_argument("--discard", 
                    action="store_true",
                    help = "Discard data from the simulation.",
                    )
    
    parser.add_argument("--multicore", 
                    action="store_true",
                    help = "Faster computation but uses more compute. Leaves one cpu core free for the OS.",
                    )

    parser.add_argument("-v",
                    "--visual", 
                    action="store_true",
                    help = "Visualiser for SIRS system evolution.",
                    )   
    
    parser.add_argument("-p",
                        "--preset",
                        type = str, default = None, choices=["absorbing", "dynamic_equilibrium", "cyclic_wave"],
                        help = "Preset SIRS simulations. Recommend length of >=100 for cyclic_wave.")
    
    parser.add_argument("-s",
                        "--simulation",
                        type = str, default = None, choices=["phase_diagram", "variance_slice", "immunity_fraction"],
                        help = "Three preset experiments/simulations.")
    
    parser.add_argument("-r",
                        "--resolution",
                        type = int, default = 20,
                        help = "Resolution for probability/immune fraction grid. For use with preset simulations (--simulation argument).")
    
    parser.add_argument("-l",
                    "--length", 
                    type = int, default = 50,
                    help = "Length of the square SIRS grid.",
                    )
    
    parser.add_argument("-c",
                    "--correlation_gap", 
                    type = int, default = 10,
                    help = "Number of sweeps in between observable measurements.",
                    )
    
    parser.add_argument("-N",
                    "--sweeps", 
                    type = int, default = 1000,
                    help = "Number of sweeps to perform (per simulation).",
                    )    
    
    parser.add_argument("--warmup", 
                    type = int, default = 100,
                    help = "Number of warmup sweeps to perform (per simulation).",
                    )    

    parser.add_argument("-p_s",
                    "--probability_susceptible", 
                    type = float, default = 0.5,
                    help = "Probability for a state to go from recovered to susceptible.",
                    )   

    parser.add_argument("-p_i",
                    "--probability_infected", 
                    type = float, default = 0.5,
                    help = "Probability for a state to go from susceptible to infected.",
                    )   

    parser.add_argument("-p_r",
                    "--probability_recovered", 
                    type = float, default = 0.5,
                    help = "Probability for a state to go from susceptible to recovered.",
                    )   

    parser.add_argument("-ip_s",
                    "--initial_probability_susceptible", 
                    type = float, default = 1/3,
                    help = "Probability for a site to be in the susceptible state in the initial SIRS grid.",
                    )   

    parser.add_argument("-ip_i",
                    "--initial_probability_infected", 
                    type = float, default = 1/3,
                    help = "Probability for a site to be in the infected state in the initial SIRS grid.",
                    )   

    parser.add_argument("-ip_r",
                    "--initial_probability_recovered", 
                    type = float, default = 1/3,
                    help = "Probability for a site to be in the recovered state in the initial SIRS grid.",
                    ) 

    parser.add_argument("-if",
                    "--immune_fraction", 
                    type = float, default = 0,
                    help = "Probability for a site to be immune in the SIRS grid. Applied after the susceptible, infected and recovered sites are distributed.",
                    ) 
    
    return parser.parse_args()

if __name__ == '__main__':
    main()