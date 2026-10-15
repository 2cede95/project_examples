from sirs_library import *
from pathlib import Path
import time
import datetime
import argparse

def main():
    """
    Main function for the program. Handles user input logic and runs the corresponding code in the sirs_library.py file.
    """
    # argument handling
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        prog = "sirs.py",
        description = "SIRS model simulating the evolution of a pandemic schochastically"
    )

    parser.add_argument("-d",
                    "--dimension", 
                    type = int, default = 50,
                    help = "Length of the square board to simulate.",
                    )
    parser.add_argument("-N", "--sweeps_per_sim",
                        type = int,
                        default = 1000,
                        help = "Number of sweeps to conduct per simulation."
                        ) 
    parser.add_argument("-w", "--warmup_sweeps",
                        type = int,
                        default = 100,
                        help = "Number of sweeps to conduct before each simulation."
                        ) 
    parser.add_argument("-r", "--resolution",
                        type = float,
                        default = 0.05,
                        help = "Resolution of probabilities/immune_fraction for their respective simulations."
                        ) 
    parser.add_argument("--meas_freq",
                        type = int,
                        default = 1,
                        help = "How often (in sweeps) to take a mesaurement."
                        ) 
    parser.add_argument("-v", "--visual",
                        type = bool,
                        default = False,
                        help = "Whether or not to show a visual representation of the board updating. No visual runs much faster."
                        )        
    parser.add_argument("--multicore",
                    type = bool,
                    default = False,
                    help = "Use the pathos.multiprocessing library to utilize hardware fully."
                    )
    parser.add_argument("-t", "--sim_type",
                    type = str,
                    default = "visual_only",
                    choices = ("visual_only", "absorbing", "dynamic_equilibrium", "oscillating", "full_phase_diagram", "slice_phase_diagram", "immunity"),
                    help = "Which simulation to produce -> See README for details."
                    )     
    parser.add_argument("--p_init_s",
                    type = float,
                    default = 1/3,
                    help = "Probaility that a site will initially be susceptible."
                    )   
    parser.add_argument("--p_init_i",
                    type = float,
                    default = 1/3,
                    help = "Probaility that a site will initially be infecfed."
                    )   
    parser.add_argument("--p_init_r",
                    type = float,
                    default = 1/3,
                    help = "Probaility that a site will initially be Recovered."
                    )      
    parser.add_argument("--p_update_s",
                    type = float,
                    default = 0.3,
                    help = "Probaility that a site will update from recovered->suscept."
                    )  
    parser.add_argument("--p_update_i",
                    type = float,
                    default = 0.3,
                    help = "Probaility that a site will update from suscept->infected."
                    )      
    parser.add_argument("--p_update_r",
                    type = float,
                    default = 0.4,
                    help = "Probaility that a site will update from infected->recovered."
                    )      
    parser.add_argument("-f", "--immune_frac",
                    type = float,
                    default = 0,
                    help = "Fraction of sites to initialize as immune. (i.e. infinitely effective vaccine)"
                    ) 
    parser.add_argument("--plot_freq",
                    type = int,
                    default = 5,
                    help = "Number of sweeps between plot updates."
                    )  
    

    args = parser.parse_args()
    p = (args.p_update_s, args.p_update_i, args.p_update_r)
    p_init = (args.p_init_s, args.p_init_i, args.p_init_r)
    
    # probability conditions enforcements
    for i in range(0, len(p)):
        if p[i] < 0 or p_init[i] < 0:
            print(f"Probabilities must be non-negative. Current input: p_init = {p_init} and p = {p}.")
            sys.exit()
    if (args.p_init_s + args.p_init_i + args.p_init_r) != 1:
        print(f"Initial probabilities must sum to 1. The default case is equal probabilities of (1/3, 1/3, 1/3). Current input: {p_init}.")
        sys.exit()


    # create log directory (if doesnt exist)
    time.sleep(1)
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y@%H-%M-%S")
    run_dir = Path("runs")
    run_dir.mkdir(exist_ok=True)
    log_dir = run_dir / f"{timestamp}"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"logs.txt"
    log = open(log_path, "a")
    print("\n" + "*"*10 + "\n" + f"Arguments: {vars(args)}" + "\n" + "*"*10 + "\n")
    print("\n" + "*"*10 + "\n" + f"Arguments: {vars(args)}" + "\n" + "*"*10 + "\n", file = log)
    print(f"start: {timestamp}")
    print(f"start: {timestamp}", file=log)
    log.close()

    # main logic handling
    if args.sim_type == 'visual_only':
        
        a = lattice(dimension=args.dimension,
                        visual=True,
                        p_init=p_init,
                        p=p,
                        plot_freq = args.plot_freq,
                        immune_fraction=args.immune_frac)
        a.sweep_method(args.sweeps_per_sim)

    elif args.sim_type == 'absorbing' or args.sim_type == 'dynamic_equilibrium' or args.sim_type == 'oscillating':
        a = simulation(dimension=args.dimension,
                        visual=True,
                        multicore=False,
                        p_init=p_init,
                        plot_freq = args.plot_freq,
                        log_path=log_path,
                        log_dir=log_dir)   
        a.preset_visualisation(state=args.sim_type,N_sweeps=args.sweeps_per_sim)   

    elif args.sim_type == 'full_phase_diagram':
        a = simulation(dimension=args.dimension,
            visual=args.visual,
            multicore=args.multicore,
            p_init=p_init,
            plot_freq = args.plot_freq,
            log_path=log_path,
            log_dir=log_dir)  
        a.phase_full(N=args.sweeps_per_sim,N_warmup=args.warmup_sweeps,res=args.resolution)

    elif args.sim_type == 'slice_phase_diagram':
        a = simulation(dimension=args.dimension,
            visual=args.visual,
            multicore=args.multicore,
            p_init=p_init,
            plot_freq = args.plot_freq,
            log_path=log_path,
            log_dir=log_dir)  
        a.phase_slice(N=args.sweeps_per_sim,N_warmup=args.warmup_sweeps,res=args.resolution, meas_freq=args.meas_freq,min_p=0.2, max_p=0.5)

    elif args.sim_type == 'immunity':
        a = simulation(dimension=args.dimension,
            visual=args.visual,
            multicore=args.multicore,
            p_init=p_init,
            plot_freq = args.plot_freq,
            log_path=log_path,
            log_dir=log_dir)  
        a.immunity(N=args.sweeps_per_sim,N_warmup=args.warmup_sweeps,meas_freq=args.meas_freq,res=args.resolution)

    log = open(log_path, "a")
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y@%H-%M-%S")
    print(f"finish: {timestamp}")
    print(f"finish: {timestamp}", file=log)
    log.close()

# need this line for multiprocessing to work correctly on windows
if __name__ == "__main__":
    main()
