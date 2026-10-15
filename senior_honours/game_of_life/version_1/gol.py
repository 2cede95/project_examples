from gol_library import *
import argparse
import time
from pathlib import Path
import datetime

def main():
    # system arguments via argparse
    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        prog = "gol.py",
        description = "John Conways Game of life simulation."
    )

    parser.add_argument("-d",
                        "--dimension", 
                        type = int, default = 50,
                        help = "Length of the square board to simulate.",
                        )
    parser.add_argument("-i", "--initialization",
                        type = str,
                        default = "uniform",
                        choices = ('uniform', 'void', 'glider', 'blinker', 'beehive'),
                        help = "Initial conditions for the game."
                        )
    parser.add_argument("-p", "--preset",
                        type = str,
                        choices = ('glider', 'blinker', 'beehive', 'uniform'),
                        help = "Preset system evolutions for checkpoint. These simulations are visual onyl so record no data."
                        )
    parser.add_argument("-t", "--sim_type",
                        type = str,
                        choices = ('equilibriation_times', 'glider_velocity'),
                        help = "Available observable measurement simulations"
                        )
    parser.add_argument("--history_length",
                        type = int,
                        default = 100,
                        help = "Number of boards to be kept in memory for pattern recognition to determine if the board is in equilibrium. Higher = better but uses more system memory."
                        ) 
    parser.add_argument("-u", "--update_period",
                        type = float,
                        default = 0.001,
                        help = "How often to update the plot."
                        )   
    parser.add_argument("-m", "--update_method",
                        type = str,
                        default = 'visual',
                        choices = ('visual', 'fast'),
                        help = "Whether or not to show a visual representation of the board updating."
                        )        
    parser.add_argument("-n", "--n_simulations",
                        type = int,
                        default = 100,
                        help = "Number of simulations to complete for the histogram of equilibriation times."
                        )      
    parser.add_argument("--N_max",
                        type = int,
                        default = 15000,
                        help = "Maximum simulation length"
                        )        


    args = parser.parse_args()

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
    
    # user interface logic
    if args.preset:
        a = simulation(dimension=args.dimension,
                       init_type=args.preset,
                       history_length=args.history_length,
                       update_period=args.update_period,
                       update_method='visual',
                       sims=args.n_simulations,
                       log_dir=log_dir,
                       log_path=log_path,
                       N_max=args.N_max)
        a.visual_only()

    elif args.sim_type:
        
        a = simulation(dimension=args.dimension,
           init_type=args.initialization,
           history_length=args.history_length,
           update_period=args.update_period,
           update_method=args.update_method,
           sims=args.n_simulations,
           log_dir=log_dir,
           log_path=log_path,
           N_max=args.N_max)

        if args.sim_type == 'glider_velocity':
            a.glider_velocity(args.N_max)
        
        elif args.sim_type == 'equilibriation_times':
            a.equilibriation_times()

    else:
        a = simulation(dimension=args.dimension,
           init_type=args.initialization,
           history_length=args.history_length,
           update_period=args.update_period,
           update_method=args.update_method,
           sims=args.n_simulations,
           log_dir=log_dir,
           log_path=log_path,
           N_max=args.N_max)
        a.simulate(verbose=True)

    log = open(log_path, "a")
    print(f"finish: {timestamp}")
    print(f"finish: {timestamp}", file=log)
    log.close()

if __name__ == '__main__':
    main()

