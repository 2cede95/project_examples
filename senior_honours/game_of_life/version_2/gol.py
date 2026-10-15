from gol_lib import *
import argparse
from time import sleep
from datetime import datetime
from os import devnull
from pathlib import Path

def main() -> None:

    # user arguments
    args = arguments()

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
    sim = simulation(save_dir) # type: ignore (save_dir type cases handled in simulation.__init__)

    if args.simulation is not None:
        
        gol = game_of_life(args.length, args.preset, args.visual)

        if "histogram" == args.simulation:
            sim.equilib_histogram(args.simulations, args.max_iterations, args.length)
        
        elif "glider_velocity" == args.simulation:
            sim.glider_velocity(args.max_iterations, args.length)

    else:
        gol = game_of_life(args.length, args.preset, visual = True)
        gol.run_method(args.max_iterations)
        
def arguments() -> argparse.Namespace:
    """
    Parses user arguments into an argparse.ArgumentParser object.

    Returns:
        command line arguments (argparse.ArgumentParser)
    """

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        prog = "gol.py",
        description = "Game of Life simulation tool."
    )

    parser.add_argument("--discard", 
                    action="store_true",
                    help = "Discard data from the simulation.",
                    )
    
    parser.add_argument("-v",
                    "--visual", 
                    action="store_true",
                    help = "Visualiser for game of life system evolution.",
                    )   
    
    parser.add_argument("-p",
                        "--preset",
                        type = str, default = "random", choices=["random", "blinker", "beehive", "glider", "mf_glider"],
                        help = "Preset Game of Life initial conditions.")
    
    parser.add_argument("-s",
                        "--simulation",
                        type = str, default = None, choices=["histogram", "glider_velocity"],
                        help = "Two preset experiments/simulations.")
    
    parser.add_argument("-l",
                    "--length", 
                    type = int, default = 50,
                    help = "Length of the square Game of Life grid.",
                    )
    
    parser.add_argument("-N_max",
                    "--max_iterations", 
                    type = int, default = 15000,
                    help = "Maximum number of iterations to perform on the Game of Life board.",
                    )    
    
    parser.add_argument("-N",
                    "--simulations", 
                    type = int, default = 1000,
                    help = "Number of GOL simulations to conduct for the histogram --simulation option.",
                    )    
    
    return parser.parse_args()

if __name__ == '__main__':
    main()