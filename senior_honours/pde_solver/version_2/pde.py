from pde_lib import *
import argparse
from time import sleep
from datetime import datetime
from os import devnull
from pathlib import Path

def main() -> None:

    # user arguments
    args = arguments()

    if None is args.boundary_values and "cahn_hilliard" == args.equation:
        args.boundary_values = (None,)*4
    elif None is args.boundary_values:
        args.boundary_values = (None,)*6

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

    sim = simulation(args.length,
                    args.initial_value,
                    args.boundary_values,
                    args.noise,
                    args.tolerance,
                    args.x_step,
                    args.t_step,
                    args.relaxation,
                    args.solver,
                    save_dir) # type: ignore
    
    if "cahn_hilliard" == args.equation:
        if True == args.visual:
            sim.cahn_hilliard_visual(args.N_max)
        else:
            sim.cahn_hilliard(args.N_max)

    elif "poisson" == args.equation:
        sim.poisson(args.N_max)

    elif "maxwell" == args.equation:
        sim.maxwell(args.N_max)

    elif "optimal_w" == args.equation:

        if "poisson" == args.w_sim:
            sim.optimal_w(args.w_range[0], args.w_range[1], args.w_res, sim.poisson, args.N_max)

        elif "maxwell" == args.w_sim:
            sim.optimal_w(args.w_range[0], args.w_range[1], args.w_res, sim.maxwell, args.N_max)
            
def arguments() -> argparse.Namespace:
    """
    Parses user arguments into an argparse.ArgumentParser object.

    Returns:
        command line arguments (argparse.ArgumentParser)
    """

    parser = argparse.ArgumentParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        prog = "pde.py",
        description = "Library for solving differential equation using the centered difference discretisation schemes. Comes with preset simulations displaying physical solutions to electromagnetism and fluid dynamics equations."
    )

    parser.add_argument("-v",
                    "--visual", 
                    action="store_true",
                    help = "Show a visualisation of the order parameter evolution whilst a simulation is running.",
                    )   

    parser.add_argument("-s",
                    "--solver", 
                    type = str, 
                    choices = ['jacobi', 'gauss_seidel', 'sor'],
                    default = 'jacobi',
                    help = "Solver to use for maxwell and poisson equations.",
                    ) 

    parser.add_argument("-e",
                    "--equation", 
                    type = str, 
                    required = True,
                    choices = ['cahn_hilliard', 'maxwell', 'poisson', 'optimal_w'],
                    help = "Equation to solve.",
                    )   
    
    parser.add_argument("--w_range", 
                    type = float, nargs = 2,
                    default = [0.1, 2.],
                    help = "Range of relaxation parameters to sample.",
                    )   
    
    parser.add_argument("--w_sim", 
                    type = str,
                    default = "poisson", choices = ["poisson", "maxwell"],
                    help = "Equation to determine the optimal relaxation parameter for.",
                    )  

    parser.add_argument("--w_res", 
                    type = int,
                    default = 100,
                    help = "Number of points to sample in relexation parameter space.",
                    )  
  
    parser.add_argument("-l",
                    "--length", 
                    type = int, default = 100,
                    help = "Length of the space grid. Each dimension has the same length.",
                    )
    
    parser.add_argument("-x",
                    "--x_step", 
                    type = float, default = 1.,
                    help = f"Space discretisation.",
                    )
    
    parser.add_argument("-t",
                    "--t_step", 
                    type = float, default = 0.01,
                    help = f"Time discretisation.",
                    )
    
    parser.add_argument("-w",
                    "--relaxation", 
                    type = float, default = 1.97,
                    help = f"Relaxation parameter for the SOR solver.",
                    )

    parser.add_argument("-iv",
                    "--initial_value", 
                    type = float, default = 0.,
                    help = f"Initial value of order parameter grid.",
                    )
    
    parser.add_argument("-n",
                    "--noise", 
                    type = float, default = 0.001,
                    help = "Multiplier added to noise on initial values for the order parameter. 0 = no noise.",
                    )   
    
    parser.add_argument("-bv",
                    "--boundary_values", 
                    type = float, nargs = '+', default = None,
                    help = f"Order parameter boundary values for maxwell and poisson equations.",
                    )
    
    parser.add_argument("-tol",
                    "--tolerance", 
                    type = float, default = 1e-2,
                    help = "Minimum value between successive time steps for convergence condition.",
                    )

    parser.add_argument("--discard", 
                    action="store_true",
                    help = "Discard data from the run.",
                    )
    
    parser.add_argument("-N",
                    "--N_max", 
                    type = int, default = 10000,
                    help = "Maximum number of iterations to perform.",
                    )  
      
    return parser.parse_args()

if __name__ == '__main__':
    main()