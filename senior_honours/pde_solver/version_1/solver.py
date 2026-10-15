from solver_library import *
import argparse
import time
import datetime
from pathlib import Path
from os import devnull


def main():
    """
    Code which handles user interaction/choices logic.
    """


    # argument handling
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
        print(f"Arguments: {vars(args)}")
        print(f"Arguments: {vars(args)}", file = log)
        log.close()

    else:
        save_dir = devnull
        print(f"Arguments: {vars(args)}" + "\n")
        warning = "\n!!!!! Data will not be saved. !!!!!\n"
        print("!"*(len(warning) - 2) + "\n!!!!! Data will not be saved. !!!!!\n" + "!"*(len(warning) - 2))

    # main user interface logic
    
    # detemine solver from arguments
    integrator = integrator_tools()
    if 'jacobi' == args.solver:
        solver = integrator.jacobi
        solver_name = 'Jacobi'
    elif 'gauss_seidel' == args.solver:
        solver = integrator.gauss_seidel
        solver_name = 'Gauss-Seidel'
    elif 'sor' == args.solver:
        solver = integrator.successive_over_relax
        solver_name = 'SOR'
    else: 
        solver_name = None

    # initialise fluid equation object
    fluid_eq = fluid_equations(args.dimension, args.t_step, args.x_step, args.initial_value, args.noise, args.tolerance, args.boundary_value, save_dir, solver_name)
    
    # determine optimal_w
    if True == args.optimal_w:
        
        if 'maxwell' == args.equation:
            sim = fluid_eq.vertical_wire
        elif 'poisson' == args.equation:
            sim = fluid_eq.point_charge
        else:
            print("Cannot solve 2D Cahn Hilliard equation with 3D SOR implementation. Must choose one of: {'maxwell', 'poisson'}") 
            return 0

        fluid_eq.determine_optimal_w(sim, args.N_max, args.w_range[0], args.w_range[1], args.w_res)
        return 0

    # handle simulation visualisation
    if True == args.visual:

        if 'cahn_hilliard' == args.equation:
            fluid_eq.cahn_hilliard_2d_visual(args.N_max)
            return 0
        
        elif 'maxwell' == args.equation:
            fluid_eq.vertical_wire_visual(args.N_max, solver=solver, w=args.relaxation)
            return 0
        
        elif 'poisson' == args.equation:
            fluid_eq.point_charge_visual(args.N_max, solver=solver, w=args.relaxation)
            return 0

    elif False == args.visual:

        if 'cahn_hilliard' == args.equation:
            fluid_eq.cahn_hilliard_2d(args.N_max)
            return 0
        
        elif 'maxwell' == args.equation:
            fluid_eq.vertical_wire(args.N_max, solver=solver, w=args.relaxation)
            return 0
        
        elif 'poisson' == args.equation:
            fluid_eq.point_charge(args.N_max, solver=solver, w=args.relaxation)
            return 0

    return 0

def arguments():
    """
    Parses user arguments using the argparse library and returns a argparse.ArgumentParser object containing the arguments.

    Returns:
        command line arguments (argparse.ArgumentParser)
    """
    stability_cond = 0.25

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

    parser.add_argument("-e",
                    "--equation", 
                    type = str, 
                    required = True,
                    choices = ('cahn_hilliard', 'maxwell', 'poisson'),
                    help = "Type of equation to solve.",
                    )   
    
    parser.add_argument("--w_range", 
                    type = float, nargs = 2,
                    default = [0., 2.],
                    help = "Range of relaxation parameters to sample.",
                    )   

    parser.add_argument("--w_res", 
                    type = int,
                    default = 100,
                    help = "Number of points to sample in relexation parameter space.",
                    )  
    
    parser.add_argument("--optimal_w", 
                    action="store_true",
                    help = "Determine optimal relaxation parameter for a given equation.",
                    )  

    parser.add_argument("-s",
                    "--solver", 
                    type = str, 
                    choices = ('jacobi', 'gauss_seidel', 'sor'),
                    default = 'jacobi',
                    help = "Solver to use for the simulation = 'maxwell' and simulation = 'poisson' arguments.",
                    )   

    parser.add_argument("-d",
                    "--dimension", 
                    type = int, default = 100,
                    help = "Dimension of simulation.",
                    )
    
    parser.add_argument("-x",
                    "--x_step", 
                    type = float, default = 1.,
                    help = f"Length scale to discretise space with. t_step/(x_step)**2 must exceed {stability_cond} for stable execution. ",
                    )
    
    parser.add_argument("-t",
                    "--t_step", 
                    type = float, default = 0.01,
                    help = f"Time scale to discretise with. t_step/(x_step)**2 must exceed {stability_cond} for stable execution in 2D.",
                    )
    
    parser.add_argument("-w",
                    "--relaxation", 
                    type = float, default = 1.97,
                    help = f"Relaxation parameter for the solver = 'SOR' argument.",
                    )

    parser.add_argument("-iv",
                    "--initial_value", 
                    type = float, default = 0.,
                    help = f"Initialisation of order parameter. A small amount of noise is added to this value - noise can be changed in fluid_equations.init_ivp. If no value is set then the order parameter is initialized with noise only.",
                    )
    
    parser.add_argument("-n",
                    "--noise", 
                    type = float, default = 0.1,
                    help = "Multiplier added to noise on initial values.",
                    )   
    
    parser.add_argument("-bv",
                    "--boundary_value", 
                    type = float, nargs = 6, default = [None, None, None, None, None, None],
                    help = f"Initialisation of order parameter boundary values. Can be changed in fluid_equations.init_bvp.",
                    )
    
    parser.add_argument("-tol",
                    "--tolerance", 
                    type = float, default = 1e-4,
                    help = "Minimum value between successive time steps for convergence condition.",
                    )

    parser.add_argument("--discard", 
                    action="store_true",
                    help = "Discard data from the run.",
                    )
    
    parser.add_argument("-N",
                    "--N_max", 
                    type = int, default = 10000,
                    help = "Maximum number of iterations to perform. Note: simulation = 'cahn_hilliard' is likely to hit this number before convergence.",
                    )  
      
    
    return parser.parse_args()

if __name__ == '__main__':
    main()
