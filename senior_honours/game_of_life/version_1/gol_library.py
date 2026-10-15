import numpy as np
import matplotlib.pyplot as plt
from os import devnull
import time
from sys import exit

class board():
    """ 
    Class which contains a game board and all its associated operations.

    Attributes:
        self.squares (numpy.ndarray): dead/alive values for a game board
        self.dimension (int): length of a side of the square board
    
    """

    def __init__(self, dimension, preset = 'uniform', inherit = None):
        """
        Init method for the board class

        Args:
            dimension (int): length of a side of the square board
            preset (str): method of distribution for alive/dead cells during initialisation; includes the 
                          blinker and glider presets.
            inherit (numpy.ndarray): inherit an array from an outside source. Dimension automatically computed
                                     in this case
        """
        self.dimension = dimension

        if inherit is not None:
            self.squares = inherit
            self.dimension = inherit.shape[0]

        else:
            match str(preset.lower()):
                case 'uniform':
                    self.squares = self.board_init_uniform(dimension)
                case 'void':
                    self.squares = self.board_init_void(dimension)
                case 'blinker':
                    # 1 1 1
                    self.squares = self.board_init_void(dimension)
                    half_way_point = int(dimension/2)
                    self.squares[half_way_point, half_way_point-1:half_way_point+2:1] = 1
                    if self.dimension < 5:
                        print("System too small for a blinker. Please use a dimension of 5 or greater.")
                        exit()
                case 'glider':
                    #   1
                    #     1
                    # 1 1 1
                    self.squares = self.board_init_void(dimension)
                    half_way_point = int(dimension/2)
                    self.squares[half_way_point+1, half_way_point] = 1
                    self.squares[half_way_point, half_way_point+1] = 1
                    self.squares[half_way_point-1, half_way_point-1:half_way_point+2] = 1
                    if self.dimension < 5:
                        print("System too small for a glider. Please use a dimension of 5 or greater.")
                        exit()

                case 'beehive':
                    #   1
                    # 1   1
                    # 1   1
                    #   1
                    self.squares = self.board_init_void(dimension)
                    half_way_point = int(dimension/2)
                    self.squares[half_way_point+1, half_way_point] = 1
                    self.squares[half_way_point, half_way_point-1] = 1
                    self.squares[half_way_point, half_way_point+1] = 1
                    self.squares[half_way_point-1, half_way_point-1] = 1
                    self.squares[half_way_point-1, half_way_point+1] = 1
                    self.squares[half_way_point-2, half_way_point] = 1
                    if self.dimension < 5:
                        print("System too small for a beehive. Please use a dimension of 5 or greater.")
                        exit()                   
          
        return None


    def board_init_uniform(self, dimension):
        """
        Initialise a board with uniformly random alive/dead cells.

        0 -> dead
        1 -> alive

        Returns:
            numpy.ndarray
        """

        board = np.random.choice((0, 1), size=(dimension, dimension))

        return board
    
    def board_init_void(self, dimension):
        """
        Initialises a board where all sites are dead (0).

        Returns
            numpy.ndarray
        """

        board = np.zeros(shape = (dimension, dimension))

        return board
        
    
    def count_neighbours(self):
        """
        Counts alive nearest neighbours for each element in the internal board object.

        Uses numpy.roll to "roll" the board 8 times so that each site sees all of its neighbours at the site. Summing over the rolled boards
        yields a numpy array which has the same shape as the internal board object with a count of nearest neighbours at each site. This relies 
        on the fact that board values are either 1 or 0.

        Returns:
            numpy.ndarray
        """
        current_board = self.squares 


        nearest_neighbours = np.zeros_like(current_board)
        
        # up down left right
        up = np.roll(current_board, 1, axis = 0)
        down = np.roll(current_board, -1, axis = 0)
        left = np.roll(current_board, -1, axis = 1) 
        right = np.roll(current_board, 1, axis = 1)

        # corners
        up_left = np.roll(up, -1, axis = 1)
        up_right = np.roll(up, 1, axis = 1)
        down_left = np.roll(down, -1, axis = 1)
        down_right = np.roll(down, 1, axis = 1)

        nearest_neighbours = sum((up, down, left, right, up_left, up_right, down_left, down_right))

        return nearest_neighbours
    
    def copy(self):
        """
        Returns a copy of the internal board.
        """
        return board(dimension = self.dimension, inherit = self.squares.copy())


class game():
    """
    Class which contains a game and the associated methods for updating the game according to its rules.

    Attributes:
        self.board (board): internal board variable
        self.active_squares (int): internal count of active squares on the board
        self.iterations (int): internal count of board updates elapsed.
        self.board_cache (int): number of previous boards to store in memory; if this number is too low then oscillating states will be missed 
        self.board_history(numpy.ndarray): numpy array containing previoous boards

    """

    def __init__(self, dimension, init_type, history_length = 1000):
        """
        Init method for the game class.

        Args:
            dimension (int): length of the side to initialize the internal board object with
            init_type (str): method of distribution for the alive and dead cells in the initial board state
            history_length (int): number of previous boards to store in memory; if this number is too low then oscillating states will be missed 

        """

        self.board = board(dimension, init_type)
        self.active_squares = 0
        self.iterations = 0

        # board history 
        self.board_cache = history_length
        self.board_history = np.empty(shape = (self.board_cache), dtype=board)

        return None


    def update_board(self):
        """ 
        Update the internal board once according to the Game of Life rules (John Conway, 1970):
            #1: Any alive cell with less than 2 alive neighbours dies.
            #2: Any alive cell with 2 or 3 alive neighbours lives on to the next step.
            #3: Any alive cell with more than 3 alive neighbours dies.
            #4: Any dead cell with exactly 3 alive neighbours becomes alive.
        
        Update is done in parallel using numpy.roll.
        """
        
        current_board = self.board
        current_squares = current_board.squares
        next_squares = self.board.squares.copy()
        nearest_neighbours = current_board.count_neighbours()

        #1: Any live cell with less than 2 live neighbours dies.
        rule_1 = (next_squares == 1) & (nearest_neighbours < 2) 
        next_squares[rule_1]-=1

        #2: Any live cell with 2 or 3 live neighbours lives on to the next step.
        rule_2 = (next_squares == 1) & ((nearest_neighbours == 2) | (nearest_neighbours == 3))
        next_squares[rule_2] = 1

        #3: Any live cell with more than 3 live neighbours dies. 
        rule_3 = (next_squares == 1) & (nearest_neighbours > 3)
        next_squares[rule_3]-=1

        #4: Any dead cell with exactly 3 live neighbours becomes alive.
        rule_4 = (next_squares == 0) & (nearest_neighbours == 3)
        next_squares[rule_4]+=1

        # update the number of active squares
        active = (current_squares != next_squares)
        self.active_squares = np.sum(active) 
        self.iterations+=1

        # update the internal board 
        # need [:] otherwise the board is not replaced in memory
        current_squares[:] = next_squares

        return None
    
    def centre_of_mass(self):
        """
        Returns the centre of mass coordinates of all live cells in the game.
        
        Returns:
            numpy.ndarray
        """

        alive_x, alive_y = np.where(self.board.squares == 1)
        if len(alive_x) == 0 & len(alive_y) == 0:
            print("No alive cells.")
            return None 
    
        centre_of_mass = np.array((np.mean(alive_x), np.mean(alive_y)))

        return centre_of_mass



class simulation():
    """
    Class which contains the relevant methods for extracting information out of a game (i.e. plots, observables, etc.)
    
    Attributes:
        self.dimension (int): length of the side of the board in the internal game object
        self.history_length (int): number of previous boards to store in memory; if this number is too low then oscillating states will be missed
        self.update_period (float): time between board updates (in seconds)
        self.init_type (str): method of distribution for the alive and dead cells in the internal game object's initial board state 
        self.sims (int): number of simulations update until equilibrium (or N_cutoff)
        self.game (game): internal game object needs to be an internal object for memory scope in the plotting process
        self.update_method (func): whether to draw a matplotlib plot of the board whilst updating or not ('fast' or 'visual')
        self.fig, self.ax, self.line, self.cbar: internal plotting variables for use with matplotlib.pyplot
        self.N_max (int): maximum number of updates to do on a single board, safety clause incase history_length does not capture a large period oscillator
    """

    def __init__(self, dimension, init_type = 'uniform', history_length = 100, update_period = 0.001, update_method = 'visual', sims = 100, N_max = 15000, log_dir = devnull, log_path=devnull):
        """
        Init method for the simulation class.

        Args:
            dimension (int): length of the side of the board in the internal game object
            init_type (str): method of distribution for the alive and dead cells in the internal game object's initial board state 
            history_length (int): number of previous boards to store in memory; if this number is too low then oscillating states will be missed
            update_period (float): time between board updates (in seconds)
            N_max (int): maximum number of updates to do on a single board, safety clause incase history_length does not capture a large period oscillator 
            sims (int): number of simulations update until equilibrium (or N_cutoff)
            update_method (str): whether to draw a matplotlib plot of the board whilst updating or not
            game (game): internal game object, needs to be an internal object for memory scope in the plotting process
            log_dir (pathlib.Path): directory to save data and figures
            log_path (pathlib.Path): path to save stdout logs
        """

        # log file stuff
        self.log_dir = log_dir
        self.log_path = log_path

        # simulation parameters
        self.dimension = dimension
        self.history_length = history_length
        self.update_period = update_period
        self.init_type = init_type
        self.sims = sims
        self.N_max = N_max
        self.game = game(dimension=dimension, init_type=init_type, history_length=history_length)

        # fast or visual board updates
        self.update_method = update_method

        match update_method:
            case 'visual':
                self.update_method = self.visualise

                # interactive plot
                plt.ion()
                self.fig, self.ax = plt.subplots(1, 1)
                self.line = self.ax.imshow(self.game.board.squares, cmap = 'gray_r', vmin = 0, vmax = 1, origin = 'lower')
                self.ax.set_title(r"Game Board Visualisation", fontweight = "bold")
                self.cbar = self.fig.colorbar(self.line, ax = self.ax, ticks = [0, 1])
            case 'fast':
                 self.update_method = self.compute

        return None
    
    
    def visualise(self, steps, game):
        """
        Progress the game of life a certain number of steps (with a matplotlib visualisation of the board).

        Args:
            steps (int): number of steps to progress the game
            game (game): the game object to update
        """
        
        the_game = game

        for _ in range(0, int(steps)):

            # update board history
            the_game.board_history[the_game.iterations%the_game.board_cache] = the_game.board.copy()

            # update board and plot
            self.line.set_data(the_game.board.squares)
            the_game.update_board()
            the_game.iterations+=1

            plt.pause(self.update_period)

        return None

    def compute(self, steps, game):
        """
        Progress the game of life a specific number of steps.
        
        Args:
            steps (int): number of steps to progress the game
            game (game): game object to update
        """
        the_game = game

        for _ in range(0, int(steps)):

            # update board history
            the_game.board_history[the_game.iterations%the_game.board_cache] = the_game.board.copy()
    
            # update board
            the_game.update_board()
            the_game.iterations+=1
    

        return None
    
    def visual_only(self):
        """
        Initializes a game and updates it until N_max iterations is reached.

        For visual purposes only.
        """

        the_game = game(dimension = self.dimension, init_type=self.init_type, history_length=self.history_length)
        self.update_method(self.N_max, the_game) 

        return None

    def simulate(self, verbose=False):
        """
        Tracks active squares as a function of time (iterations) until the Game of Life reaches equilibrium and returns
        the number of iterations elapsed at the point of equilibrium OR returns N_max if equilibrium is not reached in
        N_max iterations.

        If the iteration limit has been reached (i.e. self.N_max) then the specific game is stopped. This avoids
        the case where an oscillator's period exceeds the history length and thus causes the game to never reach equiilibrium.

        Args:
            verbose (bool): print no. iterations to stdout and log file

        Returns:
            int
        """

        the_game = game(dimension = self.dimension, init_type=self.init_type, history_length=self.history_length)

        if verbose:
            # open log file
            log = open(self.log_path, "a")

        # count active squares per update
        active_squares = []

        # main loop
        sim_iterations = 0
        start_time = time.time()

        for i in range(0, self.N_max):
            
            # update the board
            self.update_method(1, the_game)

            # count active squares
            active_squares.append(the_game.active_squares)
            sim_iterations+=1   

            if verbose and (i+1)%100 == 0:
                elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
                print(f"{sim_iterations} iterations completed in {elapsed_time}")     
                print(f"{sim_iterations} iterations completed in {elapsed_time}", file=log)    

            # check against history
            occupied_history = the_game.board_history[the_game.board_history != None]
            for i in occupied_history:
                
                # check if all elements are the same 
                if (i.squares == the_game.board.squares).all():
                    
                    if verbose:

                        print(f"equilibrium reached in {sim_iterations} iterations")     
                        print(f"equilibrium reached in {sim_iterations} iterations", file=log)   
                        log.close()

                    # board no longer active
                    return sim_iterations
                
        if verbose:
            # close log file
            print(f"equilibrium not reached in {sim_iterations} iterations")     
            print(f"equilibrium not reached in {sim_iterations} iterations", file=log)  
            log.close()
        
        return sim_iterations
    
    def equilibriation_times(self):
        """
        Performs a number of simulations (detemined by self.sims) and plots a histogram of the times (iterations) to equilibriate.
        """

        # open log file
        log = open(self.log_path, "a")

        self.game = game(dimension=self.dimension, init_type=self.init_type, history_length=self.history_length)
        equi_iters = np.empty(shape=(self.sims), dtype = int)
        start_time = time.time()

        for i in range(0, self.sims):
            equi_iters[i] = self.simulate()

            if (i+1)%(self.sims/10) == 0:
                # update every 10%
                elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
                print(f"{i+1} games complete in {elapsed_time} ({(i+1)/self.sims*100:.2f}%)")
                print(f"{i+1} games complete in {elapsed_time} ({(i+1)/self.sims*100:.2f}%)", file=log)
        
        # create new figure and turn off animation
        plt.ioff()
        fig, ax = plt.subplots(1)

        # close log file
        log = open(self.log_path, "a")

        # save data
        self.data = (equi_iters,)
        self.data_names = ("equi_iters",)
        self.save_data()
        
        # plot
        ax.hist(equi_iters, color="tab:red", edgecolor = "black", bins=50)
        ax.set_title("Equilibriation Time (Iterations)", fontweight = "bold")
        ax.set_xlabel("No. Iterations")
        ax.set_ylabel("Frequency")

        # save figure
        fig_dir = self.log_dir
        fig_dir.mkdir(exist_ok=True)
        fig_label = "equilibriation"
        extension = "png"
        file_path = fig_dir /  f"{fig_label}-figure.{extension}"
        plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()

        return None

    
    def glider_velocity(self, time_period):
        """
        Track the position of a glider over time and computes its velocity.

        Accounts for boundary conditions by ignoring points where any of the gliders alive cells are at the boundary.

        Args:
            time_period (int): length of time to track the glider for (in iterations)

        return
            float
        """

        # open log file
        log = open(self.log_path, "a")

        the_game = game(dimension=self.dimension,
                         init_type='glider',
                         history_length=self.history_length)
        the_board = the_game.board

        # need to use lists as measurements are paused at the boundary
        pos = []
        distances = [0]
        velocities = [0]
        times = [0]
        displacements = [0]

        pos.append(the_game.centre_of_mass())
        distance = 0
        time_elapsed = 0
        skipped = False
        n_skipped = 0
        displacement = 0

        for i in range(1, time_period):

            # update board
            self.update_method(1, the_game)

            # boundary check
            boundary = np.array([the_board.squares[0][:].any() == 1,
                        the_board.squares[-1][:].any() == 1,
                        the_board.squares[:][0].any() == 1,
                        the_board.squares[:][-1].any() == 1]).any()

            if boundary:
                n_skipped+=1
                skipped = True
                continue 

            # journey metrics
            pos.append(the_game.centre_of_mass())

            #if passed a boundary then need to do two iterations before calculating seperation again
            if skipped:
                skipped = False
                continue

            # update distance velocity
            ind = i - n_skipped
            separation = np.linalg.norm(pos[ind] - pos[ind-1])
            displacement = np.linalg.norm(pos[ind] - pos[0])
            displacements.append(displacement)
            
            distance += separation
            distances.append(distance)
            time_elapsed+=1
            times.append(time_elapsed)
            velocity = distance/time_elapsed
            velocities.append(velocity)

            print(f"Current velocity: {velocity:.3f} squares/second")
            print(f"Current velocity: {velocity:.3f} squares/second", file=log)
        
        # save data
        self.data = (velocities, pos, distances, times)
        self.data_names = ("velocities", "positions", "distances", "times")
        self.save_data()

        # close log file
        final_velocity = distance/time_elapsed
        print(f"Current velocity: {velocity:.3f} squares/second")

        # plot
        fig, ax = plt.subplots(1, 1)
        ax.plot(times, distances, c="tab:red")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Displacement")
        ax.set_title("Glider's Centre of Mass Displacement vs Time", fontweight = "bold")

        # save figure
        fig_dir = self.log_dir
        fig_dir.mkdir(exist_ok=True)
        fig_label = "glider_velocity"
        extension = "png"
        file_path = fig_dir /  f"{fig_label}-figure.{extension}"
        plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show plot
        plt.ioff()
        plt.show()

        return final_velocity
    
    def save_data(self):
        """
        Save the data stored in the internal data variable.

        Data is stored in the local log_dir path. Each quantity has its own .txt file.
        """

        if self.data:
            # file extension to save data as
            extension = "txt"

            # esnure directory for this run exists
            log_dir = self.log_dir
            log_dir.mkdir(exist_ok=True)

            # create directory for data
            data_dir = log_dir / "data"
            data_dir.mkdir(exist_ok=True)

            # save the files
            for i, name in enumerate(self.data_names):
                file_path = data_dir / f"{name}.{extension}"
                np.savetxt(file_path, self.data[i])

        else:
            print("No data to save")
        
        return None

    