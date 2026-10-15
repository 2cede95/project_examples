import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from os import devnull

class game_of_life:

    def __init__(self, length: int, init: str, visual = False) -> None:
        
        if length < 5:
            print("System too small. Please use a length of 5 or greater.")
            exit() 
        self.length = length

        self.visual_freq = 0.1

        # initial conditions
        if init == 'random':
            self.grid = self.init_rarray(length)

        elif 'blinker' == init:
            # 1 1 1
            grid = self.init_earray(length)
            bisect = int(length//2)
            grid[bisect, bisect-1:bisect+2:1] = 1
            self.grid = grid

        elif 'mf_glider' == init:
            # #   1
            # #     1
            # # 1 1 1
            grid = self.init_earray(length)
            bisect = int(length//2)
            # grid[bisect-1, bisect] = 1
            # grid[bisect, bisect+1] = 1
            # grid[bisect+1, bisect-1:bisect+2] = 1
            self.grid = grid
            grid[bisect + 2, bisect - 1:bisect+3] = 1
            grid[bisect + 1, bisect+2] = 1
            grid[bisect + 1, bisect-2] = 1
            grid[bisect, bisect+2] = 1
            grid[bisect - 1, bisect-2] = 1
            grid[bisect - 1, bisect+1] = 1

        elif 'glider' == init:
            # #   1
            # #     1
            # # 1 1 1
            grid = self.init_earray(length)
            bisect = int(length//2)
            grid[bisect-1, bisect] = 1
            grid[bisect, bisect+1] = 1
            grid[bisect+1, bisect-1:bisect+2] = 1
            self.grid = grid

        elif 'beehive' == init: # beehive
            #   1
            # 1   1
            # 1   1
            #   1
            grid = self.init_earray(length)
            bisect = int(length//2)
            grid[bisect+1, bisect] = 1
            grid[bisect, bisect-1] = 1
            grid[bisect, bisect+1] = 1
            grid[bisect-1, bisect-1] = 1
            grid[bisect-1, bisect+1] = 1
            grid[bisect-2, bisect] = 1
            self.grid = grid

        if visual:
            self.run_method = self.run_visual
        else:
            self.run_method = self.run

    def init_rarray(self, length: int) -> np.ndarray:
        """
        Wrapper for numpy.random.choice.

        0 - dead
        1 - alive

        Args:
            length (int): length of the square game board

        Returns:
            randomly initialised game board (numpy.ndarray)
        """

        return np.random.choice((0, 1), size=(length, length))
    
    def init_earray(self, length: int) -> np.ndarray:
        """
        Wrapper for numpy.zeros. Initialize a board full of dead sites (zeroes).

        Args:
            Length (int): length of the square board

        Returns:
            game board full of dead sites (numpy.ndarray)
        """
        return np.zeros(shape=(length, length), dtype=int)
    
    def update(self, grid: np.ndarray) -> None:
        """
        Update a grid of alive/dead values in place using Conways's Game of Life rules.

        Uses numpy.roll to compute nearest neighbour sums to count number of infected neighbours.

        Args:
            grid (numpy.ndarray): numpy array of dead/alive values (0/1)
        """

        neighobur_sum = np.zeros_like(grid)

        # neighbour sum over axes + diagonals
        for i in range(grid.ndim):
            neighobur_sum += np.roll(a=grid, shift=1, axis=i)
            neighobur_sum += np.roll(a=grid, shift=-1, axis=i)

        neighobur_sum += np.roll(a=grid, shift=(1, 1), axis = (0, 1))
        neighobur_sum += np.roll(a=grid, shift=(1, -1), axis = (0, 1))
        neighobur_sum += np.roll(a=grid, shift=(-1, 1), axis = (0, 1))
        neighobur_sum += np.roll(a=grid, shift=(-1, -1), axis = (0, 1))
    
        # Any live cell with less than 2 live neighbours dies.
        rule_1 = neighobur_sum < 2
        grid[rule_1] = 0

        # Any live cell with 2 or 3 live neighbours lives on to the next step.
        # no update required for cells between 2 and 3
        
        # Any live cell with more than 3 live neighbours dies.
        rule_3 = neighobur_sum > 3
        grid[rule_3] = 0

        # Any dead cell with exactly 3 live neighbours becomes alive.
        rule_4 = neighobur_sum == 3
        grid[rule_4] = 1
    
    def active_sites(self, grid_1: np.ndarray, grid_2: np.ndarray) -> int:
        """
        Count active sites by comparing two grids of alive/dead values. The number of sites which are different is the number of active sites.

        Explicit conversion from boolean to integer using a mask on numpyp.ones.

        Args:
            grid_1 (numpy.nparray): One of the grids to be compared
            grid_2 (numpy.nparray): as above

        Returns:
            The number of active sites (int)
        """
        a = np.ones_like(grid_1, dtype=int)
        return np.sum(a[grid_1 != grid_2])
           
    def run(self, N_max: int) -> int:
        """
        Evolve a game of life simulation until equilibrium is reached.

        To determine when the internal grid object is at equilibrium a hashtable is created of every board update.
        The updated board is checked against all previous boards and the function is exited if a duplicate occurs.
        This catches both steady states and oscillating states. The hashtable will have legnth N_Max.

        Args:
            N_max (int): Maximum number of updates to perform on the gol board.

        Returns:
            number of iterations to equilibriate (int)       
        """
        # setup
        grid = self.grid
        update = self.update

        # loop
        history = {grid.tobytes()} # hashes of previous boards
        iterations = 0      
        for _ in range(N_max):
            
            update(grid)
            iterations += 1

            bgrid = grid.tobytes()
            if bgrid in history:
                return iterations
            
            history.add(bgrid)

        return iterations
    
    def run_visual(self, N_max: int) -> int:
        """
        Evolve a game of life simulation until equilibrium is reached.

        To determine when the internal grid object is at equilibrium a hashtable is created of every board update.
        The updated board is checked against all previous boards and the function is exited if a duplicate occurs.
        This catches both steady states and oscillating states. The hashtable will have legnth N_Max.

        A live matplotlib.pyplot.imshow plot is provided as the system updates.

        Args:
            N_max (int): Maximum number of updates to perform on the gol board.

        Returns:
            number of iterations to equilibriate (int)       
        """
        # setup
        grid = self.grid
        update = self.update
        vf = self.visual_freq

        # interactive plot
        plt.ion()
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(grid, cmap = 'gray_r', vmin = 0, vmax = 1, origin = 'lower')
        ax.set_title(r"Game Board Visualisation", fontweight = "bold")
        cbar = fig.colorbar(im, ax = ax, ticks = [0, 1])
        plt.pause(vf)

        # loop
        history = {grid.tobytes()} # hash table of previous boards
        iterations = 0      
        for _ in range(N_max):
            
            update(grid)
            iterations += 1

            im.set_data(grid)
            plt.pause(vf)

            bgrid = grid.tobytes()
            if bgrid in history:
                plt.ioff()
                return iterations
            
            history.add(bgrid)
        
        plt.ioff()
        return iterations

class simulation:

    def __init__(self, save_dir = devnull) -> None:
        
        # directory for saving files
        self.save_dir = save_dir
        if devnull is not save_dir:
            # suppress type warnings because save_dir is a pathlib.Path object if not devnull
            self.data_dir = save_dir / "data" #type: ignore
            self.data_dir.mkdir(exist_ok=True)
            self.fig_dir = save_dir / "figures" #type: ignore
            self.fig_dir.mkdir(exist_ok=True)

    def save_nparray(self, data: list|tuple|np.ndarray, data_names: list|tuple|np.ndarray) -> None:
        """
        Save numpy arrays in the ./runs/{timestamp}/data directory with names in data_names.

        Can only be called if self.data_dir is not devnull; otherwise an error will be thrown.

        Files saved in the numpy binary format .npy for unpacking in python.

        Args:
            data (tuple(numpy.ndarray)): tuple containing numpy arrays with data to be saved
            data_names (tuple(numpy.ndarray)): file names for the above data
        """

        file_extension = "npy"

        # assumes self.data_dir is not devnull
        for i, d in enumerate(data):
            path = self.data_dir / f"{data_names[i]}.{file_extension}" 
            np.save(path, d)

    def equilib_histogram(self, N: int, N_max: int, length: int) -> None:
        """
        Evolve many game of life simulations and produce a histogram of number of iterations to reach
        equilibrium. A GOL simulation is considered to be at equilibrium if:

        1.) A steady state or oscillating state is reached
        2.) the maximum number of iterations (N_max) is reached

        Uses a python for loop implementation to perform the simulations. This task can take advantage
        of parallel computing however the dimensions in this checkpoint (50 grid length) do not lead to 
        large computation times.

        Args:
            N (int): number of simulations to perform
            N_max (int): Maximum number of updates to perform on the gol board.
            length (int): length of the square game of life grid
        """
        iterations = np.zeros(shape=(N,))
        for i in range(N):
            gol = game_of_life(length, 'random')
            iterations[i] = gol.run_method(N_max)
         
        # save data
        if devnull != self.save_dir:
            data = (iterations,)
            data_names = ("iterations",)
            self.save_nparray(data, data_names)

        # plot
        fig, ax = plt.subplots(1, 1)
        ax.hist(iterations, color="tab:red", edgecolor = "black", bins=50)
        ax.set_title("Equilibriation Time (Iterations)", fontweight = "bold")
        ax.set_xlabel("No. Iterations")
        ax.set_ylabel("Frequency")
        fig_label = "equilibrium_histogram"

        # save figure
        if devnull != self.save_dir:
            self.fig_dir.mkdir(exist_ok=True)
            extension = "png"
            file_path = self.fig_dir /  f"{fig_label}-figure.{extension}"
            plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        plt.show()

    def glider_com(self, grid: np.ndarray) -> np.ndarray:
        """
        Input must be game of life board with only a glider on it. Returns the glider's centre of mass by 
        comuting the mean x and y coordinates of alive cells.

        Args:   
            grid (numpy.ndarray): grid of alive/dead sites with glider on it

        Returns:
            coordinates of the centre of mass (numpy.ndarray)
        """

        alive_x, alive_y = np.where(grid == 1)
        
        return np.array([np.mean(alive_x), np.mean(alive_y)])
    
    def glider_pattern(self, grid: np.ndarray) -> np.ndarray:
        """
        Input must be game of life board with only a glider on it. Returns a reduced grid which contains only the
        glider. The glider pattern will be padded with 1 dead cell on all axes.

        Args:
            grid (numpy.ndarray): grid of alive/dead sites with glider on it

        Returns:
            reduced grid with only glider (numpy.ndarray)
        """

        active = grid==1
        x, y = np.indices(grid.shape)
        min_x, min_y = np.min(x[active]), np.min(y[active])
        max_x, max_y = np.max(x[active]), np.max(y[active])
        rgrid = np.zeros(shape=(max_x - min_x + 3, max_y - min_y + 3)) # +2 for padding; +1 because indices start at 0 
        rgrid[1:-1, 1:-1] = grid[min_x:max_x+1, min_y:max_y+1] # +1 because slice is not inclusive of end

        return rgrid

    def glider_velocity(self, N_max: int, length: int) -> np.floating|float:
        """
        Compute velocity of a given glider by storing its active sites as a hashmap and updating the board. Once the glider completes one period
        the velocity is computed by dividing the displacement of the centre of mass of the glider by the period (in iterations).

        Change the preset for the gol object or modify the gol's internal grid to use different gliders.

        The velocity is printed to console and written to disk at ./data/velocity.npy
        The glider itself is also saved at ./data/glider.npy

        Args:
            N_max (int): number of iterations to attempt; should be larger than the gliders period
            length (int): length of the square game of life grid

        Returns:
            0 (float) if N_max is less than the gliders period
            velocity (float) otherwise
        """
        gol = game_of_life(length, 'glider')
        grid = gol.grid
        update = gol.update
        com = self.glider_com
        gp = self.glider_pattern

        # interactive plot
        plt.ion()
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(grid, cmap = 'gray_r', vmin = 0, vmax = 1, origin = 'lower')
        ax.set_title(r"Glider Position", fontweight = "bold")
        cbar = fig.colorbar(im, ax = ax, ticks = [0, 1])
        plt.pause(1)

        pos = [com(grid)]
        glider = gp(grid)
        history = {gp(grid).tobytes()} # hashes of grids
        v = 0.
        for _ in range(N_max):

            update(grid)

            im.set_data(grid)
            plt.pause(0.5)

            boundary = np.array(
                    [grid[0, :].any() == 1,
                    grid[-1, :].any() == 1,
                    grid[:, 0].any() == 1,
                    grid[:, -1].any() == 1]).any()

            if boundary: continue

            pos.append(com(grid))
                
            pattern = gp(grid).tobytes()

            # measure velocity when the glider returns
            if pattern in history:
                v = np.linalg.norm(pos[0] - pos[-1])/(len(pos)-1)
                plt.pause(1)
                break

            history.add(pattern)

        # save data
        if devnull != self.save_dir:
            data = (glider, np.atleast_1d(v))
            data_names = ("glider", "velocity")
            self.save_nparray(data, data_names)

        print(f"Velocity = {v:.2f} squares/iteration")
        return v
