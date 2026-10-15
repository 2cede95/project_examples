import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sys import exit
from os import devnull
from pathlib import Path
from multiprocessing import Pool, cpu_count

class SIRS:
    """
    SIRS object. Contains an in internal 2D numpy array of susceptible, infected, recovered and immune values. System evolves via the 
    internal update method. To be used with the Simulation class for the checkpoint tasks.
    """
    def __init__(self, length: int, p_suscept: float, p_infect: float, p_recover: float, immune_fraction=0., p_init = (1/3, 1/3, 1/3)) -> None:
        
        # probailities to intiialise grid
        p_s, p_i, p_r = p_init 

        if 1 != p_s + p_i + p_r:
            print("Probabilities R -> S, S -> I and I -> R must sum to 1. Exiting...")
            exit()
            return None

        # probabilites to change into states i.e. p_suscept is the probability of going from recovered -> susceptible
        self.p_suscept = p_suscept
        self.p_infect = p_infect
        self.p_recover = p_recover
        self.immune_fraction = immune_fraction 
        self.length = length

        self.grid = self.init_sarray(length, p_s, p_i, p_r, immune_fraction)

        # track how many of each state exist
        self.susceptible = np.sum(self.grid == 0)
        self.infected = np.sum(self.grid == 1)
        self.recovered = np.sum(self.grid == 2)
        self.immune = np.sum(self.grid == 3)

    def init_sarray(self, length: int, p_suscept: float, p_infect: float, p_recover: float, immune_fraction=0.) -> np.ndarray:
        """
        Returns a grid of susceptible susceptible, infected, recovered and immune sites distributed randomly using numpy.random.choice
        and the given probabilities.

        susceptible = 0
        infected = 1
        recovered = 2
        immune = 3

        Args:
            length (int): length of the square grid
            p_suscept (float): probability for a site to go from recovered to susceptibile
            p_infect (float): probability for a site to go from susceptibile to infected
            p_recover (float): probability for a site to go from infected to recovered
            immune_fraction (float): fraction of sites immune to infection (permanently)

        Returns:
            SIRS grid (numpy.ndarray)
        """

        # randomly distribute the susceptible, infected and recovered sites
        grid = np.random.choice((0, 1, 2), size=(length, length), p=(p_suscept, p_infect, p_recover))

        # randomly distribute immune sites on top of grid
        if 0 != immune_fraction:
            number_immune = int(np.square(length)*immune_fraction)
            immune_sites = np.random.choice(np.arange(length), size=(number_immune, 2))
            grid[immune_sites[:, 0], immune_sites[:, 1]] = 3
        
        return grid
    
    def update(self, grid: np.ndarray) -> None:
        """
        Update a SIRS grid by randomly selecting a coordinate and changing it with a given probability.

        The rules are:
            1.) S -> I with probability p_infect if at least one neighbour of the site to update is infected; otherwise, the site is unchanged.
            2.) I -> R with probability p_recover (regardless of the states of the neighbours).
            3.) R -> S with probability p_susceptible.

        susceptible = 0
        infected = 1
        recovered = 2
        immune = 3

        The SIRS grid is updated in place. Probabilites used are the internal values.

        Args:
            grid (numpy.ndarray): SIRS grid to be modified in place
        """

        length = grid.shape[0]
        i, j = np.random.randint(0, length, size=(2,))
        p = np.random.uniform(0, 1)

        # immune
        if 3 == grid[i, j]:
            return None
        
        # susceptible -> infected
        if 0 == grid[i, j]:
            # periodic boundary neighbour check
            infected_neighbour = np.array([
                grid[(i-1)%length, j] == 1,
                grid[(i+1)%length, j] == 1,
                grid[i, (j-1)%length] == 1,
                grid[i, (j+1)%length] == 1
            ])
            if True == infected_neighbour.any() and p < self.p_infect:
                grid[i, j] = 1
                self.susceptible -= 1
                self.infected += 1
            
            return None
            
        # infected -> recovered
        if 1 == grid[i, j]:
            # nested to return None when 1 == grid[i, j] but p > self.p_recover (performance optimisation)
            if p < self.p_recover:
                grid[i, j] = 2
                self.infected -= 1
                self.recovered += 1

            return None
        
        # recovered -> susceptible
        if 2 == grid[i,j]:
            # nested for optimisation as above
            if p < self.p_suscept:
                grid[i, j] = 0
                self.recovered -= 1
                self.susceptible += 1

            return None
        
        return None
    
    def run_visual(self, N: int) -> None:
        """
        Perform N*sweep self.update() calls on the internals SIRS grid using SIRS update rules.

        One sweep is the length of the internal grid variable squared.

        An animated matplotlib.pyplot.imshow plot is displayed as the system evolves.

        Args:
            N (int): number of sweeps to perform
        """

        grid = self.grid
        p_s = self.p_suscept
        p_i = self.p_infect
        p_r = self.p_recover
        update = self.update

        # interactive plot
        plt.ion()
        fig, ax = plt.subplots(1, 1)
        colours = ['navy', 'red', 'cyan', 'tab:green']
        cmap = mcolors.ListedColormap(colours)
        legend_handles = [
            mpatches.Patch(color=colours[0], label='Susceptible'),
            mpatches.Patch(color=colours[1], label='Infected'),
            mpatches.Patch(color=colours[2], label='Recovered'),
            mpatches.Patch(color=colours[3], label='Immune')
        ]
        ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.375, 1))
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin = 'lower')
        ax.set_title(
            r"SIRS Model $\mathbf{P_{R\rightarrow S} = }$"
            + f"{p_s:.2f} " + r", $\mathbf{P_{S\rightarrow I} =}$ "
            + f"{p_i:.2f}" + r", $\mathbf{P_{I\rightarrow R} = }$"
            + f"{p_r:.2f}", fontweight = "bold")
        plt.tight_layout()
        plt.pause(0.1)

        sweep = np.square(grid.shape[0])
    
        # compute loop
        for i in range(N):
            
            for _ in range(sweep):
                update(grid)
            
            if 0 == i%10:
                im.set_data(grid)
                plt.pause(0.1)

        plt.ioff()
    
    def run(self, N: int) -> None:
           """
            Perform N*sweep self.update() calls on the internals SIRS grid using SIRS update rules.

           One sweep is the length of the internal grid variable squared.

           An animated matplotlib.pyplot.imshow plot is displayed as the system evolves.

           Args:
               N (int): number of sweeps 
           """

           grid = self.grid
           update = self.update

           sweep = np.square(grid.shape[0])
           for _ in range(N):
               for _ in range(sweep):
                   update(grid)


class simulation:
    """
    Preset simulations for the checkpoint tasks.
    """
    def __init__(self, correlation_gap: int, save_dir = devnull) -> None:

        self.correlation_gap = correlation_gap

        self.save_dir = save_dir

        if devnull is not save_dir:
            # suppress type warnings because save_dir is a pathlib.Path object if not devnull
            self.data_dir = save_dir / "data" #type: ignore
            self.data_dir.mkdir(exist_ok=True)
            self.fig_dir = save_dir / "figures" #type: ignore
            self.fig_dir.mkdir(exist_ok=True)

    def absorbing(self, N: int, length: int) -> None:
        """
        SIRS object internal SIRS.run_visual() function call with pre-defined probability values to
        display an absorbing state.

        Args:
            N (int): number of sweeps to run the simulation for
            length (int): length of the square SIRS grid
        """
        p_s = 0.3
        p_i = 0.3
        p_r = 0.5

        sirs = SIRS(length, p_s, p_i, p_r, 0)
        sirs.run_visual(N)

    def dyanmic_equilibrium(self, N: int, length: int) -> None:
        """
        SIRS object internal SIRS.run_visual() function call with pre-defined probability values to
        display a dynamic equilibrium state.

        Args:
            N (int): number of sweeps
            length (int): length of the square SIRS grid
        """
        p_s = 0.5
        p_i = 0.5
        p_r = 0.5

        sirs = SIRS(length, p_s, p_i, p_r, 0)
        sirs.run_visual(N)

    def cyclic_wave(self, N: int, length: int) -> None:
        """
        SIRS object internal SIRS.run_visual() function call with pre-defined probability values to
        display a cyclic wave state.

        Args:
            N (int): number of sweeps 
            length (int): length of the square SIRS grid
        """
        p_s = 0.01
        p_i = 0.8
        p_r = 0.1

        sirs = SIRS(length, p_s, p_i, p_r, 0)
        sirs.run_visual(N)
    
    def heat_plot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, xlabel: str, ylabel: str, zlabel: str, title: str, fig_label: str) -> None:
        """
        Wrapper function for matplotlib.pyplot.pcolormesh

        Figure saved to ./runs/{timestamp}/{fig_label}.png
        
        Args:
            x (numpy.ndarray): x grid values 
            y (numpy.ndarray): y grid values 
            z (nump.ndarray): z values to be mapped onto the x, y grid
            xlabel (str): label for the x axis
            ylabel (str): label for the y axis
            zlabel (str): labelf for the z axis
            title (str): title for the plot
        """
 
        fig, ax = plt.subplots(1, 1)
        im = ax.pcolormesh(x, y, z)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        cbar = fig.colorbar(im, ax = ax, label=zlabel)   
    
        # save figure
        if devnull != self.save_dir:
            fig_dir = self.fig_dir
            fig_dir.mkdir(exist_ok=True)
            extension = "png"
            file_path = fig_dir /  f"{fig_label}-figure.{extension}"
            plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        plt.show()
    
    def error_plot(self, x: np.ndarray, y: np.ndarray, xerr: np.ndarray, yerr: np.ndarray, xlabel: str, ylabel: str, title: str, fig_label: str) -> None:
        """
        Wrapper function for matplotlib.pyplot.errorbar

        Figure saved to ./runs/{timestamp}/{fig_label}.png

        Args:
            x (array): x grid values 
            y (array): y grid values 
            xerr (array): errorbar values for the x axis
            yerr (array): errorbar values for the y axis
            xlabel (str): label for the x axis
            ylabel (str): label for the y axis
            title (str): title for the plot
            fig_label (str): label to give the plot a unique name in the internal path log_dir.
        """
        #plot 
        fig, ax = plt.subplots(1, 1)
        plt.errorbar(x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    ecolor = "black",
                    color = "tab:red",
                    marker = "s",
                    markersize = 2,
                    markerfacecolor = "tab:red",
                    markeredgecolor = "black",
                    label = "Error Bars",
                    capsize = 2,
                    zorder = 3
                    )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # save figure
        if devnull != self.save_dir:
            self.fig_dir.mkdir(exist_ok=True)
            extension = "png"
            file_path = self.fig_dir /  f"{fig_label}-figure.{extension}"
            plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()
    
    def plot(self, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, fig_label: str) -> None:
        """
        Wrapper function for matplotlib.pyplot.plot

        Figure saved to ./runs/{timestamp}/{fig_label}.png

        Args:
            x (array): x grid values 
            y (array): y grid values 
            xlabel (str): label for the x axis
            ylabel (str): label for the y axis
            title (str): title for the plot
            fig_label (str): label to give the plot a unique name in the internal path log_dir.
        """
        #plot 
        fig, ax = plt.subplots(1, 1)
        plt.plot(x, y, c="tab:red")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # save figure
        if devnull != self.save_dir:
            self.fig_dir.mkdir(exist_ok=True)
            extension = "png"
            file_path = self.fig_dir /  f"{fig_label}-figure.{extension}"
            plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()
    
    def bootstrap(self, vals: np.ndarray, func, k = 1000, args = (None, )) -> np.floating:
        """
        Compute error on a given function func(data) by bootstrapping k samples of the data and computing the standard deviation of
        k func(bootstrapped data) evaluations.

        Bootstrap resampling implementation is numpy.random.choice with replacement.

        Args:
            vals (numpy.ndarray): input data for function func
            func (function):  function with signature func(data, arg1, arg2, arg3, ...)
            k (int): number of bootstrap resamples
            args (tuple): tuple of arguments to be unpackaged into func like func(data, *args)

        Returns:
            error on the function evaluation with the given data (float)
        """

        # bootstrap samples
        bs = np.random.choice(np.array(vals), size = (k, vals.size), replace=True) # if vals is a list numpy.random.choice will break

        # evaluate
        q = [func(bs[i], *args) for i in range(k)]
        
        return np.std(q)
    
    def save_nparray(self, data: np.ndarray|list|tuple, data_names: np.ndarray|list|tuple) -> None:
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

    def SI_RS_phase_diagram(self, p_res: int, N_warmup: int, N: int, length: int) -> None:
        """
        Perform p_res^2 SIRS simulations (N + N_warmup sweeps each) and plot the average fraction of infected sites
        on the P(S->I) - P(R->S) probability grid for a fixed P(I->R) = 0.5.

        Measurements of infected sites are taken with a gap to avoid correlation between states.

        Plots are saved and displayed using the internal wrapper function self.heat_plot

        Args:
            p_res (int)): number of probability values to sample per row in the probability grid i.e. p_res = 5 is a 5**2 grid of evenly spaced probilities
                          ranging from 0. -> 1.0
            N_warmup (int): number of warmup sweeps to perform before measurements begin (per SIRS simulation)
            N (int): number of sweeps to perform (per SIRS simulation)
            length (int): length of the grid for SIRS simulation
        """

        p_s = np.linspace(0, 1, p_res)
        p_i = p_s.copy()
        p_r = 0.5
        frac_immune = 0
        sweep = np.square(length)
        correlation_gap = self.correlation_gap 
        measurements = N//correlation_gap

        # compute loop (multithreaded)
        args = [(s, i, p_r, N_warmup, N, length, correlation_gap, sweep, measurements, frac_immune) for s in p_s for i in p_i]
        with Pool(cpu_count()-1) as p: # leave one cpu core available for OS
            infected = np.array(p.starmap(run, args))

        # observables 
        # starmap -> (p_res*p_res, measurements); need to unflatten the i,j grid after performing mean/var on axis 1 (infected sites axis)
        mean_infectedf = np.mean(infected/sweep, axis=1).reshape(p_res, p_res) # infected fraction so divide by number of sites
        var_infected = np.var(infected, axis=1).reshape(p_res, p_res) 
        
        # save data
        if devnull != self.save_dir:
            data = (mean_infectedf, var_infected, p_s, p_i, np.atleast_1d(p_r))
            data_names = ("mean_infected_fraction", "var_infected", "prob_susceptible", "prob_infected", "prob_recovered")
            self.save_nparray(data, data_names)

        #plots
        i, j = np.meshgrid(p_i, p_s)
        title = r"Average Fraction of Infected Sites $\mathbf{\langle I \rangle}$/N"
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Infected to Susceptible, $P_{R \rightarrow S}$"
        zlabel = r"$\langle I \rangle$/N"
        fig_label = "mean_infected"
        self.heat_plot(i, j, mean_infectedf, xlabel, ylabel, zlabel, title, fig_label)

        #plots
        i, j = np.meshgrid(p_i, p_s)
        title = r"Variance of Infected Sites $\mathbf{\langle I^{2} \rangle + \langle I \rangle^{2}}$"
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Infected to Susceptible, $P_{R \rightarrow S}$"
        zlabel = r"$\langle I^{2} \rangle + \langle I \rangle^{2}$"
        fig_label = "var_infected"
        self.heat_plot(i, j, var_infected, xlabel, ylabel, zlabel, title, fig_label)

    def SI_variance(self, p_res: int, N_warmup: int, N: int, length: int) -> None:
        """
        Perform p_res SIRS simulations (N + N_warmup sweeps each) and plot the variance of infected sites
        on a slice of the P(S->I)xP(R->S)xP(I->R) probability space with constant P(I->S) = P(I->R) = 0.5.

        Measurements of infected sites are taken with a gap to avoid correlation between states.

        Plot is saved and displayed using the internal wrapper function self.error_plot

        Args:
            p_res (int)): number of probability values to sample per row in the probability grid i.e. p_res = 5 is a 5**2 grid of evenly spaced probilities
                          ranging from 0. -> 1.0
            N_warmup (int): number of warmup sweeps to perform before measurements begin (per SIRS simulation)
            N (int): number of sweeps to perform (per SIRS simulation)
            length (int): length of the grid for SIRS simulation
        """      
        p_s = 0.5
        p_i = np.linspace(0.2, 0.5, p_res)
        p_r = 0.5
        frac_immune = 0
        sweep = np.square(length)
        correlation_gap = self.correlation_gap
        measurements = N//correlation_gap

        infected = np.zeros(shape=(p_res, measurements))

        # compute loop (multithreaded)
        args = [(p_s, i, p_r, N_warmup, N, length, correlation_gap, sweep, measurements, frac_immune) for i in p_i]
        with Pool(cpu_count()-1) as p: # leave one cpu core available for OS
            infected = np.array(p.starmap(run, args))

        # observables 
        # starmap -> (p_res, measurements) so axis 1 is measurements
        var_infected = np.var(infected, axis=1)

        # errors
        var_error = [self.bootstrap(infected[i, :], np.var) for i in range(p_res)]

        # save data
        if devnull != self.save_dir:
            data = (var_infected, var_error, np.atleast_1d(p_s), p_i, np.atleast_1d(p_r))
            data_names = ("var_infected", "var_error", "prob_susceptible", "prob_infected", "prob_recovered")
            self.save_nparray(data, data_names)

        # plot
        x = p_i
        y = var_infected
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Variance, $\langle I^{2} \rangle + \langle I \rangle^{2} / N$"
        title = r"Slice with $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r"and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "variance_cut"
        self.error_plot(x, y, None, var_error, xlabel, ylabel, title, fig_label) #type: ignore
    
    def immunity(self, immunity_res: int, N_warmup: int, N: int, length: int) -> None:
        """
        Perform immunity_res SIRS simulations (N + N_warmup sweeps each) and plot the average fraction of infected sites
        against the immunity fraction.

        Measurements of infected sites are taken with a gap to avoid correlation between states.

        Plot is saved and displayed using the internal wrapper function self.plot

        Args:
            immunity_res (int)): number of immunity fraction values to sample. Range is from 0. -> 1.0
            N_warmup (int): number of warmup sweeps to perform before measurements begin per (SIRS simulation)
            N (int): number of sweeps to perform (per SIRS simulation)
            length (int): length of the grid for SIRS simulation
        """
        p_s = p_i = p_r = 0.5
        sweep = np.square(length)
        correlation_gap = self.correlation_gap
        measurements = N//correlation_gap

        frac_immune = np.linspace(0, 1, immunity_res)

        # compute loop (multithreaded)
        args = [(p_s, p_i, p_r, N_warmup, N, length, correlation_gap, sweep, measurements, f) for f in frac_immune]
        with Pool(cpu_count()-1) as p: # leave one cpu core available for OS
            infected = np.array(p.starmap(run, args))

        # observables 
        # starmap -> (p_res, measurements) so axis 1 is measurements
        mean_infectedf = np.mean(infected/sweep, axis=1)

        # save data
        if devnull != self.save_dir:
            data = (mean_infectedf, frac_immune, p_s, p_i, np.atleast_1d(p_r))
            data_names = ("mean_infected_fraction", "immunity_fraction", "prob_susceptible", "prob_infected", "prob_recovered")
            self.save_nparray(data, data_names)

        # plot
        x = frac_immune
        y = mean_infectedf
        xlabel = r"Immunity Fraction, $f_{Immune}$"
        ylabel = r"Mean Infected Fraction, $\langle I \rangle$/N"
        title = r"Varying Immunity with $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r", $\mathbf{P_{S\rightarrow I} = }$" + f"{p_i:.2f}" + r" and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "immune_frac"
        self.plot(x, y, xlabel, ylabel, title, fig_label)

def run(p_s: float, p_i: float, p_r: float, N_warmup: int, N: int, length: int, correlation_gap: int, sweep: int, measurements: int, frac_immune: float) -> np.ndarray:
    """
    Top-level function for looping over a given number of sweeps and returning an array of infected site measurements.

    For use with multiprocessing.starmap. If the function is not top-level an AttributeError is thrown.

    Args:
        p_s (float): probability of a site transitioning from recovered to susceptible
        p_s (float): probability of a site transitioning from susceptible to infected
        p_s (float): probability of a site transitioning from infected to recovered
        N_warmup (int): number of warmup sweeps to perform before measurements begin (per SIRS simulation)
        N (int): number of sweeps to perform (per SIRS simulation)
        length (int): length of the grid for SIRS simulation
        sweep (int): length of one sweep 
        measurements (int): number of measurements to conduct 
        frac_immune (float): fraction of sites which are immune to infection (permanently)
    """
    sirs = SIRS(length, p_s, p_i, p_r, frac_immune)
    grid = sirs.grid
    update = sirs.update
    infected = np.zeros(shape=(measurements,))
    
    # warmup sweeps
    for _ in range(N_warmup):
        for _ in range(sweep):
            update(grid)
    
    # measurement sweeps
    for k in range(N):
        for _ in range(sweep):
            update(grid)
        if 0 == k%correlation_gap:
            infected[k//correlation_gap] = sirs.infected
    
    return infected