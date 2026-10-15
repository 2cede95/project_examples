import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import time
import sys
from os import devnull
from pathlib import Path
from multiprocessing import cpu_count

class lattice():
    """
    Class which stores lattice variables and update methods for a SIRS simulation.

    Attributes:
        self.dimension (int): length of a side of the square lattice
        self.unit_sweep (int): number of sites in the lattice
        self.p_init (float, float, float): tuple of probailities for the initialisation of the board
        self.p_suscept (float): probability to go from recovered to susceptible
        self.p_infected (float): probabilitiy to go from susceptible to infected
        self.p_recovered (float): probability to go from infected to recovered
        self.immune_fraction (float): fraction of board sites which are immune to infection. Selected randomly after the initialisation.
        self.data (numpy ndarray): contains all site values
        self.sweeps (int): internal counter for sweeps elapsed
        self.ax, self.line, self.fig: internel plot variables from matplotlib.subplots
        self.sweep_method (func): method used to sweep (visual or not)
        self.plot_freq: frequency to update plot (in sweeps)
    """
    def __init__(self,dimension,visual=False,p_init=(1/3, 1/3, 1/3),p=(0.4, 0.4, 0.2),plot_freq=5,immune_fraction=0):
        """
        Init method for lattice class.

        Args:
            dimension (int): length of a side of the square lattice
            visual (bool): show plot of lattice
            p_init (float, float, float): tuple of probailities for the initialisation of the board
            p (float, float, float): tuple of probailities for the update rules
            plot_freq (int): frequency to update plot (in sweeps)
            immune_fraction (float): fraction of board sites which are immune to infection. Selected randomly after the initialisation.
        """

        self.dimension = int(dimension)
        self.unit_sweep = int(dimension**2)
        self.p_init = p_init
        self.p_suscept = p[0]
        self.p_infected = p[1]
        self.p_recovered = p[2]
        self.immune_fraction = immune_fraction
        self.data = self.lattice_init()
        self.sweeps = 0

        # visual
        if visual:
            plt.ion()
            self.fig, self.ax = plt.subplots(1, 1)

            colours = ['navy', 'red', 'cyan', 'tab:green']
            cmap = mcolors.ListedColormap(colours)
            legend_handles = [
                mpatches.Patch(color=colours[0], label='Susceptible'),
                mpatches.Patch(color=colours[1], label='Infected'),
                mpatches.Patch(color=colours[2], label='Recovered'),
                mpatches.Patch(color=colours[3], label='Immune')
            ]
            self.ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.375, 1))
            self.line = self.ax.imshow(self.data, cmap=cmap, vmin=0, vmax=3, origin = 'lower')
            self.ax.set_title(r"SIRS Model $\mathbf{P_{R\rightarrow S} = }$" + f"{self.p_suscept:.2f} " + r", $\mathbf{P_{S\rightarrow I} =}$ " + f"{self.p_infected:.2f}" + r", $\mathbf{P_{I\rightarrow R} = }$" + f"{self.p_recovered:.2f}", fontweight = "bold")
            plt.tight_layout()

            self.plot_freq = plot_freq
            self.sweep_method = self.sweep_visual
        
        else:
            self.sweep_method = self.sweep

        return None
    
    def lattice_init(self):
        """
        Initalises a lattice with the probabilites inside the internal variable p_init.
        If internal variable self.immune_fraction is non-zero the immune sites are distributed randomly after the initial 
        distribution of site values.

        Returns:
            numpy.ndarray
        """
        #* 0 = susceptible
        # 1 = infected
        # 2 = recovered
        # 3 = immune
        lattice = np.random.choice(a=(0, 1, 2),
                                size=(self.dimension,
                                self.dimension), 
                                p=self.p_init
                                )
        
        if self.immune_fraction == 0:
            return lattice

        # update random sites to immune 
        no_immune = int(np.ceil((self.dimension**2)*self.immune_fraction))

        # do not replace to conserve the immune fraction
        flatten = lattice.flatten() 
        rand_ind = np.random.choice(flatten.size,size=no_immune,replace=False)
        flatten[rand_ind] = 3

        lattice = flatten.reshape(self.dimension, self.dimension)

        return lattice

        
    

    def infected_neighbours(self, i, j):
            """
            Count the number of infected neighbours for a given site.

            Args:
                i (int): column of the given site
                k (int): row of the given site

            Returns:
                tuple(bool)
            """
            # 1 if infected neighbour else 0
            return np.array([self.data[(i+1) % self.dimension, j] == 1,
             self.data[(i-1) % self.dimension, j] == 1,
             self.data[i, (j+1) % self.dimension] == 1,
             self.data[i, (j-1) % self.dimension] == 1
            ])

    def update(self):
        """
        Update the internal self.data variable according to the internal SIRS probabilities self.p
        """
        # update sites, return bool if the site was changed

        # pick a random site
        i = np.random.randint(self.dimension)
        j = np.random.randint(self.dimension)
        rand_float = np.random.uniform(low=0, high=1)
        random_site = self.data[i, j]

        # determine state and update schochastically
        # site is susceptible and will become infected with prob p_infected
        # ONLY IF at least one nearest neighbour is infected
        if (random_site == 0):
            if (rand_float < self.p_infected):
                if self.infected_neighbours(i, j).any():
                    self.data[i, j] = 1
                    return True
        
        # site is infected and will recover with prob p_recovered
        elif (random_site == 1):
            if (rand_float < self.p_recovered):
                self.data[i, j] = 2
                return True

        # site is recovered and become susceptible with prob p_suscept
        elif (random_site == 2):
            if (rand_float < self.p_suscept):
                self.data[i, j] = 0
                return True

        # the site was not updated
        return False
    
    def sweep(self, N=1):
        """
        Perform a given number of sweeps. A sweep is defined as self.unit_sweep iterations of the SIRS update rules.

        Args:
            N (int): number of sweeps to perform.
        """
        # perform N sweeps
        for _ in range(0, N):

            for _ in range(0, self.unit_sweep):
                self.update()
            
            self.sweeps+=1

        return None
    
    def sweep_visual(self, N=1):
        """
        Perform a given number of sweeps and show a plot of the internal self.date variable.
          
        A sweep is defined as self.unit_sweep iterations of the SIRS update rules.
        
        The update frequency of the plot is determined by the internal self.plot_freq variable.

        Args:
            N (int): number of sweeps to perform
        """
        # perform N sweeps
        for _ in range(0, N):

            for _ in range(0, self.unit_sweep):
                self.update()

            self.sweeps+=1

            # update matplotlib internal plot
            if self.sweeps%self.plot_freq == 0:
                self.line.set_data(self.data)
                plt.pause(0.001)
        
        return None

    def get_infected(self):
        """
        Count number of infected sites in the internal self.data variable.

        Returns:
            int
        """
        # how many sites are infected
        return np.sum(self.data == 1)

class simulation():
    """
    Class containing various simulations to perform on a lattice object.
    
    Attributes:
        self.log_path (Pathlib.path): OS independent path to the log file
        self.log_dir (Pathlib.path): OS independent path to the log directory
        self.muilticore (bool): whether or not to use multicore processing
        self.dimension (int): length of the square side for the lattice.data variable
        self.visual (bool): whether or not to visualise sweep updates
        self.p_init (float, float, float): probabilities for the initial distribution of sites in the lattice object
        self.plot_freq (int): number of sweeps between plot updates
        self.phase_full (func): method for computing observables in a 2 dof phase diagram simulation (multicore or single)
        self.phase_slice (func): method for computing observables in a 1 dof phase diagram simulation (multicore or single)
        self.immunity (func): method for computing observables in a 1dof immunity fraction simulation
    """
    def __init__(self,dimension,visual,multicore,p_init=(1/3, 1/3, 1/3),plot_freq=5,log_path=devnull,log_dir=devnull):
        """
        Init method for the simulation class.

        Args:
            dimension (int): length of the square side for the lattice.data variable
            visual (bool): whether or not to visualise sweep updates
            multicore (bool): whether or not to use multicore processing
            p_init (float, float, float): probabilities for the initial distribution of sites in the lattice object
            plot_freq (int): number of sweeps between plot updates
            log_path (Pathlib.path): OS independent path to the log file
            log_dir (Pathlib.path): OS independent path to the log directory
        """
        self.log_path = log_path
        self.log_dir = log_dir
        self.muilticore = multicore

        # simulation params
        self.dimension = dimension
        self.visual = visual
        self.p_init = p_init
        self.plot_freq = plot_freq

        # handle single or mutlicore case
        if multicore:
            self.phase_full = self.phase_full_multi
            self.phase_slice = self.phase_slice_multi
            self.immunity = self.immunity_multi
        else:
            self.phase_full = self.phase_full_single
            self.phase_slice = self.phase_slice_single
            self.immunity = self.immunity_single

        return None

    def preset_visualisation(self, state, N_sweeps):
        """
        Animated plot of a SIRS system evolution given a set of predefined outcomes for the checkpoint.
        
        Args:
            state (str): which outcome to visualise
            N_sweeps (int): how many sweeps to perform for the given state
        """
        states = ('absorbing', 'dynamic_equilibrium', 'oscillating')
        state = str(state).lower()  

        if state not in states:
            print(f"Bad preset. Must be one of {states}")   
            sys.exit()

        match state:

            case 'absorbing':

                p_init = (1/3, 1/3, 1/3)
                p = (0.3, 0.3, 0.5)
                dimension = 50

            case 'dynamic_equilibrium':
 
                p_init = (1/3, 1/3, 1/3)
                p = (0.5, 0.5, 0.5)
                dimension = 50

            case 'oscillating':
 
                p_init = (1/3, 1/3, 1/3)
                p = (0.075, 0.8, 0.5)
                dimension = 100
        
        sirs = lattice(dimension=dimension,visual=True,p_init=p_init,p=p)
        sirs.sweep_method(N_sweeps)

        return None

    def phase_full_multi(self, res, N, N_warmup):
        """
        Iterate through probabilities recovered -> susceptible and susceptible -> infected and measure mean fraction of 
        infected sites and variance of infected sites for each pair of values. Conclude with a contour plot of the 2D probability
        grid mapped with the mean fraction of infected sites and then a heatmap plot of the variance on the same 2D probability
        grid.

        Throughtout the simulation the probability to go from infected to recovered is fixed at 0.5.

        This implementation uses pathos.multiprocessing to leverage full capability of a system's CPU.

        Data and plots from the simulation are saved inside the log_dir internal path. 

        Args:
            res (float): seperation between probability points on the 2D probability grid
            N (int): number of sweeps to perform for each probability pair
            N_warmup (int): how many sweeps to perform BEFORE the N sweeps are performed. No measurements are taken during the warmup.
        """
        # import multiprocessing library
        from pathos.multiprocessing import ProcessingPool as Pool

        # open log file
        log = open(self.log_path, "a")

        # loop over a 2D array of p_suscept and p_infected pairs at constant p_recovered
        # recover No. infected sites for every pair 
        # compute mean and variance and normalise both by number of sites
        p_s = np.arange(0, res+1, res)
        p_i = np.arange(0, res+1, res)
        p_r = 0.5
        n = len(p_s)
        meas_freq = 1

        mean_i = np.empty(shape=(n, n))
        var_i = np.empty(shape=(n, n))
        start_time = time.time()
        args_list = [(int(N), int(N_warmup), p_sus, p_inf, float(p_r), int(meas_freq)) for p_sus in p_s for p_inf in p_i]

        # main loop
        if self.visual:
            # very hard to see anything with a large amount of plots
            pool = Pool(nodes=cpu_count()//2)
        else:
            pool = Pool(nodes=cpu_count())

        results = np.array(pool.map(lambda args_list: self.i_sites(*args_list), args_list)) 
        mean_i = np.mean(results, axis=1).reshape(n, n)
        var_i = np.var(results, axis=1).reshape(n, n)

        # time elapsed
        progress = 1
        elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
        print(f"{progress*100:.2f}% complete in " + elapsed_time)
        print(f"{progress*100:.2f}% complete in " + elapsed_time, file=log)

        # normlise wrt no. of sites
        mean_infected_fraction = mean_i/self.dimension**2
        self.data = (mean_i, var_i, p_s, p_i, np.atleast_1d(p_r), mean_infected_fraction)
        self.data_names = ("mean_infected_sites", "var_infected_sites", "prob_suscept", "prob_infect", "prob_recover", "mean_infected_fraction")
        self.save_data()

        # close log
        log.close()

        #plot mean infected
        x, y = np.meshgrid(p_i, p_s)
        title = r"Average Fraction of Infected Sites $\mathbf{\langle I \rangle}$/N"
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Infected to Susceptible, $P_{R \rightarrow S}$"
        zlabel = r"$\langle I \rangle$/N"
        fig_label = "mean_infected"
        self.contour_plot(x, y, mean_infected_fraction, xlabel, ylabel, zlabel, title, fig_label)

        # plot variance
        title = r"Variance of Infected Sites $\mathbf{\langle I^{2} \rangle + \langle I \rangle^{2}}$"
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Infected to Susceptible, $P_{R \rightarrow S}$"
        zlabel = r"$\langle I^{2} \rangle + \langle I \rangle^{2}$"
        fig_label = "variance"
        self.heat_plot(x, y, var_i, xlabel, ylabel, zlabel, title, fig_label)

        return None
    
    def phase_full_single(self, res, N, N_warmup):
        """
        Iterate through probabilities recovered -> susceptible and susceptible -> infected and measure mean fraction of 
        infected sites and variance of infected sites for each pair of values. Conclude with a contour plot of the 2D probability
        grid mapped with the mean fraction of infected sites and then a heatmap plot of the variance on the same 2D probability
        grid.

        Throughtout the simulation the probability to go from infected to recovered is fixed at 0.5.

        Data and plots from the simulation are saved inside the log_dir internal path. 

        Args:
            res (float): seperation between probability points on the 2D probability grid
            N (int): number of sweeps to perform for each probability pair
            N_warmup (int): how many sweeps to perform BEFORE the N sweeps are performed. No measurements are taken during the warmup.
        """
        # open log file
        log = open(self.log_path, "a")

        # loop over a 2D array of p_suscept and p_infected pairs at constant p_recovered
        # recover No. infected sites for every pair 
        # compute mean and variance and normalise both by number of sites
        p_s = np.arange(0, res+1, res)
        p_i = np.arange(0, res+1, res)
        p_r = 0.5
        n = len(p_s)
        meas_freq = 1

        mean_i = np.empty(shape=(n, n))
        var_i = np.empty(shape=(n, n))
        start_time = time.time()

        for i in range(0, n):
            for j in range(0, len(p_i)):
                infected_sites = self.i_sites(int(N), int(N_warmup), p_s[i], p_i[j], p_r, meas_freq=meas_freq)
                mean_i[i, j] = np.mean(infected_sites)
                var_i[i, j] = np.var(infected_sites, mean = mean_i[i, j])

            # print progress
            progress = (i+1)/n
            elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
            print(f"{progress*100:.2f}% complete in " + elapsed_time)
            print(f"{progress*100:.2f}% complete in " + elapsed_time, file=log)

        # normlise wrt no. of sites
        mean_infected_fraction = mean_i/self.dimension**2
        self.data = (mean_i, var_i, p_s, p_i, np.atleast_1d(p_r), mean_infected_fraction)
        self.data_names = ("mean_infected_sites", "var_infected_sites", "prob_suscept", "prob_infect", "prob_recover", "mean_infected_fraction")
        self.save_data()

        # close log
        log.close()

        #plot mean infected
        x, y = np.meshgrid(p_i, p_s)
        title = r"Average Fraction of Infected Sites $\mathbf{\langle I \rangle}$/N"
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Infected to Susceptible, $P_{R \rightarrow S}$"
        zlabel = r"$\langle I \rangle$/N"
        fig_label = "mean_infected"
        self.contour_plot(x, y, mean_infected_fraction, xlabel, ylabel, zlabel, title, fig_label)

        # plot variance
        title = r"Variance of Infected Sites $\mathbf{\langle I^{2} \rangle + \langle I \rangle^{2}}$"
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Infected to Susceptible, $P_{R \rightarrow S}$"
        zlabel = r"$\langle I^{2} \rangle + \langle I \rangle^{2}$"
        fig_label = "variance"
        self.heat_plot(x, y, var_i, xlabel, ylabel, zlabel, title, fig_label)

        return None
    
    def phase_slice_multi(self,min_p,max_p,res,N,N_warmup,meas_freq=int(1)):
        """
        Iterate through values from min_p -> max_p for the probability to go from susceptible -> infected and plot the mean
        fraction of infected sites as a function of probability. Also plots the variance of infected sites as a function of probability.

        Throughtout the simulation the probabilities to go from infected to recovered and from 
        recovered to susceptible are fixed at 0.5.
        
        This implementation uses pathos.multiprocessing to leverage full capability of a system's CPU.

        Data and plots from the simulation are saved inside the log_dir internal path. 

        Args:
            min_p (float), max_p (float): defines the range of probabilities to iterate through
            res (float): separation between probability points in the 1D probailitiy range
            N (int): number of sweeps to perform per probability value
            N_warmup (int): warmup sweeps to perform before the N sweeps are computed (per probability value)
            meas_freq (int): how often to take measurements (in sweeps)
        """
        # open log file
        log = open(self.log_path, "a")

        # import multiprocessing library
        from pathos.multiprocessing import ProcessingPool as Pool

        # variance along p_s = 0.5 for p_i = {0.2, ..., 0.5}
        p_r = 0.5
        p_s = 0.5
        p_i = np.arange(min_p, max_p+res, res)
        n = len(p_i)

        # handle visual case
        if self.visual:
            # very hard to see anything with a large amount of plots
            pool = Pool(nodes=cpu_count()//2)
        else:
            pool = Pool(nodes=cpu_count())

        # main compute loop
        start_time = time.time()
        args_list = [(int(N), int(N_warmup), float(p_s), p_inf, float(p_r), int(meas_freq)) for p_inf in p_i]
        results = np.array(pool.map(lambda args_list: self.i_sites(*args_list), args_list)) 
        var_i = np.var(results, axis=1)
        mean_i = np.mean(results, axis=1)
        mean_infected_fraction = self.mean_fraction(results, axis=1)

        # time elapsed
        progress = 1
        elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
        print(f"{progress*100:.2f}% complete in " + elapsed_time)
        print(f"{progress*100:.2f}% complete in " + elapsed_time, file=log)

        # compute error
        var_err = np.empty_like(var_i)
        for i in range(0, len(results)):
            var_err[i] = self.bootstrap(results[i], np.var, (0,), 1000)

        mean_err = np.empty_like(mean_i)
        for i in range(0, len(results)):
            mean_err[i] = self.bootstrap(results[i], self.mean_fraction, (0,), 1000)
    
        # close log
        log.close()

        # save data
        self.data = (mean_i, var_i, np.atleast_1d(p_s), p_i, np.atleast_1d(p_r), mean_infected_fraction, mean_err, var_err)
        self.data_names = ("mean_infected_sites", "var_infected_sites", "prob_suscept", "prob_infect", "prob_recover", "mean_infected_fraction", "mean_err", "var_err")
        self.save_data()

        # plot
        x = p_i
        y = var_i
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Variance, $\langle I^{2} \rangle + \langle I \rangle^{2}$"
        title = r"Variance of Infected Sites $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r"and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "variance_cut"
        self.error_plot(x, y, None, var_err, xlabel, ylabel, title, fig_label)

        x = p_i
        y = mean_infected_fraction
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Mean, $\langle I \rangle$"
        title = r"Mean Fraction of Infected Sites with $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r"and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "mean_cut"
        self.error_plot(x, y, None, mean_err, xlabel, ylabel, title, fig_label)

        return None
    
    def phase_slice_single(self,min_p,max_p,res,N,N_warmup,meas_freq=int(1)):
        """
        Iterate through values from min_p -> max_p for the probability to go from susceptible -> infected and plot the mean
        fraction of infected sites as a function of probability. Also plots the variance of infected sites as a function of probability.

        Throughtout the simulation the probabilities to go from infected to recovered and from 
        recovered to susceptible are fixed at 0.5.

        Data and plots from the simulation are saved inside the log_dir internal path. 

        Args:
            min_p (float), max_p (float): defines the range of probabilities to iterate through
            res (float): separation between probability points in the 1D probailitiy range
            N (int): number of sweeps to perform per probability value
            N_warmup (int): warmup sweeps to perform before the N sweeps are computed (per probability value)
            meas_freq (int): how often to take measurements (in sweeps)
        """
        # open log file
        log = open(self.log_path, "a")

        # variance along p_s = 0.5 for p_i = {0.2, ..., 0.5}
        p_r = 0.5
        p_s = 0.5
        p_i = np.arange(min_p, max_p+res, res)
        n = len(p_i)

        # main compute loop
        results = np.empty(shape=(n, N))
        start_time = time.time()
        for i in range(0, len(p_i)):
            results[i][:] = self.i_sites(N, N_warmup, p_s, p_i[i], p_r, meas_freq)

            # print progress every 10%
            if i%(n/10) == 0:
                progress = (i+1)/n
                elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
                print(f"{progress*100:.2f}% complete in " + elapsed_time)
                print(f"{progress*100:.2f}% complete in " + elapsed_time, file=log)

        var_i = np.var(results, axis=1)
        mean_i = np.mean(results, axis=1)
        mean_infected_fraction = self.mean_fraction(results, axis=1)

        # compute errors
        var_err = np.empty_like(var_i)
        for i in range(0, len(results)):
            var_err[i] = self.bootstrap(results[i], np.var, (0,), 1000)

        mean_err = np.empty_like(var_i)
        for i in range(0, len(results)):
            mean_err[i] = self.bootstrap(results[i], self.mean_fraction, (0,), 1000)
    
        # close log
        log.close()
    
        # save data
        self.data = (mean_i, var_i, np.atleast_1d(p_s), p_i, np.atleast_1d(p_r), mean_infected_fraction, mean_err, var_err)
        self.data_names = ("mean_infected_sites", "var_infected_sites", "prob_suscept", "prob_infect", "prob_recover", "mean_infected_fraction", "error_mean", "error_var")
        self.save_data()

        # plot
        x = p_i
        y = var_i
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Variance, $\langle I^{2} \rangle + \langle I \rangle^{2}$"
        title = r"Slice with $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r"and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "variance_cut"
        self.error_plot(x, y, None, var_err, xlabel, ylabel, title, fig_label)

        x = p_i
        y = mean_infected_fraction
        xlabel = r"Susceptible to Infected, $P_{S \rightarrow I}$"
        ylabel = r"Mean, $\langle I \rangle$"
        title = r"Mean Fraction of Infected Sites with $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r"and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "mean_cut"
        self.error_plot(x, y, None, mean_err, xlabel, ylabel, title, fig_label)

        return None
    
    def immunity_multi(self,N,N_warmup,meas_freq,res):
        """
        Iterate through immunity fraction values from 0. -> 0.1 with constant update probabilites of 0.5. Plots the 
        fraction of average fraction of infected sites as a function of the immunity fraction.

        This implementation uses pathos.multiprocessing to leverage full capability of a system's CPU.

        Data and plots from the simulation are saved inside the log_dir internal path. 
        
        Args:
            N (int): number of sweeps to perform per probability value
            N_warmup (int): warmup sweeps to perform before the N sweeps are computed (per probability value)
            meas_freq (int): how often to take measurements (in sweeps)  
            res (float): separation between probability points in the 1D probailitiy range        
        """
        # open log file
        log = open(self.log_path, "a")

        # import multiprocessing library
        from pathos.multiprocessing import ProcessingPool as Pool

        # probabilities fixed and immune fraction varied
        p_s = 0.5
        p_i = 0.5
        p_r = 0.5
        frac_immune = np.arange(start=0, stop=1, step=res)

        # handle visual case
        if self.visual:
            # very hard to see anything with a large amount of plots
            pool = Pool(nodes=cpu_count()//2)
        else:
            pool = Pool(nodes=cpu_count())

        # main compute loop
        args_list = [(int(N), int(N_warmup), float(p_s), float(p_i), float(p_r), int(meas_freq), float(f_i)) for f_i in frac_immune]
        results = np.array(pool.map(lambda args_list: self.i_sites(*args_list), args_list))  
        mean_i = np.mean(results, axis=1)
        mean_infected_fraction = mean_i/(self.dimension**2)
        
        # close log
        log.close()

        # save data
        self.data = (mean_i, np.atleast_1d(p_s), np.atleast_1d(p_i), np.atleast_1d(p_r), mean_infected_fraction, frac_immune)
        self.data_names = ("mean_infected_sites", "prob_suscept", "prob_infect", "prob_recover", "mean_infected_fraction", "immunity_fraction")
        self.save_data()

        # plot
        x = frac_immune
        y = mean_infected_fraction
        xlabel = r"Immunity Fraction, $f_{Immune}$"
        ylabel = r"Mean Infected Fraction, $\langle I \rangle$/N"
        title = r"Varying Immunity with $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r", $\mathbf{P_{S\rightarrow I} = }$" + f"{p_i:.2f}" + r" and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "immune_frac"
        self.plot(x, y, xlabel, ylabel, title, fig_label)

        return None
    
    def immunity_single(self,N,N_warmup,meas_freq,res):
        """
        Iterate through immunity fraction values from 0. -> 0.1 with constant update probabilites of 0.5. Plots the 
        fraction of average fraction of infected sites as a function of the immunity fraction.

        Data and plots from the simulation are saved inside the log_dir internal path. 
        
        Args:
            N (int): number of sweeps to perform per probability value
            N_warmup (int): warmup sweeps to perform before the N sweeps are computed (per probability value)
            meas_freq (int): how often to take measurements (in sweeps)  
            res (float): separation between probability points in the 1D probailitiy range        
        """
        # open log file
        log = open(self.log_path, "a")

        # probabilities fixed and immune fraction varied
        p_s = 0.5
        p_i = 0.5
        p_r = 0.5
        frac_immune = np.arange(start=0, stop=1, step=res)

        # main compute loop
        start_time = time.time()
        n = N//meas_freq
        results = np.empty(shape=(len(frac_immune), n))
        for i in range(0, len(frac_immune)):
            results[i][:] = self.i_sites(N, N_warmup, p_s, p_i, p_r, meas_freq, frac_immune[i])

        mean_i = np.mean(results, axis=1)
        mean_infected_fraction = mean_i/(self.dimension**2)
        
        # close log
        log.close()

        # save data
        self.data = (mean_i, np.atleast_1d(p_s), np.atleast_1d(p_i), np.atleast_1d(p_r), mean_infected_fraction, frac_immune)
        self.data_names = ("mean_infected_sites", "prob_suscept", "prob_infect", "prob_recover", "mean_infected_fraction", "immunity_fraction")
        self.save_data()

        # plot
        x = frac_immune
        y = mean_infected_fraction
        xlabel = r"Immunity Fraction, $f_{Immune}$"
        ylabel = r"Mean Infected Fraction, $\langle I \rangle$/N"
        title = r"Varying Immunity with $\mathbf{P_{R\rightarrow S} = }$" + f"{p_s:.2f} " + r", $\mathbf{P_{S\rightarrow I} = }$" + f"{p_i:.2f}" + r" and $\mathbf{P_{I\rightarrow R} = }$" + f"{p_r:.2f}"
        fig_label = "immune_frac"
        self.plot(x, y, xlabel, ylabel, title, fig_label)

        return None
    
    def mean_fraction(self, values, axis):
        """
        Wrapper function for mean fraction of infected sites.

        Used only to compute the error on this value.

        Args:
            values (numpy.ndarray): values to calculate the mean
        """

        return (np.mean(values, axis=axis))/(self.dimension**2)

    def i_sites(self, N, warmup, p_s, p_i, p_r, meas_freq, immune_fraction=0):
        """
        Method to perform a set amount of sweeps and return a numpy array of infected counts for each set of meas_freq sweeps.

        Args:
            N (int): number of sweeps to perform
            warmup (int): warmup sweeps to perform before the N sweeps commence; no measurements recorded for these sweeps
            p_s (float): probability to go from recovered to susceptible
            p_i (float): probability to go from susceptible to infected
            p_r (float): probability to go from infected to recovered
            meas_freq (int): how often (in sweeps) to record a measurement of infected sites
            immune_fraction (float): fraction of sites which are immmune for this run; immune sites never change state

        Returns:
            numpy.ndarray
        """
        # return number of infected sites after N sweeps of a SIRS run
        sirs = lattice(dimension=self.dimension,
                        visual=self.visual,
                        p_init=self.p_init,
                        plot_freq=self.plot_freq,
                        p = (p_s, p_i, p_r),
                        immune_fraction=immune_fraction
                        )
        
        # warmup
        sirs.sweep_method(int(warmup))

        # compute
        n = N//meas_freq
        infected_sites = np.empty(n)

        for i in range(0, n):
            sirs.sweep_method(meas_freq)
            infected_sites[i] = sirs.get_infected()

        if self.visual:
            plt.close(sirs.fig)
        
        #return
        return infected_sites

    def heat_plot(self, x, y, z, xlabel, ylabel, zlabel, title, fig_label):
        """
        Wrapper function to plot the given variables as a heatmap using matplotlib.pyplot.

        Args:
            x (array): x grid values 
            y (array): y grid values 
            z (2Darray): z values to be mapped onto the x, y grid
            xlabel (str): label for the x axis
            ylabel (str): label for the y axis
            zlabel (str): labelf for the z axis
            title (str): title for the plot
            fig_label (str): label to give the plot a unique name in the internal path log_dir.
        """

        #plot 
        fig, ax = plt.subplots(1, 1)
        im = ax.pcolormesh(x, y, z)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        cbar = fig.colorbar(im, ax = ax, label=zlabel)   

        # save figure
        fig_dir = self.log_dir
        fig_dir.mkdir(exist_ok=True)
        extension = "png"
        file_path = fig_dir /  f"{fig_label}-figure.{extension}"
        plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()

        return None
    
    def contour_plot(self, x, y, z, xlabel, ylabel, zlabel, title, fig_label):
        """
        Wrapper method to plot the given variables as a contour plot using matplotlib.pyplot.

        Args:
            x (array): x grid values 
            y (array): y grid values 
            z (2Darray): z values to be mapped onto the x, y grid
            xlabel (str): label for the x axis
            ylabel (str): label for the y axis
            zlabel (str): labelf for the z axis
            title (str): title for the plot
            fig_label (str): label to give the plot a unique name in the internal path log_dir.
        """

        #plot 
        fig, ax = plt.subplots(1, 1)
        im = ax.contourf(x, y, z)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        cbar = fig.colorbar(im, ax = ax, label=zlabel)   

        # save figure
        fig_dir = self.log_dir
        fig_dir.mkdir(exist_ok=True)
        extension = "png"
        file_path = fig_dir /  f"{fig_label}-figure.{extension}"
        plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()

        return None
    
    def plot(self, x, y, xlabel, ylabel, title, fig_label):
        """
        Wrapper method to plot the given using matplotlib.pyplot.

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
        fig_dir = self.log_dir
        fig_dir.mkdir(exist_ok=True)
        extension = "png"
        file_path = fig_dir /  f"{fig_label}-figure.{extension}"
        plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()

        return None
    
    def error_plot(self, x, y, xerr, yerr, xlabel, ylabel, title, fig_label):
        """
        Wrapper method to plot the given variables as an error plot using matplotlib.pyplot.

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
        fig_dir = self.log_dir
        fig_dir.mkdir(exist_ok=True)
        extension = "png"
        file_path = fig_dir /  f"{fig_label}-figure.{extension}"
        plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()

        return None
    
    def bootstrap(self, raw_data, q_func, args = None, k=1000):
        """
        Bootstrap method for error calculations.

        Take a sample of data and randomly sample from it to create a new sample of the same size k times. Compute the desired quantity
        with the bootstrapped samples. The error on the quantity calculated with the orginal sample is given by:
            sqrt(mean(quantity^2) - mean(quantity)^2)

        Args:
            raw_data (np.ndarray): 1D np.ndarray containing the sample data
            q_func (function): function which returns the desired quantity calculated using the sample data
            args (tuple): additional argument required for q_func
            k (int): number of bootstrapped samples to take

        Returns:
            float 
        """

        # random sample from input data 
        bootstrap_sample = np.random.choice(np.array(raw_data), size = (k, raw_data.size) , replace = True)

        # evaluate quantity
        q = np.array([q_func(bootstrap_sample[i, :], *args) for i in range(k)])

        # standard deviation of these means is the error
        # 1D array axis 0 ->
        q_mean = np.mean(q)
        q_squared_mean = np.mean(np.square(q))
        error = np.sqrt(q_squared_mean - np.square(q_mean))

        return error

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
    
