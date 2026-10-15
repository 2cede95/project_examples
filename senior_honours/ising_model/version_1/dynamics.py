import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import time
from pathlib import Path
from time import sleep
from os import devnull

class ising_lattice():
    """
    Spin lattice object.

    Square lattice of spin values {-1, 1} with operations on the lattice.

    Attributes:
        dimension (integer): side length of the square lattice
        lattice (numpy.ndarray, arr.dtype = float): square lattice with spin values at the sites 
        vsite_energy (numpy.vectorize): vectorised site_energy() method
        energy (float): total energy of the lattice computed with total_energy()
        spins (float): total number of spin sites in the lattice
    """

    def __init__(self, dimension, initialise):
        """
        Init method for ising_lattice object.

        Args:
            dimension (int): side length of the square Ising lattice
            initialise (str): type of lattice to create {'random', 'aligned'}
        """

        self.dimension = dimension

        match initialise:
            case 'random':
                self.lattice = self.create_lattice_disordered()
            case 'aligned':
                self.lattice = self.create_lattice_ordered()

        # vectorised methods
        self.vsite_energy = np.vectorize(self.site_energy)

        self.energy = self.total_energy()
        self.spins = dimension ** 2

        return None
    
    def create_lattice_disordered(self):
        """
        Return random numpy.ndarry object.

        Creates a (dimension, dimension) numpy.ndarry object with uniformly distributed spin values selected from {-1, 1}.
       
        Returns:
            numpy.ndarray
        """

        ising_lattice = np.random.choice([-1, 1], size = (self.dimension, self.dimension))

        return ising_lattice
        
    def create_lattice_ordered(self):
        """
        Return spin-aligned numpy.ndarry object.

        Create a (dimension, dimension) numpy.ndarray object with spins aligned randomly. i.e. all sites are 1 or all sites are -1 with 0.5 probability

        Returns:
            numpy.ndarray
        """

        rand_float = np.random.uniform(0, 1)

        if rand_float < 0.5:

            # spin up
            lattice = np.ones((self.dimension, self.dimension))
        else:

            # spin down
            lattice = -np.ones((self.dimension, self.dimension))

        return lattice

    def site_energy(self, column, row, exclude = None):
        """
        Computes the energy at a given site in the internal lattice.

        Formula for energy: 
            E = -J * nearest_neighbour_pairwise_sum(S_i * S_j)
        where J = 1 sets the units of energy as E = [Joules] / [J]

        Arguments: 
            column (int): site column
            row (int): site row
            exclude (tuple): coordinates to exclude from the sum 

        Returns:
            float 
        """
        
        energy = 0  

        # up down left right
        nearest_neighbours = (
            ((column + 1) % self.lattice.shape[0], row),
            ((column - 1) % self.lattice.shape[0], row),
            (column, (row + 1) % self.lattice.shape[1]),
            (column, (row - 1) % self.lattice.shape[1])
            )
        
        for i in nearest_neighbours:

            # exclude if specified in arguments (i.e. kawasaki nearest neighbour)
            # ensure exclude is a tuple
            if exclude is not None and i == tuple(exclude):
                continue

            energy += self.lattice[i] * self.lattice[column, row]

        return -energy
    
    def total_energy(self):
        """
        Computes the total energy of the internal lattice object.

        Sums over all sites energies in the lattice. Result is halved to correct for double counting.

        Returns: 
            float
        """
        
        column, row = np.indices(self.lattice.shape)
        total_energy = np.sum(self.vsite_energy(column, row))
        
        return total_energy/2    
    
    def abs_magnetisation(self): 
        """
        Calculate the absolute value of the total magnetisation of the internal lattice object.

        Formula for magnetisation:
            M = sum_over_sites(spin_at_site)

        Returns:
            float
        """

        total_magnetisation = np.sum(self.lattice)

        return np.abs(total_magnetisation)  

class markov():
    """
    Single Markov chain object.

    Contains an internal ising_lattice object and methods to update the markov chain stochasticly (Metropolis Algorithm) via Glauber or Kawasaki dynamics methods. 

    Attributes:
        thermal_energy (float): value of the thermal energy in the markov chain
        state (ising_lattice): internal ising lattice object to be updated each iteration
        sweeps (int): internal counter for sweeps elapsed in the markov chain
        sweep (int): unit of "time" in the markov chain equivalent to each spin being updated one time (on average)
        visual (matplotlib.axes.Axes): axis for the --visual_only plot argument. Used in visualisation()
        plot_freq (int): gap between plot updates in units of sweeps
        fig (matplotlib.figure.Figure): figure object for spin lattice visualisation during simulation
        line (matplotlib.axes.Axes): axis for spin lattice visualisation during simulation
        cbar (matplotlib.colorbar.Colorbar): colorbar object for spin lattice visualisation during simulation
        step_method (method): compute markov chain via Glauber or Kawasaki dynamics to determine trial states

    """
    def __init__(self, dimension, algorithm, thermal_energy, start = 'random', plot_freq = 10, visualise = False):
        """
        Init method for markov object.

        Args:
            dimension (int): side length of the internal ising_lattice object
            algorithm (str): trial state determination method
            thermal_energy (float): thermal energy of the markov chain
            start (str): type of lattice to create {'random', 'aligned'} at the beginning of the markov chain
            plot_freq (int): gap between plot updates in units of sweeps
            visualise (bool): if true then spin lattice visualisation is displayed during simulation
        """


        self.thermal_energy = thermal_energy
        self.state = ising_lattice(dimension=dimension, initialise=start)
        self.sweeps = 0
        self.sweep = self.state.dimension ** 2

        # for visual_only
        self.visual = None

        # for visualiser and visual_only
        self.plot_freq = plot_freq

        # visualiser
        if visualise:
            
            # start plot
            plt.ion()
            self.fig, self.ax = plt.subplots()
            self.line = self.ax.imshow(self.state.lattice, vmin = -1, vmax = 1)
            self.cbar = self.fig.colorbar(self.line, ax=self.ax, ticks=[-1, 1])
            self.cbar.ax.set_yticklabels(["Spin down", "Spin up"])
            self.fig.canvas.draw_idle()
            self.ax.set_title("Ising lattice spin state visualisation")
            plt.pause(0.001)


        # which algorithm should be used?
        match str(algorithm).lower():
            case 'glauber':
                self.step_method = self.glauber_step
            case 'kawasaki':
                self.step_method = self.kawasaki_step

        return None

    def glauber_step(self):
        """
        Generates a trial state using Glauber dynamics and updates the system according to the Metropolis update rules.

        A site is chosen at random and the energy difference is computed if the spin at that site had been flipped. If the energy increases then the trial state is rejected
        with a probability of exp(-energy_difference/kbT). If the energy decreases or stays the same then the new state is accepted with a probability of 1. If the trial state is accepted then 
        the internal lattice object is updated to the trial state and the energy of the lattice object is corrected for this new state. 
        """
        lattice = self.state.lattice
        state = self.state
        d = self.state.dimension


        #energy after the glauber trial
        r_col = np.random.randint(0, d)
        r_row = np.random.randint(0, d)

        site = (r_col, r_row)

        #delta E = Eafter - Ebefore  AND   E after = -Ebefore so deltaE = -2*Ebefore
        energy_difference = -2 * state.site_energy(r_col, r_row) 

        # Metropolis steps:
        # - If energy decreases or stays the same accept new state and flip the spin
        # - If energy increases accept with probability equal to the boltzmann weight; else do nothing
        
        p_boltzmann = np.exp(-energy_difference/self.thermal_energy)
        P_uniform = np.random.uniform(0, 1)

        if energy_difference <= 0 or P_uniform < p_boltzmann:
            lattice[site] *= -1
            state.energy += energy_difference

        return None

    def kawasaki_step(self):
        """
        Generates a trial state using Kawasaki dynamics and updates the system according to the Metropolis update rules.

        Two sites are chosen at random and the energy difference is computed as if the spin value at the sites has been swapped.  If the energy increases then the trial state is rejected
        with a probability of exp(-energy_difference/kbT). If the energy decreases or stays the same then the new state is accepted with a probability of 1. If the trial state is accepted then 
        the internal lattice object is updated to the trial state and the energy of the lattice object is corrected for this new state. 
        """
        lattice = self.state.lattice
        state = self.state
        d = self.state.dimension

        #4 random ints
        r_col_1 = np.random.randint(0, d)
        r_col_2 = np.random.randint(0, d)
        r_row_1 = np.random.randint(0, d)
        r_row_2 = np.random.randint(0, d)

        site_1 = (r_col_1, r_row_1)
        site_2 = (r_col_2, r_row_2)

        #check if the coordinates are the same
        if site_1 == site_2:
            return None
        
        #check if the spins are the same at each site
        if lattice[site_1] == lattice[site_2]:
            return None

        # neighbour check
        # distance between sites must be 1 exactly i.e. vertical or horizontal neighbours only
        neighbour = (
            np.abs(r_col_1 - r_col_2) == 1 and np.abs(r_row_1 - r_row_2) == 0 or 
            np.abs(r_col_1 - r_col_2) == 0 and np.abs(r_row_1 - r_row_2) == 1         
        )

        # if they are neighbours then exclude eachother in their lattice site sums
        if neighbour:
            exclude = (site_2, site_1)
        else:
            exclude = (None, None)

        # compute energy difference
        energy_difference = -2*(state.site_energy(r_col_1, r_row_1, exclude=exclude[0]) + state.site_energy(r_col_2, r_row_2, exclude=exclude[1]))

        # if energy increases accept with boltzmann, else accept always
        p_boltzmann = np.exp(-energy_difference/self.thermal_energy)
        p_uniform = np.random.uniform(0, 1)

        if energy_difference <= 0 or p_uniform < p_boltzmann:
            lattice[r_col_1][r_row_1], lattice[r_col_2][r_row_2] = lattice[r_col_2][r_row_2], lattice[r_col_1][r_row_1]        
    
        return None

    def animation_function(self, frame):
        """
        Function called to update the Ising lattice spin state visualisation from the --visual_only flag

        Args:
            frame (int): required argument for matplotlib.animation.FuncAnimation
        
        Returns:
            matplotlib.axes.Axes
        """

        # sweeps per frame
        freq = self.plot_freq

        # do the sweeps
        self.sweeps += freq
        for _ in range(0, int(freq*self.sweep)):
            self.step_method()

        # update plot
        self.visual.set_data(self.state.lattice)

        return (self.visual,)

    def visualisation(self):
        """
        Ising lattice spin state visualisation for the --visual_only flag

        Plots the spins and their sites and progresses the markov chain at a constant thermal energy
        """

        fig, ax = plt.subplots()

        #explicit vmin and vmax to fix colourbar in cold starts
        self.visual = ax.imshow(self.state.lattice, vmin=-1, vmax=1)
        cbar = fig.colorbar(self.visual, ax=ax, ticks=[-1, 1])
        cbar.ax.set_yticklabels(["Spin down", "Spin up"])
        ax.set_title("Ising lattice spin state visualisation")

        #animation
        ani = animation.FuncAnimation(fig=fig, func=self.animation_function, interval=1, cache_frame_data=False)
        plt.show()

        return None
    
    def compute(self, n):
        """
        Performs a specified number sweeps on the internal lattice object.

        Args:
            n (int): number of sweeps to perform
        """
        for _ in range(0, n):

            self.sweeps += 1
            # do one sweep
            for _ in range(0, self.sweep):
                self.step_method()  

        return None
        
    def compute_visual(self, n):
        """
        Performs a specified number sweeps on the internal lattice object. Includes an Ising lattice spin state visualisation.

        Plot information stored in the internal interactive plot axes and fig variables using matplotlib.pyplot.ion()

        Args:
            n (int): number of sweeps to perform
        """
        for _ in range(0, n):

            self.sweeps += 1
            # do one sweep
            for _ in range(0, self.sweep):
                self.step_method()  
            
            if self.sweeps % self.plot_freq == 0:
                # update plot with state
                self.line = self.ax.imshow(self.state.lattice)
                self.fig.canvas.draw_idle()
                plt.pause(0.001)


        return None
    
class simulation():
    """
    Object containing simulation data and logic.

    Attributes:

        thermal_energies (numpy.ndarray): min and max values of the thermal_energy to adiabatically iterate the markov chain through
        thermal_energy_gap (float): spacing between thermal energy values in the internal thermal_energies numpy.ndarray
        markov_chain (markov): internal markov object to adiabatically iterate and measure observables on
        error_method (method): method used to calculate errors on observable measurements; one of {'bootstrap', 'jackknife'}
        log_path (pathlib.Path): path to the log file save location
        data (tuple): tuple containing all of the observable data after the simulation concludes
        data_quantities (tuple): tuple containing the names (as str objects) of the different observables measured
        compute_method (method): sweep computation method; one of {'markov.compute_visual', 'markov.compute'}
    """
    def __init__(self, dimension = 50, algorithm = 'glauber', thermal_energy_range = (1.0, 3.0), thermal_energy_gap = 0.1, start = 'disordered', visualise = False, plot_freq = 10, error_method = 'bootstrap', log_path = devnull):
        """
        Init method for simulation object.

        Args:
            dimension (int): side length of the internal ising_lattice object
            algorithm (str): trial state determination method
            thermal_energy_range (tuple): min and max values (float) for the thermal energies to adiabatically iterate the internal markov object through
            thermal_energy_gap (float): spacing between thermal energy values between the specified thermal_energy_range
            start (str): type of lattice to create {'random', 'aligned'} at the beginning of the markov chain
            visualise (bool): if true then spin lattice visualisation is displayed during simulation
            plot_freq (int): gap between plot updates in units of sweeps
            error_method (str): method used to calculate errors on observable measurements
            log_path (pathlib.Path): path to the save location for the log file
        """

        # int() rounds down
        self.thermal_energies = np.linspace(thermal_energy_range[1],
                                            thermal_energy_range[0],
                                            int((thermal_energy_range[1] - thermal_energy_range[0])/thermal_energy_gap + 1)
                                            )
        
        # saved to use as error on critical temperature
        self.thermal_energy_gap = thermal_energy_gap

        self.markov_chain = markov(dimension=dimension,
                                    algorithm=algorithm,
                                    thermal_energy=thermal_energy_range[1],
                                    start=start,
                                    plot_freq = plot_freq,
                                    visualise=visualise)

        # handle error method choices
        match error_method:
            case 'bootstrap':
                self.error_method = self.bootstrap
            case 'jackknife':
                self.error_method = self.jackknife

        self.log_path = log_path

        # store data
        self.data = None
        self.data_quantities = ("energy", "energy_mean", "heat_capacity", "energy_mean_err",
                            "heat_capacity_err", "abs_magnetisation", "abs_magnetisation_mean",
                            "susceptibility", "abs_magnetisation_mean_err", "susceptibility_err",
                            "thermal_energy", "crit_T_heat_capacity", "crit_T_susceptibility")
        if visualise:
            # set compute method
            self.compute_method = self.markov_chain.compute_visual
        else:
            self.compute_method = self.markov_chain.compute


        return None
    
    def susceptibility(self, magnetisation, thermal_energy):
        """
        Computes the magnetic susceptibility (unitless).

        Not restricted for use with internal variables.

        Args:
            magnetisation (numpy.ndarray): 1D numpy array containing magnetisation measurements from a constant thermal energy value
            thermal_energy (float): thermal energy value which measurements took place

        Returns:
            float
        """


        # Susceptibilitiy in units of 1/J
        magnetisation_mean = np.mean(magnetisation)
        magnetisation_squared_mean = np.mean(np.square(magnetisation))
        N_spins = self.markov_chain.state.spins

        return 1/(N_spins*thermal_energy)*(magnetisation_squared_mean - magnetisation_mean**2)
    
    def heat_capacity(self, energy, thermal_energy):
        """
        Computes the heat capacity per spin (unitless).

        Not restricted for use with internal variables.

        Args:
            energy (numpy.ndarray): 1D numpy array containing energy measurements from a constant thermal energy value
            thermal_energy (float): thermal energy value which measurements took place

        Returns:
            float
        """
        # Heat capacity in units per kB
        energy_mean = np.mean(energy)
        energy_squared_mean = np.mean(np.square(energy))
        N_spins = self.markov_chain.state.spins

        return 1/(N_spins*thermal_energy**2)*(energy_squared_mean - energy_mean**2)

    def run(self, sweeps_per_temperature = 1000, warmup_sweeps = 100, correlation_gap = 10):
        """
        Main compute block of the simulaton.

        Data from the simulation is returned AFTER conclusion in the format:

                (energies, energy_means, heat_capacities, energy_means_err, heat_capacities_err,
                magnetisations, magnetisation_means, susceptibilities, magnetisation_means_err, susceptibilities_err,
                thermal_energies, T_crit_heat_capacity, T_crit_susceptibility)
        which have data types:
                (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                numpy.ndarray, tuple(T_crit, index), tuple(T_crit, index, err))

        Updates are saved to a log file at log_path but the DATA is not saved until save_data is called.

        Args:
            sweeps_per_temperature (int): number of sweeps to perform at each thermal energy value
            warmup_sweeps (int): number of sweeps to perform at the beginning of the simulation to ensure the simulation begins at an equilibrium state
            correlation_gap (int): number of sweeps to compute inbetween observable measurements to ensure states are uncorrelated (good statistics)

        Returns:
            tuple 
        """


        # save args for plotting
        self.sweeps_per_temperature = sweeps_per_temperature
        self.warmup_sweeps = warmup_sweeps
        self.correlation_gap = correlation_gap

        # open log file
        log = open(self.log_path, "a")

        # suggested values: sweeps = 1000, warmup_sweeps = 100, measurement_gap = 10
        # assigning internal variables local variables for performance in a large loop (I think?)
        state = self.markov_chain.state
        markov_chain = self.markov_chain
        error_method = self.error_method
        thermal_energies = self.thermal_energies
        compute = self.compute_method

        # create numpy arrays with preallocated sizes
        temperatures = len(thermal_energies)
        measurements_per_temp = int(sweeps_per_temperature/correlation_gap)
        total_measurements = measurements_per_temp*temperatures
        total_sweeps = warmup_sweeps + total_measurements*correlation_gap

        energies = np.empty((temperatures, measurements_per_temp))
        magnetisations = np.empty((temperatures, measurements_per_temp))

        # warmup 
        start_time = time.time()
        compute(warmup_sweeps)

        elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
        fraction_completed = markov_chain.sweeps/total_sweeps
        print(f"Warmup complete: {warmup_sweeps} sweeps @ T = {thermal_energies[0]:.2f} kbT/J in {elapsed_time} ({fraction_completed * 100:.2f}%)")
        print(f"Warmup complete: {warmup_sweeps} sweeps @ T = {thermal_energies[0]:.2f} kbT/J in {elapsed_time} ({fraction_completed * 100:.2f}%)", file=log)

        # energy and magnetisation have column unique to temperaute (ith column)
        # rows filled with the values at that temperature   

        # main block of code
        for i, thermal_energy in enumerate(thermal_energies):

            markov_chain.thermal_energy = thermal_energy
            for measurement in range(0, measurements_per_temp):
                compute(correlation_gap)
                energies[i, measurement] = state.total_energy()
                magnetisations[i, measurement] = state.abs_magnetisation()

            # need to round down after divison for progress at ~10%
            if i // temperatures/10 == 0:
                fraction_completed = markov_chain.sweeps/total_sweeps
                elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
                print(f"Progress: {markov_chain.sweeps} sweeps @ T = {thermal_energies[i]:.2f} kbT/J in {elapsed_time} ({fraction_completed * 100:.2f}%)")
                print(f"Progress: {markov_chain.sweeps} sweeps @ T = {thermal_energies[i]:.2f} kbT/J in {elapsed_time} ({fraction_completed * 100:.2f}%)", file=log) 

        elapsed_time = time.strftime("%H Hours %M Minutes %S Seconds", time.gmtime(time.time() - start_time))
        fraction_completed = markov_chain.sweeps/total_sweeps
        print(f"Simulation complete: {total_sweeps} sweeps @ T = {{{thermal_energies[0]:.2f}, {thermal_energies[1]:.2f}, ... {thermal_energies[-2]:.2f}, {thermal_energies[-1]:.2f}}} kbT/J in {elapsed_time:} ({fraction_completed * 100:.2f}%)")
        print(f"Simulation complete: {total_sweeps} sweeps @ T = {{{thermal_energies[0]:.2f}, {thermal_energies[1]:.2f}, ... {thermal_energies[-2]:.2f}, {thermal_energies[-1]:.2f}}} kbT/J in {elapsed_time:} ({fraction_completed * 100:.2f}%)", file=log)

        # compute quantities of interest

        # energy
        # 1D array axis 0 is right; 2D array axis 0 down
        energy_means = np.mean(energies, axis = 1)
        energy_means_err = np.array([error_method(energies[i, :], q_func=np.mean, args=(0,)) for i in range(temperatures)])

        # heat capacity
        heat_capacities = np.array([self.heat_capacity(energy=energies[i, :], thermal_energy=thermal_energies[i]) for i in range(temperatures)])
        heat_capacities_err = np.array([error_method(energies[i, :], q_func=self.heat_capacity, args = (thermal_energies[i],)) for i in range(temperatures)])

        # mean absolute magnetisation 
        magnetisation_means = np.mean(magnetisations, axis = 1)
        magnetisation_means_err = np.array([error_method(magnetisations[i, :], q_func=np.mean, args=(0,)) for i in range(temperatures)])

        # susceptibility
        susceptibilities = np.array([self.susceptibility(magnetisation=magnetisations[i, :], thermal_energy=thermal_energies[i]) for i in range(temperatures)])
        susceptibilities_err = np.array([error_method(magnetisations[i, :], q_func=self.susceptibility, args=(thermal_energies[i],)) for i in range(temperatures)])

        # critical temperature
        critical_temp_err = self.thermal_energy_gap
        max_heat_capacity_ind = np.argmax(heat_capacities)
        T_crit_heat_capacity = (thermal_energies[max_heat_capacity_ind], max_heat_capacity_ind, critical_temp_err)
        max_susceptibility_ind = np.argmax(susceptibilities)
        T_crit_susceptibility = (thermal_energies[max_susceptibility_ind], max_susceptibility_ind, critical_temp_err)

        print(f"Critical Temperature estimate from Heat Capacity data: {T_crit_heat_capacity[0]:.2f} ± {critical_temp_err:.2f} KbT/J")
        print(f"Critical Temperature estimate from Heat Capacity data: {T_crit_heat_capacity[0]:.2f} ± {critical_temp_err:.2f} KbT/J", file=log)
        print(f"Critical Temperature estimate from Susceptibility data: {T_crit_susceptibility[0]:.2f} ± {critical_temp_err:.2f} KbT/J")
        print(f"Critical Temperature estimate from Susceptibility data: {T_crit_susceptibility[0]:.2f} ± {critical_temp_err:.2f} KbT/J", file=log)

        # update internal data object for accesibility outside of run()
        self.data = (energies, energy_means, heat_capacities, energy_means_err, heat_capacities_err,
                magnetisations, magnetisation_means, susceptibilities, magnetisation_means_err, susceptibilities_err,
                thermal_energies, T_crit_heat_capacity, T_crit_susceptibility)
        

        # close log file
        log.close()
        
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
    
    def jackknife(self, raw_data, q_func, args = None):
        """
        Jackknife method for error calculations.

        Systematically remove each value from the sample data uniquely to form len(raw_data) samples each of length len(raw_data) - 1.
        Compute the desired quantity for each of the jackknife samples. The error on the quantity calculated with the original sample is
        given by:
              sqrt(sum_over_jackknife_samples( (jackknife_quantity - original_quantity)^2 ))

        Args:
            raw_data (np.ndarray): 1D np.ndarray containing the sample data
            q_func (function): function which returns the desired quantity calculated using the sample data
            args (tuple): additional argument required for q_func

        Returns:
            float 
        """
        # enforce input array as numpy array
        raw_data = np.array(raw_data)
        k = raw_data.size

        # remove each element separetly for (k, k) numpy array of k samples
        jackknife_sample = np.array([np.delete(raw_data, i) for i in range(k)])

        # evaluate quantity
        q = np.array([q_func(jackknife_sample[i, :], *args) for i in range(k)])

        # compute error
        original_q = q_func(raw_data, *args)
        difference_squared = np.square(q - original_q)
        error = np.sqrt(np.sum(difference_squared))

        return error

    def save_data(self, timestamp):
        """
        Save the data stored in the internal data variable.

        Data is stored in ./{timestamp}/data. Each quantity has its own .txt file.

        Args:
            timestamp (str): the timestamp the simulation begun via datetime.datetime.now().strftime()
        """
        # file extension to save data as
        extension = "txt"

        # esnure directory for this run exists
        time_dir = Path(f"{timestamp}")
        time_dir.mkdir(exist_ok=True)

        # create directory for data
        data_dir = time_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # save the files
        for i, name in enumerate(self.data_quantities):
            file_path = data_dir / f"{name}.{extension}"
            np.savetxt(file_path, self.data[i])

        # also save the last ising lattice state
        filepath = data_dir / f"final_spin_states.{extension}"
        np.savetxt(filepath, self.markov_chain.state.lattice)
        
        return None

    def simulation_params(self):
        """
        Returns internal parameters for use with plotter class.

        Format:
            (dimesnion of ising lattice, thermal energy gap, sweeps per temp of most recent run,
            warmup sweeps of most recent run, correlation gap of most recent run, list of thermal energies,
            compute method, error methods)

        Returns:
            tuple
        """

        params = (self.compute_method, self.thermal_energy_gap, self.sweeps_per_temperature, self.warmup_sweeps, self.correlation_gap, self.thermal_energies, self.error_method)
        
        return params

class plotter():
    """
    Object containing data and logic for plotting in a consistent and easily configurable manner.

    Attributes:
        thermal_energies (np.ndarray): thermal energy data for the x-axis of the plots
        fig (matplotlib.figure.Figure): figure object for plotting
        axes (tuple): flattened tuple of axes for plots
        plot_data (tuple): data to be plotted formatted in a tuple as below:
                            (y_data, error on y_data, title of axes, x_label of axes, y_label of axes, (critical temperature, index) )  
        size (tuple): dimensions of the plot via matplotlib.pyplot.gcf().set_size_inches()      
    """
    def __init__(self, data, algorithm, params = None):
        """
        Init method for plotter object.

        Args:
            data (tuple): tuple of data from the simulation object (i.e. simulation.data); simulation.run() must have been completed at least one time for this to not return a None type object
                          data must be in the following format:
                                            (energies, energy_means, heat_capacities, energy_means_err, heat_capacities_err,
                                            magnetisations, magnetisation_means, susceptibilities, magnetisation_means_err, susceptibilities_err,
                                            thermal_energies, T_crit_heat_capacity, T_crit_susceptibility)
                          which have data types:
                                            (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                            numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                            numpy.ndarray, tuple(T_crit, index), tuple(T_crit, index, err))
            algorithm (str): algorithm which the data was generated with
            params (tupe): tuple containing data from simulation.simulation_params() useful for plotting 
                            must be in the following format:
                                            (dimension of ising lattice, thermal energy gap, sweeps per temp of most recent run,
                                            warmup sweeps of most recent run, correlation gap of most recent run, list of thermal energies,
                                            compute method, error methods)
                            which have data types:
                                            (int, float, int, int, int, numpy.ndarray, method, method)

        """

        # data from an instance of simulation.run()
        # flipped for increasing temperature plots
        # flip the data to be ordered in increasing temperature (cannot use np.flip because data does not come rectangular)
        # energies, mean energies, heat capacities, mean energy errors, heat capacity errors  = data[0:4]
        mean_energies = np.flip(data[1])
        mean_energies_err = np.flip(data[3])
        heat_capacities = np.flip(data[2])
        heat_capacities_err = np.flip(data[4])

        # absolute magnetisations, mean absolute magnetisations, susceptibilities, magnetisation errors, susceptibility errors  = data[5:9]
        mean_magnetisations = np.flip(data[6])
        mean_magnetisations_err = np.flip(data[8])
        susceptibilities = np.flip(data[7])
        susceptibilities_err = np.flip(data[9])

        # thermal energies = data[10]
        # need to keep for plots 
        self.thermal_energies = np.flip(data[10])

        # critical temperatures
        T_crit_susceptibility = data[12]
        T_crit_heat_capacity = data[11]

        # captial for plot main title
        algorithm_name = algorithm
        algorithm_name = algorithm_name.capitalize()
        error_method_name = params[-1].__name__.capitalize()

        if params is not None:
            self.subptitle = f"{algorithm_name} Dynamics w/ {params[2]} Sweeps per {params[1]} " + r"$\mathbf{k_{b}T J^{-1}}$" + f" ({params[3]} Warmup Sweeps, Errorbars via {error_method_name} Resampling)"

        else:
            self.subptitle = ""

        # format data
        # both algorithms have meaningful energy and heat capacity data
        energy_plot_data = (mean_energies,
                            mean_energies_err,
                            r"Average Energy $\langle E \rangle$ against Thermal Energy T",
                            r"Thermal Energy T [$k_{b}T J^{-1}$]",
                            r"Average Energy $\langle E \rangle$",
                            T_crit_heat_capacity
                            )
        heat_capacity_plot_data = (heat_capacities,
                                    heat_capacities_err,
                                    r"Heat Capacity per Spin $C_{v}$ against Thermal Energy T",
                                    r"Thermal Energy T [$k_{b}T J^{-1}$]",
                                    r"Heat Capacity per Spin $C_{v}$",
                                    T_crit_heat_capacity
                                    )

        # handle axes
        match algorithm:
            case 'glauber':
                
                # axes
                self.fig, (axes) = plt.subplots(2, 2)
                plt.subplots_adjust(wspace=0.5, hspace=0.5)

                # magnetisation and suceptibility data only meaningful for glauber dynamics
                magnetisation_plot_data = (mean_magnetisations,
                            mean_magnetisations_err,
                            r"Absolute Magnetisation $\left| M \right|$ against Thermal Energy T",
                            r"Thermal Energy T [$k_{b}T J^{-1}$]",
                            r"Absolute Magnetisation $\left| M \right|$",
                            T_crit_susceptibility
                            )
                
                susceptibility_plot_data = (susceptibilities,
                            susceptibilities_err,
                            r"Magnetic Susceptibility $\chi$ against Thermal Energy T",
                            r"Thermal Energy T [$k_{b}T J^{-1}$]",
                            r"Magnetic Susceptibility $\chi$",
                            T_crit_susceptibility
                            )
                
                # package plot data into a tuple for show_plots
                self.plot_data = (energy_plot_data, heat_capacity_plot_data, magnetisation_plot_data, susceptibility_plot_data)

                # plot size
                # (horizontal, vertical)
                self.size = (12, 9)


            case 'kawasaki':

                # axes
                self.fig, (axes) = plt.subplots(1, 2)
                plt.subplots_adjust(wspace=0.5, hspace=0.5)

                # package plot data into a tuple for show_plots
                self.plot_data = (energy_plot_data, heat_capacity_plot_data)

                # (horizontal, vertical)
                # plot size
                self.size = (12, 6)

        # need to flatten axes tuple to a 1D array for the show_plot() method
        self.axes = axes.flatten()

        return None

    def plot_ax(self, y_data, err, title, x_label, y_label, T_crit, ax):
        """
        Plot the data for a single matplotlib.axes.Axes object.

        Args:
            y_data (np.ndarray): data to plot on the y-axis
            err (np.ndarray): error data for the y_data values
            title (str): title of the matplotlib.axes.Axes object
            x_label (str): label for the x axis on the matplotlib.axes.Axes object
            y_label (str): label for the y axis on the matplotlib.axes.Axes object
            T_crit (tuple): contains the critical temperature to be plotted as a vertical line on all axes and the index for the 
                            critical temperature in the internal thermal_energies np.ndarray object
            ax (matplotlib.axes.Axes): matplotlib.axes.Axes object to plot on
        """
        ax.errorbar(self.thermal_energies,
                    y_data,
                    yerr = err,
                    ecolor = "black",
                    color = "tab:red",
                    marker = "s",
                    markersize = 2,
                    markerfacecolor = "tab:red",
                    markeredgecolor = "black",
                    label = "Error Bars",
                    capsize = 2,
                    zorder = 3)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

        # negative indices because the data order has been flipped in __init__
        # + 1 because -1 is the "0th" spot from the right side of the numpy.ndarray
        ax.vlines(x=self.thermal_energies[-1*(T_crit[1]+1)],
                    ymax=y_data[-1*(T_crit[1]+1)],
                    ymin=np.min(y_data),
                    linestyle='--',
                    zorder=1,
                    label = r"$T_{c}$ =" + f" {T_crit[0]:.2f}" + f" ± {T_crit[2]:.2f}",
                    color = "tab:cyan"
                    )
        # move legend outside the plot (1.0, 1.0) is the top right (0.5, 0.5) is the center
        ax.legend()

        return None

    def show_plots(self, timestamp, save=True):
        """
        Save the plot, draw plots from memory to the screen using plt.show() and perform additional formatting.

        Args:
            timestamp (str): name of the file to save the figure as
            save (bool): if true the plot will be saved in the ./{timestamp} directory. OS independant via pathlib.Path
        
        """
        for i, data in enumerate(self.plot_data):

            # unpack data as the positional arguments
            self.plot_ax(*data, self.axes[i])

        # set figure size
        plt.gcf().set_size_inches(*self.size)

        # main title 
        self.fig.suptitle(self.subptitle, fontweight = "bold")

        if save:

            # ensure directory for this run exists
            fig_dir = Path(f"{timestamp}")
            fig_dir.mkdir(exist_ok=True)
    
            # save plots
            extension = "png"
            file_path = fig_dir /  f"figure.{extension}"
            plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight')

        plt.show()

        return None

