import numpy as np
import matplotlib.pyplot as plt
import time
from os import devnull
from pathlib import Path

class markov_chain:
    
    def __init__(self, length: int, beta: float, init_cond: str) -> None:

        # simulation parameters
        self.length = length
        self.beta = beta

        # fields
        self.J = self.init_jarray(length)
        self.h = self.init_harray(length)
        
        # grid 1 = up; down = -1
        if 'random' == init_cond:
            self.spins = self.init_rarray(length)
        elif 'up' == init_cond:
            self.spins = self.init_oarray(length, 1)
        else:
            self.spins = self.init_oarray(length, -1)

    def init_rarray(self, N: int) -> np.ndarray:
        """
        Initialise NxN 2D numpy array with spins values 1 or -1; wrapper for numpy.random.choice

        Args:
            N (int): length of square numpy array

        Returns:
            array with random spin orientations (numpy.ndarray)
        """
        a = np.random.choice(a=(-1,1), size=(N, N))

        return a
    
    def init_oarray(self, N: int, spin: int) -> np.ndarray:
        """
        Initialise NxN 2D numpy array with aligned spins; wrapper for numpy.full

        Args:
            N (int): length of square numpy array
            spin (int): spin value to fill array with
            
        Returns:
            array with aligned spin orientations (numpy.ndarray)
        """

        a = np.full(fill_value=spin, shape=(N, N))

        return a
    
    def init_jarray(self, N: int) -> np.ndarray:
        """
        Initialise coupling parameter matrices. 

        This function is effectively a wrapper for numpy.ones; modify the code for a given coupling matrix
         
        Must have shape (4, N, N) for the nearest neighbour coupling constant in 2D corresponding to +0, -0, +1, -1

        i.e. J[1, i, j] = 5 means that the spin at i and j has a coupling strength of 5 with it's neighbour in the + 0 axis
                                                        (J[0, i, j])
                                                             s
                                              (J[3, i, j]) s s s (J[2, i, j])
                                                             s
                                                        (J[1, i, j])
       
        Args:
            N (int): length of square numpy array

        Returns:
            J matrix (numpy.ndarray)
        """
        return np.ones(shape=(4, N, N))
    
    def init_harray(self, N: int) -> np.ndarray:
        """
        Initialise the external field matrix.

        The spin at site i, j has an energy from the local contribution S_ij * h_ij

        Currently a wrapper for numpy.zeros; modify the code for a given external field.

        Output must have shape (N, N)

        Args:
            N (int): length of square numpy array

        Returns:
            h matrix (numpy.ndarray)
        """
        return np.zeros(shape=(N, N))
    
    def energy(self, arr: np.ndarray, row: int, col: int) -> float:
        """
        Return energy of a given site in the Ising lattice. 

        Performs a sum over the product of nearest neighbour spin sites.

        Periodic boundary conditions.

        Args:
            arr (numpy.ndarray): grid of spins
            row, col (int): coordinates of a site on the grid

        Returns:
            total energy at the site (float)
        """

        # periodic boundaries
        n, m = arr.shape

        # external field + coupliong; shape (4,)
        j = self.J[:, row, col]

        # E_i = S_i * sum_j(S_j * (J_ij + h_i))
        sum = arr[row, col] * (
            j[0]*arr[(row + 1) % n, col] +
            j[1]*arr[(row - 1) % n, col] +
            j[2]*arr[row, (col + 1) % m] +
            j[3]*arr[row, (col - 1) % m]
            )
        
        return -(sum + self.h[row, col]*arr[row, col])
    
    def total_energy(self, arr: np.ndarray) -> float:
        """
        Compute total energy of the spin grid by summing over individual site energies.

        Corrects double counting with a factor of 1/2 in the nearest neighbour sums.

        Args:
            arr (numpy.ndarray): grid of spins   

        Returns:
            total energy of the grid (float)
        """

        e = self.energy  
        n, m = arr.shape 
        h = self.h

        energy = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                J = self.J[:, i, j]
                sum = arr[i, j] * (
                J[0]*arr[(i + 1) % n, j] +
                J[1]*arr[(i - 1) % n, j] +
                J[2]*arr[i, (j + 1) % m] +
                J[3]*arr[i, (j - 1) % m]
                )
                energy += sum/2 + h[i, j]*arr[i, j]

        return -energy
    
    def total_mag(self, arr: np.ndarray) -> float:
        """
        Compute the total magnetisation of the sping grid. Wrapper for numpy.sum

        Args:
            arr (numpy.ndarray): grid of spins 
        
        Returns:
            total magnetisation of grid (float)
        """

        return np.sum(arr)

    def glauber(self, arr: np.ndarray) -> tuple:
        """
        Perform Glauber update on a grid of spins. 

        If the energy is decreased by a single spin flip at a random site then arr is updated in place with this new spin value. Otherwise, 
        the change is accepted with the Boltzmann factor.

        Accounts for an external field, h and directional coupling constants J.

        Args:
            arr (numpy.ndarray): grid of spins 

        Returns:
            update array (numpy.ndarray), change in energy (float)
        """
        energy = self.energy
        beta = self.beta
        row, col = np.random.randint(low=0, high=arr.shape[0], size=(2,))
 
        delta_e = -2*energy(arr, row, col)

        # metropolis check
        if delta_e < 0 or np.exp(-delta_e/beta) > np.random.random():
            arr[row, col] = -arr[row, col] 
            delta_mag = 2*arr[row,col]
            return arr, delta_e, delta_mag

        # no changes
        delta_e = delta_mag = 0
        
        return arr, delta_e, delta_mag
    
    def kawasaki(self, arr: np.ndarray) -> tuple:
        """
        Perform Kawasaki update on a grid of spins. 

        If the energy is decreased by a spin swap at two random sites then arr is updated in place with this new spin value. Otherwise, 
        the change is accepted with the Boltzmann factor.

        Special optimisations:
            - if the the random selected sites have the same spin then the original array is returned
            - if the randomly chosen sites are the same then the original array is returned
            - if sites are neighbours then their interaction is ignored

        Accounts for an external field h and directional coupling constants J.

        Args:
            arr (numpy.ndarray): grid of spins 

        Returns:
            updated array (numpy.ndarray), change in energy (float), change in magnetisation (float)
        """
        beta = self.beta
        e = self.energy
        row1, col1, row2, col2 = np.random.randint(low=0, high=arr.shape[0], size=(4,))
        delta_mag = 0
        delta_e = 0

        # no change in energy if the swapped site is the same
        if (row1, col1) == (row2, col2):
            return arr, delta_e, delta_mag
        
        #similarly, if the spins are the same then no change
        if arr[row1, col1] == arr[row2, col2]:
            return arr, delta_e, delta_mag
        
        # sites affected by the swap
        n, m = arr.shape 
        affected_sites = {
            (row1, col1),
            (row2, col2),
            ((row1 + 1) % n, col1),
            ((row1 - 1) % n, col1),
            (row1, (col1 + 1) % m),
            (row1, (col1 - 1) % m),
            ((row2 + 1) % n, col2),
            ((row2 - 1) % n, col2),
            (row2, (col2 + 1) % m),
            (row2, (col2 - 1) % m)
        } # set -> unique items handles neighbour case

        def affected_energy():
            energy = 0.
            for i in affected_sites:
                energy += e(arr, i[0], i[1])
            return energy

        # perform kawasaki swap
        e_before = affected_energy()
        arr[row1, col1], arr[row2, col2] = arr[row2, col2], arr[row1, col1]
        e_after = affected_energy()
        delta_e = (e_after - e_before)/2 # double counting

        # metropolis check
        if delta_e < 0 or np.exp(-delta_e/beta) > np.random.random(): return arr, delta_e, delta_mag
        else: arr[row1, col1], arr[row2, col2] = arr[row2, col2], arr[row1, col1] # swap back
        
        # no changes
        delta_e = delta_mag = 0.

        return arr, delta_e, delta_mag
    
    def update_beta(self, beta: float) -> None:
        """
        Update internal beta variable without losing the spin grid.

        Args:
            beta (float): new beta value
        """
        self.beta = beta

class ising_model:

    def __init__(self, length: int, init_cond: str, correlation_gap: int, warmup: int, sweeps: int, save_dir: Path, visual: bool, error_method = 'bootstrap') -> None:

        # sim params
        self.correlation_gap = correlation_gap
        self.length = length
        self.sweep = self.length**2
        self.init_cond = init_cond
        self.warmup_sweeps = warmup
        self.sweeps = sweeps
        if 'jackknife' == error_method:
            self.error_method = self.jackknife
        else:
            self.error_method = self.bootstrap 
        
        # filepath for saving data
        self.save_dir = save_dir
        if save_dir != devnull:
            self.data_dir = save_dir / "data"
            self.data_dir.mkdir(exist_ok=True)
            self.fig_dir = save_dir / "figures"
            self.fig_dir.mkdir(exist_ok=True)
        else:
            self.data_dir = self.fig_dir = devnull

        if True == visual:
            self.sim = self.vsimulation
        else:
            self.sim = self.simulation
        
        return None
    
    def susceptibility(self, mag: np.ndarray, beta: float, N_spins: int) -> np.floating:
        """
        Compute magnetic susceptibility for a given state defined by a set of magnetisation values.

        Args:
            mag (numpy.ndarray): magnetisation values
            beta (float): inverse of thermal energy
            N_spins (int): number of spins the spin grid

        Returns:
            susceptibility (float)
        """

        return np.var(mag)/(beta*N_spins)
    
    def heat_capacity(self, energy: np.ndarray, beta: float, N_spins: int) -> np.floating:
        """
        Compute heat capacity for a given state defined by a set of energy values.

        Args:
            energy (numpy.ndarray): energy values
            beta (float): inverse of thermal energy
            N_spins (int): number of spins the spin grid

        Returns:
            heat capacity (float)
        """
                
        return np.var(energy)/(np.square(beta)*N_spins)
  
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

    def jackknife(self, vals: np.ndarray, func, args = (None,)) -> np.floating:
        """
        Compute error on a given function func(data) via the jackknife method. 
        
        One element is removed from the vals array and the statistic is computed; repeat until all 
        elements have been removed once. The returned error is the standard deviation on the 
        jackknifed statistic values.

        Args:
            vals (numpy.ndarray): input data for function func
            func (function):  function with signature func(data, arg1, arg2, arg3, ...)
            args (tuple): tuple of arguments to be unpackaged into func like func(data, *args)

        Returns:
            error on the function evaluation with the given data (float)
        """
        # jackknife samples
        length = vals.size
        jk = [np.delete(vals, i) for i in range(length)]

        # evalute function
        actual = func(vals, *args)
        q = [func(jk[i], *args) for i in range(length)]

        return np.sqrt(np.sum(np.square(q - actual)))

    def simulation(self, markov_chain: markov_chain, N: int, beta: float, update) -> tuple:
        """
        Perform N sweeps on a grid of spin values using a Markov Chain. Metropolis update rules
        determine system evolution. 

        Takes energy and magnetisation measurements after every self.correlation_gap sweeps.

        Args:
            markov_chain (markov_chain): markov chain object to evolve
            N (int): number of sweeps to perform
            beta (float): thermal energy kbT
            update (callable): function which applies dynamics update rules

        Returns:
            energy measurements (numpy.ndarray), magnetisation measurements (numpy.ndarray)
        """
        # sim params
        sweep = self.sweep
        cgap = self.correlation_gap
        spins = markov_chain.spins
        measurements = N//cgap

        # setup main loop
        energy = np.empty(shape=(measurements,))
        magnetisation = np.empty(shape=(measurements,))
        energy[0] = markov_chain.total_energy(spins)
        magnetisation[0] = markov_chain.total_mag(spins)
        start_time = time.time()

        # compute loop
        for i in range(1, measurements):
            
            # setup observables
            # energy[i] = energy[i-1]
            magnetisation[i] = magnetisation[i-1]

            # sweeps 
            for _ in range(sweep*cgap):
                spins, delta_e, delta_mag = update(spins)
                # energy[i] += delta_e
                magnetisation[i] += delta_mag

            energy[i] = markov_chain.total_energy(spins)

        elapsed_time = time.strftime("%H hours %M minutes %S seconds", time.gmtime(time.time() - start_time))
        print(f"{N} sweeps @ β = {beta:.2f} complete in {elapsed_time}")

        return energy, magnetisation
    
    def vsimulation(self, markov_chain: markov_chain, N: int, beta: float, update) -> tuple:
        """
        Perform N sweeps on a grid of spin values using a Markov Chain. Metropolis update rules
        determine system evolution. 

        Takes energy and magnetisation measurements after every self.correlation_gap sweeps.

        Args:  
            markov_chain (markov_chain): markov chain object to evolve
            N (int): number of sweeps to perform
            beta (float): thermal energy kbT
            update (callable): function which applies dynamics update rules

        Returns:
            energy measurements (numpy.ndarray), magnetisation measurements (numpy.ndarray)
        """
        # sim params
        sweep = self.sweep
        cgap = self.correlation_gap
        spins = markov_chain.spins
        measurements = N//cgap

        # setup main loop
        energy = np.empty(shape=(measurements,))
        magnetisation = np.empty(shape=(measurements,))
        energy[0] = markov_chain.total_energy(spins)
        magnetisation[0] = markov_chain.total_mag(spins)
        start_time = time.time()

        # interactive plot
        fig, ax = plt.subplots(1, 1)
        hm = ax.imshow(spins, vmin=-1, vmax=1, origin='lower')
        cbar = plt.colorbar(hm)
        plt.title(f"Spin Grid Visualisation, β = {beta:.2f}", fontweight="bold")
        plt.ion()
        plt.pause(0.5)
        plot_freq = 10 # sweeps

        # compute loop
        for i in range(1, measurements):
            
            # setup observables
            energy[i] = energy[i-1]
            magnetisation[i] = magnetisation[i-1]

            # sweeps
            for _ in range(sweep*cgap):
                spins, delta_e, delta_mag = update(spins)
                energy[i] += delta_e
                magnetisation[i] += delta_mag

            # update plot
            if (i)%(cgap//plot_freq) == 0:
                hm.set_data(spins)
                plt.pause(0.001)

        plt.ioff()
        plt.close()

        elapsed_time = time.strftime("%H hours %M minutes %S seconds", time.gmtime(time.time() - start_time))
        print(f"{N} sweeps @ β = {beta:.2f} complete in {elapsed_time}")

        return energy, magnetisation
     
    def beta_sim(self, beta_min: float, beta_max: float, beta_res: int, dynamics) -> list:
        """
        Perform multiple simulation calls with a range of beta values. Takes the desired dynamics update rules as an argument.

        All measurements returned in a tuple and saved to disk if class is initialised with a save directory.

        Args:
            beta_min (float): minimum beta value for the range
            beta_max (float): maximum beta value for the range
            bet_res (int): number of beta values to iterate over
            dynamics (callable): function which applies dynamics update rules

        Returns:
            tuple containing all the numpy arrays from the simulation (tuple)
        """
        # sim params
        sweeps = self.sweeps
        sweep = self.sweep
        heat_capacity = self.heat_capacity
        susceptibility = self.susceptibility
        sim = self.sim
        length = self.length
        init_cond = self.init_cond
        warmup = self.warmup_sweeps
        cgap = self.correlation_gap
        err = self.error_method
        measurements = sweeps//cgap

        # setup; hot -> cold
        beta_vals = np.linspace(beta_max, beta_min, beta_res)
        energy = np.zeros(shape=(len(beta_vals), measurements))
        mean_e = np.zeros_like(beta_vals)
        heatc = np.zeros_like(beta_vals)
        mag = np.zeros_like(energy)
        mean_m = np.zeros_like(beta_vals)
        suscept = np.zeros_like(beta_vals)

        # dynamics choice
        mc = markov_chain(length, beta_vals[0], init_cond)
        spins = mc.spins
        if dynamics == 'kawasaki':
            update = mc.kawasaki
        else:
            update = mc.glauber

        # warmup
        start_time = time.time()
        for _ in range(0, warmup*sweep):
            spins, _, _ = update(spins)
        elapsed_time = time.strftime("%H hours %M minutes %S seconds", time.gmtime(time.time() - start_time))
        print(f"{warmup} warmup sweeps @ β = {beta_vals[0]:.2f} complete in {elapsed_time}")

        # compute loop
        for i, beta in enumerate(beta_vals):
            mc.update_beta(beta)
            e, m = sim(mc, sweeps, beta, update)
            energy[i, :] = e
            mean_e[i] = np.mean(e)
            heatc[i] = heat_capacity(e, beta, sweep) # per spin
            mag[i, :] = m
            mean_m[i] = np.mean(m)
            suscept[i] = susceptibility(m, beta, sweep) # per spin

        # compute errors; for bootstrap k=1000 is the default
        emean_e = [err(energy[i], np.mean, args = (0, )) for i in range(len(beta_vals))]
        emean_m = [err(mag[i], np.mean, args = (0, )) for i in range(len(beta_vals))]
        eheatc = [err(energy[i], heat_capacity, args = (beta_vals[i], sweep)) for i in range(len(beta_vals))]
        esuscept = [err(mag[i], susceptibility, args = (beta_vals[i], sweep)) for i in range(len(beta_vals))]

        # reverse list order for cold -> hot
        results = beta_vals, mean_e, mean_m, heatc, suscept, emean_e, emean_m, eheatc, esuscept
        results = [x[::-1] for x in results]

        # save data
        if devnull != self.save_dir:
            arr = results
            name = ["beta", "mean_energy", "mean_mag", "heat_capacity",
                    "susceptibility", "mean_energy_err", "mean_mag_err",
                    "heat_capacity_err", "susceptibility_err"]
            self.save_nparray(arr, name)

        return results
    
    def plot_glauber(self, beta_vals: np.ndarray, mean_e: np.ndarray, mean_m: np.ndarray, heatc: np.ndarray, suscept: np.ndarray, emean_e: np.ndarray, emean_m: np.ndarray, eheatc: np.ndarray, esuscept: np.ndarray) -> None:
        """
        Preset plot configuartion for the Glauber simulation. Argument signature matches the output signature of self.beta_sim.

        Args:
            beta_vals (numpy.ndarray): beta values 
            mean_e (numpy.ndarray): mean energy values 
            mean_m (numpy.ndarray): not used
            heatc (numpy.ndarray): heat capacity values
            suscept (numpy.ndarray): susceptibility values
            emean_e (numpy.ndarray): error on mean energy values
            emean_m (numpy.ndarray): error on mean magnetisation values
            eheatc (numpy.ndarray): error on heat capacity values
            esuscept (numpy.ndarray): error on susceptibility values
        """      
        # plot setup
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
        plot = lambda ax, x, y, err: ax.errorbar(x,
                    y,
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
        xlabel = r"Thermal Energy T [$\text{Joules per J}$]"

        # mean energy
        plot(ax1, beta_vals, mean_e, emean_e)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"Average Energy $\langle E \rangle$, [$J$]")

        # heat capacity
        plot(ax2, beta_vals, heatc, eheatc)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"Heat Capacity per Spin $C_{v}$, [$k_{B}$]")

        # mean magnetisation
        plot(ax3, beta_vals, mean_m, emean_m)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(r"Average Magnetisation $\langle M \rangle$, [unitless]")

        # susceptibility
        plot(ax4, beta_vals, suscept, esuscept)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(r"Magnetic Susceptibility $\chi$, [$\text{unitless}$]")

        # main title
        plt.suptitle(f"Glauber Dynamics w/ {self.sweeps} Sweeps per {np.abs(beta_vals[-1] - beta_vals[0])/len(beta_vals):.2f} "
                    + r"$\mathbf{k_{b}T}$"
                    + f" ({self.warmup_sweeps} Warmup Sweeps)",
                    fontweight = 'bold'
                    )
        
        # critical temperature
        plot_crit_T = lambda statistic, ax, crit_T: ax.vlines(
            x = crit_T,
            ymax=np.max(statistic),
            ymin=np.min(statistic),
            linestyle='--',
            zorder=1,
            label = r"$T_{c}$ =" + f" {crit_T:.2f}" + f" ± {beta_vals[1] - beta_vals[0]:.2f}",
            color = "tab:cyan"
        )
        crit_T1 = beta_vals[np.argmax(heatc)]
        plot_crit_T(heatc, ax2, crit_T1)
        ax2.legend()
        crit_T2 = beta_vals[np.argmax(suscept)]
        plot_crit_T(suscept, ax4, crit_T2)
        ax4.legend()

        # save figure
        plt.tight_layout()
        if devnull != self.save_dir:
            fmt = "png"
            fig_path = self.fig_dir / f"glauber.{fmt}" # type: ignore
            plt.savefig(fig_path, format=fmt)

        plt.show()
        
    def plot_kawasaki(self, beta_vals: np.ndarray, mean_e: np.ndarray, mean_m: np.ndarray|None, heatc: np.ndarray, suscept: np.ndarray|None, emean_e: np.ndarray, emean_m: np.ndarray|None, eheatc: np.ndarray, esuscept: np.ndarray|None) -> None:
        """
        Preset plot configuartion for the Kawasaki simulation. Argument signature matches the output signature of self.beta_sim.

        mean_m, suscept, emean_m and esuscept values are not meaningful in the Kawasaki simulation so are not plotted. They are included in the arguments
        to keep the function signature the same as self.plot_glauber.

        Args:
            beta_vals (numpy.ndarray): beta values 
            mean_e (numpy.ndarray): mean energy values 
            mean_m (numpy.ndarray): not used; can be None but must explicitly be passed as None if so
            heatc (numpy.ndarray): heat capacity values
            suscept (numpy.ndarray): susceptibility values; can be None but must explicitly be passed as None if so
            emean_e (numpy.ndarray): error on mean energy values
            emean_m (numpy.ndarray): error on mean magnetisation values; can be None but must explicitly be passed as None if so
            eheatc (numpy.ndarray): error on heat capacity values
            esuscept (numpy.ndarray): error on susceptibility values; can be None but must explicitly be passed as None if so
        """
        # plot setup
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        plot = lambda ax, x, y, err: ax.errorbar(x,
                    y,
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
        xlabel = r"Thermal Energy $k_{B}T$, [$\text{Joules per J}$]"

        # mean energy
        plot(ax1, beta_vals, mean_e, emean_e)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"Average Energy $\langle E \rangle$, [$J$]")

        # heat capacity
        plot(ax2, beta_vals, heatc, eheatc)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"Heat Capacity per Spin $C_{v}$, [$k_{B}$]")

        # main title 
        plt.suptitle(f"Kawasaki Dynamics w/ {self.sweeps} Sweeps per {np.abs(beta_vals[-1] - beta_vals[0])/len(beta_vals):.2f} "
                    + r"$\mathbf{k_{b}T}$"
                    + f" ({self.warmup_sweeps} Warmup Sweeps)",
                    fontweight = 'bold'
                    )
        
        # critical temperature
        plot_crit_T = lambda statistic, ax, crit_T: ax.vlines(
            x = crit_T,
            ymax=np.max(statistic),
            ymin=np.min(statistic),
            linestyle='--',
            zorder=1,
            label = r"$T_{c}$ =" + f" {crit_T:.2f}" + f" ± {beta_vals[1] - beta_vals[0]:.2f}",
            color = "tab:cyan"
        )
        crit_T1 = beta_vals[np.argmax(heatc)]
        plot_crit_T(heatc, ax2, crit_T1)
        ax2.legend()

        # save figure
        plt.tight_layout()
        if devnull != self.save_dir:
            fmt = "png"
            fig_path = self.fig_dir / f"kawasaki.{fmt}" # type: ignore
            plt.savefig(fig_path, format=fmt)

        # show
        plt.show()
    
    def save_nparray(self, data: list|tuple, data_names: list|tuple) -> None:
        """
        Saves provided data in the ./runs/{timestamp}/data directory with names in data_names.

        Can only be called if self.data_dir is not devnull; otherwise an error will be thrown.

        Args:
            data (tuple(numpy.ndarray)): tuple containing numpy arrays with data to be saved
            data_names (tuple(numpy.ndarray)): file names for the above data
        """

        # use numpy binary extension for compatibility
        file_extension = "npy"

        # assumes self.data_dir is not devnull
        for i, d in enumerate(data):
            path = self.data_dir / f"{data_names[i]}.{file_extension}" # type: ignore
            np.save(path, d)