import numpy as np
import numpy.typing as npt
from time import time_ns, time, strftime, gmtime
import matplotlib.pyplot as plt
from typing import Callable, Any


class Sqarray():
    """
    Numpy array with 2 dimensions of equal length. Internal np.random.Generator for random processes.
    """
    def __init__(self, sqlen: int, cell_value: float | tuple[float, ...], rng: np.random.Generator = np.random.default_rng(time_ns())):

        self.sqlen = sqlen
        self.rng = rng 
        self.array: npt.NDArray[np.float64] = self.rng.choice(a=cell_value, size=(sqlen, sqlen))
        self.shape = self.array.shape
        self.size = self.array.size

    def __getitem__(self, ind: tuple[int, int]) -> float:
        return self.array[ind]

    def __setitem__(self, ind: tuple[int, int], value: float):
        self.array[ind] = value

    @classmethod
    def from_array(cls, array: npt.NDArray[np.float64], rng: np.random.Generator = np.random.default_rng(time_ns())) -> "Sqarray":
        """
        Alternate Sqarray constructor. Numpy array must be square; checked at runtime.
        """
        shape = array.shape
        ndim = array.ndim
        assert ((shape[0] == shape[1]) & (ndim == 2)), "Array must be square"

        obj = cls.__new__(cls)
        obj.array = array.copy()
        obj.sqlen = shape[0]
        obj.rng = rng 

        return obj

    def copy(self) -> "Sqarray":
        """
        Copies rng.random.Generator and array contents.
        """
        return Sqarray.from_array(self.array, rng = self.rng)

    def random_cell_ind(self) -> tuple[int, int]:

        row, col = self.rng.integers(low = 0, high = self.shape[0], size = (2,)) 

        return row, col

    def neighbour_sum(self, row, col) -> float:
        """
        Sums values of nearest neighbours.
        Nearest neighbours being left, right, above or below.
        """
        neighbour_sum = (self[(row + 1) % self.sqlen, col] 
                              + self[(row - 1) % self.sqlen, col]
                              + self[row, (col + 1) % self.sqlen]
                              + self[row, (col - 1) % self.sqlen])

        return float(neighbour_sum)

    def adjacent_cells(self, row1, col1, row2, col2) -> bool:
        """
        Check if two cells are adjacent. 
        Adjacent meaning left, right, above or below.
        """
        row_separation = np.abs(row1 - row2)
        col_separation = np.abs(col1 - col2)
        min_row_separation = min(row_separation, self.sqlen - row_separation)
        min_col_separation = min(col_separation, self.sqlen - col_separation)

        if (min_row_separation, min_col_separation) in ((0, 1), (1, 0)):
            return True
        else:
            return False
        
        
class SquareIsingLattice():
    """
    Square arrays relevant for an Ising model simulation.
    """
    def __init__(self, sqlen: int):
        # Formula for energy: E_i = S_i * sum_j(S_j * (J_ij + h_i)) where units fix J = 1 and no external field
        # Source: I hope so
        self.spins = Sqarray(sqlen, cell_value=(-1, 1))
        self.energy = self.energy_sqarray_from(self.spins)

        self.total_energy = Observables.total_energy_from(self.energy)
        self.total_magnetisation = Observables.total_magnetisation_from(self.spins)

    def update_cell_energy_inplace(self, row: int, col: int):

        spins = self.spins
        neighbour_sum = spins.neighbour_sum(row, col)
        self.energy[row, col] = -spins[row, col] * neighbour_sum

    def update_neighbour_cell_energy_inplace(self, row: int, col: int):

        for i in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbour_row, neighbour_col = (row + i[0]) % self.spins.sqlen, (col + i[1]) % self.spins.sqlen
            self.update_cell_energy_inplace(neighbour_row, neighbour_col)

    def update_neighbour_cell_energy_inplace_inclusive(self, row: int, col: int):
        
        for i in ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)):
            neighbour_row, nieghbour_col = (row + i[0]) % self.spins.sqlen, (col + i[1]) % self.spins.sqlen
            self.update_cell_energy_inplace(neighbour_row, nieghbour_col)

    @staticmethod
    def energy_sqarray_from(spins: Sqarray) -> Sqarray:

        temp = spins.copy()
        energy = temp.copy()

        it = np.nditer(temp.array, flags=['multi_index'])
        for _ in it:
            row, col = it.multi_index
            neighbour_sum = temp.neighbour_sum(row, col)
            energy[row, col] = -temp[row, col] * neighbour_sum

        return energy


class MarkovChain():
    """
    Markov-Chain logic with Metropolis update rules.
    """
    @staticmethod
    def metropolis_check(energy_delta: float, beta: float, rng: np.random.Generator) -> bool:

        if (energy_delta < 0 or np.exp(-energy_delta/beta) > rng.random()): 
            return True
        else: 
            return False

    @staticmethod
    def update_glauber_inplace(ising_lattice: SquareIsingLattice, beta: float):
        spins = ising_lattice.spins; energy = ising_lattice.energy
        rng = spins.rng
        row, col = spins.random_cell_ind()

        # glauber energy trick
        energy_delta = float(-2.*energy[row, col])

        if MarkovChain.metropolis_check(energy_delta, beta, rng):
            spins[row, col] *= -1
            ising_lattice.total_magnetisation += float(2*spins[row, col])
            ising_lattice.total_energy += energy_delta
            ising_lattice.update_neighbour_cell_energy_inplace_inclusive(row, col)

    @staticmethod
    def update_kawasaki_inplace(ising_lattice: SquareIsingLattice, beta: float):
        # Magnetisation constant for Kawasaki
        spins = ising_lattice.spins
        rng = spins.rng
        row1, col1 = spins.random_cell_ind(); row2, col2 = spins.random_cell_ind()

        if (row1, col1) == (row2, col2):
            return

        if spins[row1, col1] == spins[row2, col2]:
            return

        energy_delta = (spins[row1, col1] - spins[row2, col2]) * (spins.neighbour_sum(row1, col1) - spins.neighbour_sum(row2, col2))
        adjacent = spins.adjacent_cells(row1, col1, row2, col2)
        if adjacent: energy_delta += 4. # correction for overlapping neighbour sums
        
        if MarkovChain.metropolis_check(energy_delta, beta, rng):
            spins[row1, col1], spins[row2, col2] = spins[row2, col2], spins[row1, col1]
            ising_lattice.update_neighbour_cell_energy_inplace_inclusive(row1, col1)
            ising_lattice.update_neighbour_cell_energy_inplace_inclusive(row2, col2)
            ising_lattice.total_energy += energy_delta


class Observables():
    """
    Container for quantities of interest in the Ising model.
    """
    def __init__(self, size: int):
        self.size = size
        self.mean_energy = np.zeros(shape=(size,))
        self.mean_magnetisation = np.zeros(shape=(size,))
        self.susceptibility = np.zeros(shape=(size,))
        self.heat_capacity = np.zeros(shape=(size,))
    
    @staticmethod
    def total_energy_from(energy: Sqarray) -> float:
        return 0.5*np.sum(energy.array)
    
    @staticmethod
    def total_magnetisation_from(magnetisation: Sqarray) -> float:
        return np.sum(magnetisation.array)

    @staticmethod
    def susceptibility_from(magnetisation: npt.NDArray[np.float64], size: int, beta: float) -> float:
        return float(np.var(magnetisation)/(beta*size))
    
    @staticmethod
    def heat_capacity_from(energy: npt.NDArray[np.float64], size: int, beta: float) -> float:
        return float(np.var(energy)/(np.square(beta)*size))

    @staticmethod
    def bootstrap(values: npt.NDArray[np.float64], resamples: int, function: Callable[..., float], args: tuple[Any, ...] = (), rng: np.random.Generator = np.random.default_rng(time_ns())) -> float :

        bs_samples = rng.choice(values, size = (resamples, values.size), replace = True)
        bs_evaluation = [function(bs_samples[i], *args) for i in range(resamples)] 

        return float(np.std(bs_evaluation))

    @staticmethod
    def jackknife(values: npt.NDArray[np.float64], function: Callable[..., float], args: tuple[Any, ...] = (), rng: np.random.Generator = np.random.default_rng(time_ns())) -> float :

        jk_samples = [np.delete(values, i) for i in range(values.size)]
        
        actual_evaluation = function(values, *args)
        jk_evaluation = np.array([function(jk_samples[i], *args) for i in range(values.size)])

        return float(np.sqrt(np.sum(np.square(jk_evaluation - actual_evaluation))))

    def update_all_values(self, energy: npt.NDArray[np.float64] | None, magnetisation: npt.NDArray[np.float64] | None, size: int, beta: float, index: int):
        if (energy is not None): 
            self.mean_energy[index] = np.mean(energy)
            self.heat_capacity[index] = self.heat_capacity_from(energy, size, beta)
        if (magnetisation is not None): 
            self.mean_magnetisation[index] = np.mean(magnetisation)
            self.susceptibility[index] = self.susceptibility_from(magnetisation, size, beta)

    def update_all_error_values_bootstrap(self, energy: npt.NDArray[np.float64] | None, magnetisation: npt.NDArray[np.float64] | None, size: int, beta: float, index: int, resamples: int = 1000, rng: np.random.Generator = np.random.default_rng(time_ns())):
        if (energy is not None): 
                self.mean_energy[index] = self.bootstrap(energy, resamples, function = np.mean, rng = rng)
                self.heat_capacity[index] = self.bootstrap(energy, resamples, function = self.heat_capacity_from, args = (size, beta), rng = rng)  
        if (magnetisation is not None): 
                self.mean_magnetisation[index] = self.bootstrap(magnetisation, resamples, function = np.mean, rng =  rng)
                self.susceptibility[index] = self.bootstrap(magnetisation, resamples, function = self.susceptibility_from, args = (size, beta), rng = rng)

    def update_all_error_values_jackknife(self, energy: npt.NDArray[np.float64] | None, magnetisation: npt.NDArray[np.float64] | None, size: int, beta: float, index: int, rng: np.random.Generator = np.random.default_rng(time_ns())):
        if (energy is not None): 
                self.mean_energy[index] = self.jackknife(energy, function = np.mean, rng = rng)
                self.heat_capacity[index] = self.jackknife(energy, function = self.heat_capacity_from, args = (size, beta), rng = rng)  
        if (magnetisation is not None): 
                self.mean_magnetisation[index] = self.jackknife(magnetisation, function = np.mean, rng =  rng)
                self.susceptibility[index] = self.jackknife(magnetisation, function = self.susceptibility_from, args = (size, beta), rng = rng)


class Simulation():
    """
    Pre-made Ising simulation logic.
    """
    def __init__(self, sweeps: int, correlation_gap: int, ising_lattice: SquareIsingLattice, warmup_sweeps: int = 100):

        self.sweeps = sweeps
        self.warmup_sweeps = warmup_sweeps
        self.correlation_gap = correlation_gap
        self.ising_lattice = ising_lattice

        self.measurements = sweeps//correlation_gap

    @staticmethod
    def sweep_glauber_inplace_cgap(ising_lattice: SquareIsingLattice, correlation_gap: int, beta: float):

        sweep_len = ising_lattice.spins.size
        for _ in range(0, sweep_len*correlation_gap):
            MarkovChain.update_glauber_inplace(ising_lattice, beta)

    @staticmethod
    def sweep_kawasaki_inplace_cgap(ising_lattice: SquareIsingLattice, correlation_gap: int, beta: float):

        sweep_len = ising_lattice.spins.size
        for _ in range(0, sweep_len*correlation_gap):
            MarkovChain.update_kawasaki_inplace(ising_lattice, beta)

    def compute_energy_magnetisation_glauber(self, beta: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        For loop of sweeps; measurement every [correlation_gap] sweeps.
        """
        ising_lattice = self.ising_lattice
        energy_measurements = np.zeros(shape=(self.measurements,))
        magnetisation_measurements = np.zeros_like(energy_measurements)
        start_time = time()

        for i in range(0, self.measurements):
            self.sweep_glauber_inplace_cgap(ising_lattice, self.correlation_gap, beta)
            energy_measurements[i] = ising_lattice.total_energy
            magnetisation_measurements[i] = ising_lattice.total_magnetisation

        elapsed_time = strftime("%H hours %M minutes %S seconds", gmtime(time() - start_time))
        print(f"{self.sweeps} sweeps @ β = {beta:.2f} in {elapsed_time}")

        return energy_measurements, magnetisation_measurements

    def compute_energy_kawasaki(self, beta: float) -> npt.NDArray[np.float64]:

        ising_lattice = self.ising_lattice
        energy_measurements = np.zeros(shape=(self.measurements,))
        start_time = time()

        for i in range(0, self.measurements):
            self.sweep_kawasaki_inplace_cgap(ising_lattice, self.correlation_gap, beta)
            energy_measurements[i] = ising_lattice.total_energy

        elapsed_time = strftime("%H hours %M minutes %S seconds", gmtime(time() - start_time))
        print(f"{self.sweeps} sweeps @ β = {beta:.2f} in {elapsed_time}")

        return energy_measurements

    def compute_observables_and_plot_glauber(self, min_beta: float, max_beta: float, beta_step: float):
        """
        For loop of sweeps; measurement every [correlation_gap] sweeps.
        """
        beta_values = np.arange(start=max_beta, stop=min_beta, step=-beta_step, dtype=float)
        observables = Observables(size = beta_values.size)
        errors = Observables(size = beta_values.size)

        for _ in range(0, self.warmup_sweeps//self.correlation_gap): # warmup
            self.sweep_glauber_inplace_cgap(self.ising_lattice, self.correlation_gap, beta_values[0])

        for i, beta in enumerate(beta_values):
            energy, magnetisation = self.compute_energy_magnetisation_glauber(beta)
            observables.update_all_values(energy, magnetisation, self.ising_lattice.spins.size, beta, i)
            errors.update_all_error_values_bootstrap(energy, magnetisation, self.ising_lattice.spins.size, beta, i)

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
        plot(ax1, beta_values, observables.mean_energy, errors.mean_energy)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"Average Energy $\langle E \rangle$, [$J$]")

        # heat capacity
        plot(ax2, beta_values, observables.heat_capacity, errors.heat_capacity)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"Heat Capacity per Spin $C_{v}$, [$k_{B}$]")

        # mean magnetisation
        plot(ax3, beta_values, observables.mean_magnetisation, errors.mean_magnetisation)
        ax3.set_xlabel(xlabel)
        ax3.set_ylabel(r"Average Magnetisation $\langle M \rangle$, [unitless]")

        # susceptibility
        plot(ax4, beta_values, observables.susceptibility, errors.susceptibility)
        ax4.set_xlabel(xlabel)
        ax4.set_ylabel(r"Magnetic Susceptibility $\chi$, [$\text{unitless}$]")

        # main title
        plt.suptitle(f"Glauber Dynamics w/ {self.sweeps} Sweeps per {np.abs(beta_values[-1] - beta_values[0])/len(beta_values):.2f} "
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
            label = r"$T_{c}$ =" + f" {crit_T:.2f}" + f" ± {np.abs(beta_values[1] - beta_values[0]):.2f}",
            color = "tab:cyan"
        )
        crit_T1 = beta_values[np.argmax(observables.heat_capacity)]
        plot_crit_T(observables.heat_capacity, ax2, crit_T1)
        ax2.legend()
        crit_T2 = beta_values[np.argmax(observables.susceptibility)]
        plot_crit_T(observables.susceptibility, ax4, crit_T2)
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def compute_observables_and_plot_kawasaki(self, min_beta: float, max_beta: float, beta_step: float):

        beta_values = np.arange(start=max_beta, stop=min_beta, step=-beta_step, dtype=float)
        observables = Observables(size = beta_values.size)
        errors = Observables(size = beta_values.size)

        for _ in range(0, self.warmup_sweeps//self.correlation_gap): # warmup
            self.sweep_kawasaki_inplace_cgap(self.ising_lattice, self.correlation_gap, beta_values[0])

        for i, beta in enumerate(beta_values):
            energy = self.compute_energy_kawasaki(beta)
            observables.update_all_values(energy, None, self.ising_lattice.spins.size, beta, i)
            errors.update_all_error_values_jackknife(energy, None, self.ising_lattice.spins.size, beta, i)

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
        plot(ax1, beta_values, observables.mean_energy, errors.mean_energy)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(r"Average Energy $\langle E \rangle$, [$J$]")

        # heat capacity
        plot(ax2, beta_values, observables.heat_capacity, errors.heat_capacity)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(r"Heat Capacity per Spin $C_{v}$, [$k_{B}$]")

        # main title 
        plt.suptitle(f"Kawasaki Dynamics w/ {self.sweeps} Sweeps per {np.abs(beta_values[-1] - beta_values[0])/len(beta_values):.2f} "
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
            label = r"$T_{c}$ =" + f" {crit_T:.2f}" + f" ± {np.abs(beta_values[1] - beta_values[0]):.2f}",
            color = "tab:cyan"
        )
        crit_T1 = beta_values[np.argmax(observables.heat_capacity)]
        plot_crit_T(observables.heat_capacity, ax2, crit_T1)
        ax2.legend()
        plt.tight_layout()
        plt.show()
            