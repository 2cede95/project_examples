import numpy as np 
import matplotlib.pyplot as plt
from os import devnull
from pathlib import Path

class fluid_equations:
    """
    Contains fluid equations which are solved with centered difference schemes in
    the integrator_tools library.
    """
    def __init__(self, length, t_step, x_step, iv, noise, tol, bv, save_dir, solver_name):
        """
        Variables defined here apply globally to all simulations unless a specific value is required.

        Args:
            length (int): length of square side to be used in simulations
            t_step (float): time discretisation factor
            x_step (float): space discretisation factor in all spacial dimensions
            iv (float): initial value taken in the array for simulation
            noise (float): noise factor to be added to iv for the simulation array
            tol (float): difference between adjacent timesteps for the system to be considered converged
            bv (tup(float)): boundary values in the form (axis 0 start axis 0 end, axis 1 start, axis 1 end, ....., axis x start, axis x end)
                must match the dimensions of the problem, i.e. for a 3d spacial grid bv must be a length 6 tuple
            save_dir (pathlib.Path): parent directory for a specific simulation run
            solver_name (str): name of the solver used in the 'point_charge' and 'vertical_wire' preset simulations 
        """

        # simulation variables
        self.dx = x_step
        self.dt = t_step
        self.tol = tol
        self.length = length
        self.integrator = integrator_tools()

        # boundary values
        self.bv = bv

        # initial values
        self.iv = iv
        self.noise = noise

        # os independent path handling for saving data
        self.save_dir = save_dir

        if save_dir != devnull:
            self.data_dir = save_dir / "data"
            self.data_dir.mkdir(exist_ok=True)
            self.fig_dir = save_dir / "figures"
            self.fig_dir.mkdir(exist_ok=True)
        else:
            self.data_dir = self.fig_dir = devnull

        # name of solver (used in plots)
        self.solver_name = solver_name

        return None
    
    def init_array(self, length, dim, noise=0):
        """
        Initialises a dim dimensional square numpy array with optional uniform random noise.

        Args:
            length (int): length of square array
            dim (int): number of dimensions the array should have
            noise (float): noise factor to be added to the array

        Returns:
            Initialised array (numpy.ndarray)
        """
        # initialise
        shape = np.empty(shape=(dim), dtype=int)

        # loop over dimensions to create tup like (length, lenth, ... , length)
        for i in range(0, dim):
            shape[i] = int(length)

        # initialize numpy array full of iv and add numpy uniform noise
        arr = self.iv*np.ones(shape=shape) + noise*np.random.uniform(low=0, high=1, size=shape)

        return arr
      
    def apply_boundary(self, vals, boundary_vals):
        """
        Apply boundary conditions to a given array.

        Order of application: axis 0 top, axis 0 bottom, axis 1 top, .... , axis n bottom

        Args:
            vals (numpy.ndarray): numpy array to be modified
            boundary_vals (tuple(float)): boundary values in the form (axis 0 start axis 0 end, axis 1 start, axis 1 end, ....., axis x start, axis x end)
                must match the dimensions of the problem, i.e. for a 3d spacial grid bv must be a length 6 tuple

        Returns:
            Modified array (numpy.ndarray)
        """

        # copy array
        dim = vals.ndim
        applied = vals.copy()

        # loop over dimensions
        for i in range(0, dim):

            # create mask like [0, :, :, ...], [-1, :, :, ...], [:, 0, :, :, ....], ...., [..., :, :, -1]
            # and modify array
            mask = [slice(None)]*dim
            for j, k in enumerate((0, -1)):
                mask[i] = k
                applied[tuple(mask)] = boundary_vals[(2*i + j)]

        return applied
   
    def chemical_potential(self, vals, dx):
        """
        Computes the discretised chemical potential given a grid of compositional order parameters.
        Works in N dimensions.
       
         Args:
            vals (numpy.ndarray): grid of order parameters
            dx (float): space discretisation for centered difference scheme
            
        Returns:
            Discretised chemical potential grid (numpy.ndarray)
        """
        mu = (-1)*vals + np.pow(vals, 3) - self.integrator.lap(vals, dx)
        return mu
    
    def free_energy(self, vals, dx):
        """
        Computes the discretised free energy density for a grid of compositional order parameters and sums the results
        for the total free energy of the state. Works in N dimensions.

        Args:
            vals (numpy.ndarray): grid of order parameters
            dx (float): space discretisation for centered difference scheme

        Returns:
            Total free energy of state (float)
        
        """
        g = np.sum(np.square(self.integrator.grad(vals, dx)), axis=0)
        fe_density = (-1)*np.square(vals)/2 + np.pow(vals, 4)/4 + g/2
        fe = np.sum(fe_density)
        return fe
    
    def electric_field(self, vals, dx):
        """
        Computes the discretised electric field components given a grid of electrostatic potentials.
        Works in N dimensions. Output shape will be (dimension, length, length, ..., length)

        i.e. e_0, e_1, e_2 = electric_field(...) is a tuple of the components E_0, E_1, E_2

        Args:
            vals (numpy.ndarray): grid of electrostatic potentials
            dx (float): space discretisation for centered difference scheme

        Returns:
            Electric field components (numpy.ndarray)

        """
        e = (-1)*self.integrator.grad(vals, dx)
        return e
    
    def magnetic_field(self, vals, dx):
        """
        Computes the discretised magnetic field given a grid of magnetic vector potential values. Curl is defined strictly
        in 3D so the input grid must have shape (3, length, length, .... ,length). 

        i.e. b_0, b_1, b_2 = magnetic_field(...) is a tuple of the components E_0, E_1, E_2
        
        Args:
            vals (numpy.ndarray): grid of vector potential values
            dx (float): space discretisation for centered difference scheme

        Returns:
            Magnetic field components (numpy.ndarray)
        """
        b = self.integrator.curl(vals, dx)
        return b
           
    def point_charge(self, N_max, solver, w=1, plot=True, savedata=True):
        """
        Preset simulation which solves Poisson's equation for the discretised electrostatic potential of a point charge with specified initial conditions. Returns the number of iterations to converge.
        Four plots are produced if the plot argument is true:
            - 2D contour plot of plane which cuts the point charge in half (i-j plane);
            - 3D quiver plot using matplotlib library of the discretised electric field solution;
            - 1D plot of discretised electrostatic potential strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised electric field strength against distance from the centre of the point charge on the same line as above.

        Data is saved in the ./runs/{timestamp}/data as .npy binaries. 
        Plots saved in ./runs/{timestamp}/figures as .png images with a dpi of 150.

        Args:
            N_max (int): maximum number of iterations to attempt for convergence
            solver (function): ODE solver to use; jacobi, gauss_seidel and successive_over_relax solvers are provided in the integaration_tools class
            w (float): the relaxation factor if a successive over relaxation solver is used
            plot (bool): plot after convergence
            savedata (bool): saves electric field and electrostatic potential data as .npy binaries after convergence

        Returns:
            Number of iterations to converge (int)
        """

        # sim variables
        dx = 1
        bv = self.bv
        tol = self.tol
        if solver == self.integrator.successive_over_relax:
            args = (dx, w)
        else:
            args = (dx,)

        # functions
        electric_field = self.electric_field
        apply_boundary = self.apply_boundary

        # initial conditions
        dim = 3
        order_parameter = self.init_array(self.length, dim=dim, noise=self.noise)
        order_parameter = apply_boundary(order_parameter, bv)
        
        charge_density = np.zeros_like(order_parameter)
        charge = -100
        mid_point = charge_density.shape[0]//2
        pos = (mid_point, mid_point, mid_point)
        charge_density[pos] = charge # point charge
   

        # compute loop
        iterations = 0
        for i in range(1, N_max):
            # copy + update order parameter + re-apply boundary values
            order_parameter_old = order_parameter.copy()
            order_parameter = apply_boundary(solver(order_parameter, charge_density, *args), bv)

            # convergence check
            if np.max(np.abs(order_parameter - order_parameter_old)) < tol:
                iterations = i + 1 
                print(f"converged in {iterations} iterations.")
                break

            if i == N_max-1:
                iterations = i + 1 
                print(f"did not converge. {iterations} iterations elapsed.")
                break
        
        # compute electric field
        e = electric_field(order_parameter, dx)

        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (order_parameter, e)
            name = ("electrostatic_potential", "electric_field")
            self.save_nparray(arr, name)

        # plot
        if True == plot:
            titles = ("Electrostatic Potential Slice",
                    r"Electric Field, $\vec{E} = -\vec{\nabla}\varphi$",
                    "Magnitude of Electrostatic Potential vs Distance",
                    "Magnitude of Electric field vs Distance")
            xlabels = (None,
                    "i",
                    r"Displacement From Centre, $r_{ijk}$",
                    r"Displacement From Centre, $r_{ijk}$")
            ylabels = (None,
                    "j",
                    r"Electrostatic Potential Magnitude, $\|\varphi\|$",
                    r"Electric Field Magnitude, $\|\vec{E}\|$")
            zlabels = (r"Electrostatic potential, $\varphi$",
                    "k",
                    None,
                    None)
            filename = "point_charge"
            main_title = r"$\vec{\nabla}^{2}\varphi = - \frac{\rho}{\epsilon}$ w/ Point Charge Distribution solved with " + f"{self.solver_name} scheme (tol = {tol})"

            self.vector_field_plot(order_parameter, e, titles, xlabels, ylabels, zlabels, filename, main_title)

        return iterations
    
    def vertical_wire(self, N, solver, w=1, plot=True, savedata=True):
        """
        Preset simulation which solves Maxwell's equations for the magnetic vector potential of a line of current with specified intiial and boundary conditions. Returns the number of iterations to converge.
        Four plots are produced if the plot argument is true:
            - 2D contour plot of plane which cuts the current line in half (i-j plane);
            - 3D quiver plot using matplotlib library of the magnetic field solution;
            - 1D plot of the k componenet of the discretised magnetic vector potential strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised electric field strength against distance from the centre of the point charge on the same line as above.

        Data is saved in the ./runs/{timestamp}/data as .npy binaries. 
        Plots saved in ./runs/{timestamp}/figures as .png images with a dpi of 150.

        Args:
            N_max (int): maximum number of iterations to attempt for convergence
            solver (function): solver to use; jacobi, gauss_seidel and successive_over_relax solvers are provided in the integaration_tools class
            w (float): the relaxation factor if a successive over relaxation solver is used
            plot (bool): plot after convergence
            savedata (bool): saves magnetic field and k component of vector potential as .npy binaries after convergence

        Returns:
            Number of iterations to converge (int)
        """
        # simulation variables
        dx = 1
        bv = self.bv
        tol = self.tol
        if solver == self.integrator.successive_over_relax:
            args = (dx, w)
        else:
            args = (dx,)

        # functions
        magnetic_field = self.magnetic_field
        apply_boundary = self.apply_boundary

        # initial conditions, order parameter is constant at 0 in i and j direction, solving in k direction only
        dim = 3
        order_parameter_k = self.init_array(self.length, dim=dim, noise=self.noise)
        order_parameter_k = apply_boundary(order_parameter_k, bv)

        current_density = np.zeros_like(order_parameter_k)
        current = -10
        mid_point = current_density.shape[0]//2
        current_density[mid_point, mid_point, :] = current
        
        # compute loop
        iterations = 0
        for i in range(1, N):
            # copy + update order parameter + re-apply boundary values
            order_parameter_k_old = order_parameter_k.copy()
            order_parameter_k = apply_boundary(solver(order_parameter_k, current_density, *args), bv)

            # convergence check
            if np.max(np.abs(order_parameter_k - order_parameter_k_old)) < tol:
                iterations = i + 1 
                print(f"converged in {iterations} iterations.")
                break

            if i == N-1:
                iterations = i + 1 
                print(f"did not converge. {iterations} iterations elapsed.")
                break
        
        # compute magnetic field; need order parameter like (3, length, length, length)
        order_parameter = np.array([np.zeros_like(order_parameter_k), np.zeros_like(order_parameter_k), order_parameter_k])
        b = magnetic_field(order_parameter, dx)

        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (order_parameter, b)
            name = ("magnetic_vector_potential_k", "magnetic_field")
            self.save_nparray(arr, name)

        # plot
        if True == plot:    
            titles = ("Vector Potential Slice",
                    r"Magnetic Field, $\vec{B} = \vec{\nabla}\times\vec{A}$",
                    "k-component of Vector Potential vs Distance",
                    "Magnitude of Magnetic Field vs Distance")
            xlabels = (None,
                    "i",
                    r"Displacement From Centre, $r_{ijk}$",
                    r"Displacement From Centre, $r_{ijk}$")
            ylabels = (None,
                    "j",
                    r"k-component of Vector Potential, $\vec{A}\cdot\vec{e_{k}}$",
                    r"Magnetic Field Magnitude, $\|\vec{B}\|$")
            zlabels = (r"k-componenet of Vector Potential, $\vec{A}\cdot\vec{e_{k}}$",
                    "k",
                    None,
                    None)
            filename = "vertical_wire"
            main_title = r"$\vec{\nabla}^{2}A_{k} = - \mu_{0}J_{k}$ w/ Vertical Current Line solved with " + f"{self.solver_name} scheme (tol = {tol})"

            self.vector_field_plot(order_parameter[2], b, titles, xlabels, ylabels, zlabels, filename, main_title)

        return iterations

    def point_charge_visual(self, N_max, solver, w=1, plot=True, savedata=True):
        """
        Preset simulation which solves Poisson's equation for the discretised electrostatic potential of a point charge with specified initial conditions. Returns the number of iterations to converge.
        Four plots are produced if the plot argument is true:
            - 2D contour plot of plane which cuts the point charge in half (i-j plane);
            - 3D quiver plot using matplotlib library of the discretised electric field solution;
            - 1D plot of discretised electrostatic potential strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised electric field strength against distance from the centre of the point charge on the same line as above.

        Data is saved in the ./runs/{timestamp}/data as .npy binaries. 
        Plots saved in ./runs/{timestamp}/figures as .png images with a dpi of 150.

        A live heatmap plot displaying a 2D slice through the centre of the point charge is displayed as the equation is solved. 

        Args:
            N_max (int): maximum number of iterations to attempt for convergence
            solver (function): ODE solver to use; jacobi, gauss_seidel and successive_over_relax solvers are provided in the integaration_tools class
            w (float): the relaxation factor if a successive over relaxation solver is used
            plot (bool): plot after convergence
            savedata (bool): saves electric field and electrostatic potential data as .npy binaries after convergence

        Returns:
            Number of iterations to converge (int)
        """

        # sim variables
        dx = 1
        bv = self.bv
        tol = self.tol
        if solver == self.integrator.successive_over_relax:
            args = (dx, w)
        else:
            args = (dx,)

        # functions
        electric_field = self.electric_field
        apply_boundary = self.apply_boundary

        # initial conditions
        dim = 3
        order_parameter = self.init_array(self.length, dim=dim, noise=self.noise)
        order_parameter = apply_boundary(order_parameter, bv)
        
        charge_density = np.zeros_like(order_parameter)
        charge = -100
        mid_point = charge_density.shape[0]//2
        pos = (mid_point, mid_point, mid_point)
        charge_density[pos] = charge # point charge
   
        #interactive plot
        fig, ax = plt.subplots(1, 1)
        iplot_slice = slice(None), slice(None), mid_point 
        hplot = ax.imshow(order_parameter[iplot_slice], origin="lower", cmap="RdBu")
        ax.set_title(r"Electrostatic Potential $\varphi_{ijk}$ @ k midpoint")
        ax.set_xlabel("i", fontweight="bold")
        ax.set_ylabel("j", fontweight="bold")
        cbar = plt.colorbar(hplot)
        cbar.set_label(label=r"Electrostatic Potential, $\varphi_{ijk}$", fontweight="bold")
        plt.ion()

        # compute loop
        iterations = 0
        for i in range(1, N_max):
            # copy + update order parameter + re-apply boundary values
            order_parameter_old = order_parameter.copy()
            order_parameter = apply_boundary(solver(order_parameter, charge_density, *args), bv)

            # update interactive plot
            if i%2 == 0:
                op_slice = order_parameter[iplot_slice]
                vmax, vmin = np.max(op_slice), np.min(op_slice) 
                hplot.set_clim(vmin, vmax)
                hplot.set_data(op_slice)
                plt.pause(0.001)

            # convergence check
            if np.max(np.abs(order_parameter - order_parameter_old)) < tol:
                iterations = i + 1 
                print(f"converged in {iterations} iterations.")
                break

            if i == N_max-1:
                iterations = i + 1 
                print(f"did not converge. {iterations} iterations elapsed.")
                break
        
        # compute electric field
        e = electric_field(order_parameter, dx)

        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (order_parameter, e)
            name = ("electrostatic_potential", "electric_field")
            self.save_nparray(arr, name)

        # plot
        plt.ioff()
        if True == plot:
            titles = ("Electrostatic Potential Slice",
                    r"Electric Field, $\vec{E} = -\vec{\nabla}\varphi$",
                    "Magnitude of Electrostatic Potential vs Distance",
                    "Magnitude of Electric field vs Distance")
            xlabels = (None,
                    "i",
                    r"Displacement From Centre, $r_{ijk}$",
                    r"Displacement From Centre, $r_{ijk}$")
            ylabels = (None,
                    "j",
                    r"Electrostatic Potential Magnitude, $\|\varphi\|$",
                    r"Electric Field Magnitude, $\|\vec{E}\|$")
            zlabels = (r"Electrostatic potential, $\varphi$",
                    "k",
                    None,
                    None)
            filename = "point_charge"
            main_title = r"$\vec{\nabla}^{2}\varphi = - \frac{\rho}{\epsilon}$ w/ Point Charge Distribution solved with " + f"{self.solver_name} scheme (tol = {tol})"

            self.vector_field_plot(order_parameter, e, titles, xlabels, ylabels, zlabels, filename, main_title)

        return iterations

    def vertical_wire_visual(self, N, solver, w=1, plot=True, savedata=True):
        """
        Preset simulation which solves Maxwell's equations for the magnetic vector potential of a line of current with specified intiial and boundary conditions. Returns the number of iterations to converge.
        Four plots are produced if the plot argument is true:
            - 2D contour plot of plane which cuts the current line in half (i-j plane);
            - 3D quiver plot using matplotlib library of the magnetic field solution;
            - 1D plot of the k componenet of the discretised magnetic vector potential strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised electric field strength against distance from the centre of the point charge on the same line as above.

        Data is saved in the ./runs/{timestamp}/data as .npy binaries. 
        Plots saved in ./runs/{timestamp}/figures as .png images with a dpi of 150.

        A live heatmap plot displaying a 2D slice through the centre of the current line (i-j plane) is displayed as the equation is solved. 

        Args:
            N_max (int): maximum number of iterations to attempt for convergence
            solver (function): solver to use; jacobi, gauss_seidel and successive_over_relax solvers are provided in the integaration_tools class
            w (float): the relaxation factor if a successive over relaxation solver is used
            plot (bool): plot after convergence
            savedata (bool): saves magnetic field and k component of vector potential as .npy binaries after convergence

        Returns:
            Number of iterations to converge (int)
        """
        # simulation variables
        dx = 1
        bv = self.bv
        tol = self.tol
        if solver == self.integrator.successive_over_relax:
            args = (dx, w)
        else:
            args = (dx,)

        # functions
        magnetic_field = self.magnetic_field
        apply_boundary = self.apply_boundary

        # initial conditions, order parameter is constant at 0 in i and j direction, solving in k direction only
        dim = 3
        order_parameter_k = self.init_array(self.length, dim=dim, noise=self.noise)
        order_parameter_k = apply_boundary(order_parameter_k, bv)

        current_density = np.zeros_like(order_parameter_k)
        current = -10
        mid_point = current_density.shape[0]//2
        current_density[mid_point, mid_point, :] = current
        
        #interactive plot
        fig, ax = plt.subplots(1, 1)
        iplot_slice = slice(None), slice(None), mid_point 
        hplot = ax.imshow(order_parameter_k[iplot_slice], origin="lower", cmap="RdBu")
        ax.set_title(r"Vector Potential $A_{k} Slice$ @ k midpoint")
        ax.set_xlabel("i", fontweight="bold")
        ax.set_ylabel("j", fontweight="bold")
        cbar = plt.colorbar(hplot)
        cbar.set_label(label=r"Vector Potential Slice, $A_{k}$", fontweight="bold")
        plt.ion()

        # compute loop
        iterations = 0
        for i in range(1, N):
            # copy + update order parameter + re-apply boundary values
            order_parameter_k_old = order_parameter_k.copy()
            order_parameter_k = apply_boundary(solver(order_parameter_k, current_density, *args), bv)

            # update interactive plot
            if i%2 == 0:
                op_slice = order_parameter_k[iplot_slice]
                vmax, vmin = np.max(op_slice), np.min(op_slice) 
                hplot.set_clim(vmin, vmax)
                hplot.set_data(op_slice)
                plt.pause(0.001)

            # convergence check
            if np.max(np.abs(order_parameter_k - order_parameter_k_old)) < tol:
                iterations = i + 1 
                print(f"converged in {iterations} iterations.")
                break

            if i == N-1:
                iterations = i + 1 
                print(f"did not converge. {iterations} iterations elapsed.")
                break
        
        # compute magnetic field; need order parameter like (3, length, length, length)
        order_parameter = np.array([np.zeros_like(order_parameter_k), np.zeros_like(order_parameter_k), order_parameter_k])
        b = magnetic_field(order_parameter, dx)

        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (order_parameter, b)
            name = ("magnetic_vector_potential_k", "magnetic_field")
            self.save_nparray(arr, name)

        # plot
        plt.ioff()
        if True == plot:    
            titles = ("Vector Potential Slice",
                    r"Magnetic Field, $\vec{B} = \vec{\nabla}\times\vec{A}$",
                    "k-component of Vector Potential vs Distance",
                    "Magnitude of Magnetic Field vs Distance")
            xlabels = (None,
                    "i",
                    r"Displacement From Centre, $r_{ijk}$",
                    r"Displacement From Centre, $r_{ijk}$")
            ylabels = (None,
                    "j",
                    r"k-component of Vector Potential, $\vec{A}\cdot\vec{e_{k}}$",
                    r"Magnetic Field Magnitude, $\|\vec{B}\|$")
            zlabels = (r"k-componenet of Vector Potential, $\vec{A}\cdot\vec{e_{k}}$",
                    "k",
                    None,
                    None)
            filename = "vertical_wire"
            main_title = r"$\vec{\nabla}^{2}A_{k} = - \mu_{0}J_{k}$ w/ Vertical Current Line solved with " + f"{self.solver_name} scheme (tol = {tol})"

            self.vector_field_plot(order_parameter[2], b, titles, xlabels, ylabels, zlabels, filename, main_title)

        return iterations

    def cahn_hilliard_2d(self, N, plot=True, savedata=True):
        """
        Solve the Cahn Hilliard equation using a forward euler scheme in 2 dimensions.
        If plot is True then a 1D plot of free energy vs iterations is presented.

        The plot is saved at ./runs/{timestamp}/figures as a .png
        Free energy data and iteration no. data are saved in ./runs/{timestamp}/data

        Args:
            N (int): maximum number of iterations to perform
            plot (bool): show a plot of free energy vs iterations
            savedata (bool): saves magnetic field and k component of vector potential as .npy binaries after convergence
        """        

        # simulation parameters
        dx = self.dx
        dt = self.dt

        # functions
        solver = self.integrator.forward_euler
        chemical_potential_func = self.chemical_potential
        free_energy_func = self.free_energy

        # initial conditions
        dim = 2
        order_parameter = self.init_array(self.length, dim, self.noise)
        free_energy_array = np.empty(shape=(N))
        free_energy_array[0] = free_energy_func(order_parameter, dx)

        # compute loop
        for i in range(1, N):
            # compute chemical potential
            chemical_potential = chemical_potential_func(order_parameter, dx)

            # update order parameter
            order_parameter = solver(order_parameter, chemical_potential, dx=dx, dt=dt)

            # update free energy
            free_energy = free_energy_func(order_parameter, dx)
            free_energy_array[i] = free_energy
        
        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (free_energy_array,)
            name = ("free_energy",)
            self.save_nparray(arr, name)

        # plot
        if True == plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(range(N), free_energy_array, c="tab:red")
            ax.set_xlabel("No. Iterations", fontweight="bold")
            ax.set_ylabel("Total Free Energy", fontweight="bold")
            ax.set_title(r"$\phi_{initial} =$ " + f"{self.iv}, " + r"$\delta x =$ " +f"{self.dx}, " + r"$\delta t =$ " + f"{self.dt}")

            # save plot
            if self.save_dir != devnull:
                fmt = "png"
                fig_path = self.fig_dir / f"free_energy.{fmt}"
                plt.savefig(fig_path, format=fmt)

            plt.show()

        return None

    def cahn_hilliard_2d_visual(self, N, plot=True, savedata=True):
        """
        Solve the Cahn Hilliard equation using a forward euler scheme.
        If plot is True then a 1D plot of free energy vs iterations is presented. 

        The plot is saved at ./runs/{timestamp}/figures as a .png
        Free energy data and iteration no. data are saved in ./runs/{timestamp}/data

        A 2D animation of the order parameter evolution is presented during computaton. The plot is updated every 1000 iterations.

        Args:
            N (int): maximum number of iterations to perform
            plot (bool): show a plot of free energy vs iterations
            savedata (bool): saves magnetic field and k component of vector potential as .npy binaries after convergence
        """  
        # simulation parameters
        dx = self.dx
        dt = self.dt

        # functions
        solver = self.integrator.forward_euler
        chemical_potential_func = self.chemical_potential
        free_energy_func = self.free_energy

        #arrays
        dim = 2
        order_parameter = self.init_array(self.length, dim, self.noise)
        free_energy_array = np.empty(shape=(N))
        free_energy_array[0] = free_energy_func(order_parameter, dx)

        #interactive plot
        fig, ax = plt.subplots(1, 1)
        line = ax.imshow(order_parameter, vmin=-1.0, vmax=1.0, origin="lower", cmap="RdBu")
        ax.set_title(r"$\phi_{initial} =$ " + f"{self.iv}, " + r"$\delta x =$ " +f"{dx}, " + r"$\delta t =$ " + f"{dt}")
        ax.set_xlabel("i", fontweight="bold")
        ax.set_ylabel("j", fontweight="bold")
        cbar = plt.colorbar(line)
        cbar.set_label(label=r"Order Parameter, $\phi_{ij}$", fontweight="bold")
        plt.ion()

        #compute loop
        for i in range(1, N):
            # compute chemical potential
            chemical_potential = chemical_potential_func(order_parameter, dx)

            # update order parameter
            order_parameter = solver(order_parameter, chemical_potential, dx=dx, dt=dt)

            # update free energy
            free_energy = free_energy_func(order_parameter, dx)
            free_energy_array[i] = free_energy

            # update plot
            if i%1000 == 0:
                line.set_data(order_parameter)
                plt.pause(0.001)
        
        #save data
        if devnull != self.save_dir and True == savedata:
            arr = (free_energy_array, np.arange(N))
            name = ("free_energy", "iterations")
            self.save_nparray(arr, name)

        #plot
        plt.ioff()
        if True == plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(np.arange(N), free_energy_array, c="tab:red")
            ax.set_xlabel("No. Iterations", fontweight="bold")
            ax.set_ylabel("Total Free Energy", fontweight="bold")
            ax.set_title(r"$\phi_{initial} =$ " + f"{self.iv}, " + r"$\delta x =$ " +f"{self.dx}, " + r"$\delta t =$ " + f"{self.dt}")

            #save plot
            if self.save_dir != devnull:
                fmt = "png"
                fig_path = self.fig_dir / f"free_energy.{fmt}"
                plt.savefig(fig_path, format=fmt)

            #show
            plt.show()

        return None
    
    def determine_optimal_w(self, sim, N_max, w_min, w_max, w_res, plot=True, savedata=True):
        """
        Computes a range of preset simluations, sim, to deteminre the optimal relaxation parameter when sim is solved using the
        successive over relaxation scheme. A 1D plot is presented with the no. of iterations to converge against relaxation parameter w.

        The optimal value is printed to stdout and marked on the plot with a vertical line.

        Relaxation parameter and iteration data are saved at ./runs/{timestamp}/data in .npy binaries.
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Intermediate data for sims runs are NOT saved if the provided point_charge or vertical_wire functions are used.
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        sim must have the function signature (N_max, solver, w=1, plot=True, savedata=True)

        Args:
            sim (function): preset simulation to solve; point_charges and vertical_wire are provided; sim must return the number of iterations to convergence
            N_max (int): maximum number of iterations to attempt for convergence
            w_min (float): starting value of w to define the sample space w_vals
            w_max (float): maximum value of w to define the sample space w_vals
            w_res (int): number of w points to sample
            plot (bool): show a plot of relaxation parameter and number of iterations to converge

        Returns:
            Optimal value of relaxation parameter (float)
        """
        # functions
        solver = self.integrator.successive_over_relax

        # array setup
        w_vals = np.linspace(w_min, w_max, w_res)
        print(f"w values: [{w_vals[0]:.2f}, {w_vals[1]:.2f}, {w_vals[2]:.2f}, ..., {w_vals[-2]:.2f}, {w_vals[-1]:.2f}]")
        iterations = np.empty_like(w_vals)

        # compute loop
        for i, w in enumerate(w_vals):

            # will converge instantly but will not give correct solution
            if 0. == w:
                continue

            print(f"w = {w:.2f} -> ", end="")
            iterations[i] = sim(N_max, solver, w=w, plot=False, savedata=False)

        min_ind = np.argmin(iterations)
        optimal_w = w_vals[min_ind]
        print(f"Optimal w is {optimal_w:.2f}.")

        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (w_vals, iterations)
            name = ("relaxation_parameter", "iterations")
            self.save_nparray(arr, name)

        # plot
        if True == plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(w_vals, iterations, c="tab:red")
            ax.set_xlabel(r"relaxation parameter, $\omega$", fontweight="bold")
            ax.set_ylabel(r"No. of Iterations to Converge", fontweight="bold")
            ax.set_title(r"Optimal $\omega$ Determination for SOR")
            ax.axvline(x=optimal_w, color="tab:blue", label=r"Optimal $\omega =$" + f" {optimal_w:.2f}")
            plt.legend()

            # save plot
            if self.save_dir != devnull:
                fmt = "png"
                fig_path = self.fig_dir / f"optimal_w.{fmt}"
                plt.savefig(fig_path, format=fmt)

            plt.show()
        
        return optimal_w

    def vector_field_plot(self, scalar_field, vector_field, titles, xlabels, ylabels, zlabels, filename, main_title):
        """
        Method which produces the following plots for use with the point_charge and vertical_wire functions:
            - 2D contour plot of plane which cuts the source in half (i-j plane);
            - 3D quiver plot using matplotlib library of the discretised vector field solution;
            - 1D plot of discretised order parameter strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised vector field strength against distance from the centre of the point charge on the same line as above.

        Args:
            scalar_field (numpy.ndarray): scalar order parameter data
            vector_field (numpy.ndarray): vector field data
            title (tuples(str)): tuple containing plotting labels 
            xlabels (tuples(str)): tuple containing x axis labels 
            ylabels (tuples(str)): tuple containing y axis labels 
            zlabels (tuples(str)): tuple containing z axis labels 
            filename (str): file name for the plot
        """


        # constants
        length = self.length
        mid_point = scalar_field.shape[0]//2

        # formatting
        vector_field_0, vector_field_1, vector_field_2 = vector_field
        
        # setup axes
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
        ax1.set_title(titles[0])
        ax1.set_xlabel("i", fontweight="bold")
        ax1.set_ylabel("j", fontweight="bold")

        ax2.remove()
        ax2 = fig.add_subplot(2, 2, 2, projection = '3d') # (rows, cols, position to add)
        ax2.set_title(titles[1])
        ax2.set_xlabel(xlabels[1], fontweight="bold")
        ax2.set_ylabel(ylabels[1], fontweight="bold")
        ax2.set_zlabel(zlabels[1], fontweight="bold")


        ax3.set_title(titles[2])
        ax3.set_xlabel(xlabels[2], fontweight="bold")
        ax3.set_ylabel(ylabels[2], fontweight="bold")
        
        ax4.set_title(titles[3])
        ax4.set_xlabel(xlabels[3], fontweight="bold")
        ax4.set_ylabel(ylabels[3], fontweight="bold")

        # contour plot of potential
        order_parameter_slice = scalar_field[:, :, mid_point]
        x, y = np.meshgrid(
            np.arange(0, length),
            np.arange(0, length),
            indexing='ij'
        )
        c = ax1.contourf(x, y, order_parameter_slice, cmap="RdBu")
        cbar = fig.colorbar(c, ax=ax1)
        cbar.set_label(label=zlabels[0], fontweight="bold")

        # field strength plots
        x, y = np.arange(length), mid_point
        x = x - mid_point
        e_strength = np.linalg.norm(vector_field, axis=0)
        op_strength = scalar_field

        mid_line = mid_point, slice(None), mid_point
        ax3.plot(x, op_strength[mid_line], color="tab:red")
        ax4.plot(x, e_strength[mid_line], color="tab:red", )

        # vector plot
        step = length//10
        x, y, z = np.meshgrid(
        np.arange(0, length, step),
        np.arange(0, length, step),
        np.arange(0, length, step),
        indexing='ij')
        mask = slice(None, None, step), slice(None, None, step), slice(None, None, step)
        q = ax2.quiver(x, y, z, vector_field_0[mask], vector_field_1[mask], vector_field_2[mask], length=step*5, color='tab:red')

        # formatting
        plt.suptitle(main_title, fontweight="bold")
        plt.tight_layout()

        # save plot
        if self.save_dir != devnull:
            fmt = "png"
            fig_path = self.fig_dir / f"{filename}.{fmt}"
            plt.savefig(fig_path, format=fmt)

        plt.show()

        return None
        
    def save_nparray(self, data, data_names):
        """
        Saves provided data in the ./runs/{timestamp}/data directory with names in data_names.

        Can only be called if self.data_dir is not devnull; otherwise an error will be thrown

        Args:
            data (tuple(numpy.ndarray)): tuple containing numpy arrays with data to be saved
            data_names (tuple(numpy.ndarray)): file names for the above data
        """

        # use numpy binary extension for compatibility
        file_extension = "npy"

        # assumes self.data_dir is not devnull
        for i, d in enumerate(data):
            path = self.data_dir / f"{data_names[i]}.{file_extension}" 
            np.save(path, d)

        return None
    
class integrator_tools:
    """
    Contains centered difference methods for solving discretised differential equations.
    
    """
    # ----------------------------
    #       vector calc
    # ----------------------------
     
    def grad(self, vals, dx=1):
        """
        Centered difference n-dimensional grad function on a grid of discretised values
        
        Args:
            vals (numpy.ndarray): grid of points to operate on
            dx (float): discretised space step

        Returns:
            altered grid of points (numpy.ndarray)
        """

        # setup dimensions
        ndim = vals.ndim
        shape = vals.shape

        # look through dimensions 
        grads = np.empty(shape=(ndim, *shape))
        for i in range(0, ndim):
            # compute derivative using centered difference scheme
            grads[i] = (np.roll(vals, shift=-1, axis=i) - np.roll(vals, shift=1, axis=i))/(2*dx)

        return grads
    
    def lap(self, vals, dx=1):
        """
        Centered difference n-dimensional laplacian function on a grid of discretised values
        
        Args:
            vals (numpy.ndarray): grid of points to operate on
            dx (float): discretised space step

        Returns:
            altered grid of points (numpy.ndarray)
        """

        # dimensions
        ndim = vals.ndim

        # loop through dimensions
        laps = 0
        for i in range(0, ndim):
            # compute laplacian using centered difference scheme
            laps+=(np.roll(vals, shift=-1, axis=i) + np.roll(vals, shift=1, axis=i) - 2*vals)/(np.square(dx))

        return laps
    
    def curl(self, vals, dx=1):
        """
        Centered difference curl function on a grid of discretised values.
        Curl is strictly defined in 3 dimensions.
        
        Args:
            vals (numpy.ndarray): grid of points to operate on
            dx (float): discretised space step

        Returns:
            altered grid of points (numpy.ndarray)
        """

        # centered difference scheme derivative in a given axis
        derivative = lambda axis, x: (np.roll(x, shift=-1, axis=axis) - np.roll(x, shift=1, axis=axis))/(2*dx)

        # compute for all axes
        curl_0 = derivative(1, vals[2]) - derivative(2, vals[1])
        curl_1 = -(derivative(0, vals[2]) - derivative(2, vals[0]))
        curl_2 = derivative(0, vals[1]) - derivative(1, vals[0])

        return curl_0, curl_1, curl_2
    
    def div(self, vals, dx=1):
        """
        Centered difference n-dimensional divergence function on a grid of discretised values.
        
        Args:
            vals (numpy.ndarray): grid of points to operate on
            dx (float): discretised space step

        Returns:
            altered grid of points (numpy.ndarray)
        """

        # handle dimensions
        s = vals.shape
        dim = s[0]
        shape = s[1:]

        # centered difference scheme derivative in a given axis
        derivative = lambda axis, x: (np.roll(x, shift=-1, axis=axis) - np.roll(x, shift=1, axis=axis))/(2*dx)
        
        # loop over dimensions 
        div = np.zeros(shape=shape)
        for i in range(dim):
            div += derivative(i, vals[i])

        return div


    # ----------------------------
    #           solvers
    # ----------------------------

    def forward_euler(self, vals, f_vals, dx, dt):
        """
        Step an n-dimensional grid of points forward in time one discretised time step. Using the forward difference
        Euler scheme. 

        Args:
            vals (numpy.ndarray): grid of points to evolve
            f_vals (numpy.ndarray): values produced by the function to be solved
            dx (float): discretised space step
            dt (float): discretised time step

        Returns:
            evolved grid of points (numpy.ndarray)
        """
        ch = vals + dt * self.lap(f_vals, dx)

        return ch

    def jacobi(self, vals, f_vals, dx=1):
        """
        Step an n-dimensional grid of points forward in time one discretised time step using the Jacobi scheme.

        The time step has been optimised as half of the square of the space step and thus is not taken as an argument. 

        Args:
            vals (numpy.ndarray): grid of points to evolve
            f_vals (numpy.ndarray): values produced by the function to be solved
            dx (float): discretised space step

        Returns:
            evolved grid of points (numpy.ndarray)
        """
        
        # dimensions
        dim = vals.ndim

        # loop and sum
        j = np.zeros_like(vals)
        for i in range(0, dim):
            j += (np.roll(vals, shift=-1, axis=i) + np.roll(vals, shift=1, axis=i))

        # apply function value term
        j = (j + np.square(dx)*f_vals)/(2*dim)

        return j

    def gauss_seidel(self, vals, f_vals, dx=1):
        """
        Step an n-dimensional grid of points forward in time one discretised time step using the Gauss-Seidel scheme.

        The time step has been optimised as half of the square of the space step and thus is not taken as an argument. 
        
        The Gauss-Seidel scheme is a modified version of the Jacobi scheme which is slower in real time but takes less
        iterations to converge.

        Split up the grid of points into a checkerboard -> update one group with jacobi -> save the modified array 
        -> update the second group using the modified array

        i.e. twice as slow in real time as the Jacobi; perhaps possible to optimise better in compiled languages such as C++

        Args:
            vals (numpy.ndarray): grid of points to evolve
            f_vals (numpy.ndarray): values produced by the function to be solved
            dx (float): discretised space step

        Returns:
            evolved grid of points (numpy.ndarray)
        """
        # functions
        jacobi = self.jacobi

        # checkerboard mask
        white = (np.indices(vals.shape).sum(axis=0) % 2).astype(bool)
        black = ~white
        gs = vals.copy()

        # jacobi on white
        j = jacobi(gs, f_vals, dx)
        gs[white] = j[white]

        # jacobi on black
        j = jacobi(gs, f_vals, dx)
        gs[black] = j[black]

        return gs
    
    def successive_over_relax(self, vals, f_vals, dx=1, w=1):
        """
        Step an n-dimensional grid of points forward in time one discretised time step using the successive
        over relaxation (SOR) scheme.

        The SOR scheme is a modified version of the Gauss Seidel scheme which uses a mixture of the current timestep and the Gauss Seidel 
        computed next time step to produce a grid of points for the next time step. Depending on the relaxation parameter w, the convergence can 
        take less or more iterations. 

        To obtain the optimal relaxation parameter, see fluid_equations.determine_optimal_w.

        The time step has been optimised as half of the square of the space step and thus is not taken as an argument. 

        Args:
            vals (numpy.ndarray): grid of points to evolve
            f_vals (numpy.ndarray): values produced by the function to be solved
            dx (float): discretised space step
            w (float): the relaxation parameter

        Returns:
            evolved grid of points (numpy.ndarray)
        """

        gs = self.gauss_seidel(vals, f_vals, dx)
        s = w*gs + (1-w)*vals 
        return s

