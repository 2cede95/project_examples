import numpy as np 
import matplotlib.pyplot as plt
from sys import exit
from os import devnull
from pathlib import Path
from scipy.optimize import curve_fit

class difference_schemes:
    """
    Contains centered difference methods for solving discretised differential equations.
    """
    # ----------------------------
    #       vector calc
    # ----------------------------
     
    def grad(self, vals: np.ndarray, dx=1.) -> np.ndarray:
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
    
    def laplacian(self, vals: np.ndarray, dx=1.) -> np.ndarray:
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

        return laps #type: ignore
    
    def curl(self, vals: np.ndarray, dx=1.) -> tuple:
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
    
    # ----------------------------
    #       update methods
    # ----------------------------

    def forward_euler(self, vals: np.ndarray, f_vals: np.ndarray, dx: float, dt: float) -> np.ndarray:
        """
        Step an n-dimensional grid of points forward in time one discretised time step. Using the forward difference
        Euler scheme. 

        Stability requires dt/dx^2 <= 1/(2N) -> N is number of dimensions (Von Neumann Analysis)

        Args:
            vals (numpy.ndarray): grid of points to evolve
            f_vals (numpy.ndarray): values produced by the function to be solved
            dx (float): discretised space step
            dt (float): discretised time step

        Returns:
            evolved grid of points (numpy.ndarray)
        """
        fe = vals + dt * self.laplacian(f_vals, dx)

        return fe

    def jacobi(self, vals: np.ndarray, f_vals: np.ndarray, dx=1.) -> np.ndarray:
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

    def gauss_seidel(self, vals: np.ndarray, f_vals: np.ndarray, dx=1.) -> np.ndarray:
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

        # jacobi on black (with updated white squares)
        j = jacobi(gs, f_vals, dx)
        gs[black] = j[black]

        return gs
    
    def successive_over_relax(self, vals: np.ndarray, f_vals: np.ndarray, dx=1., w=1.) -> np.ndarray:
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

class simulation:

    def __init__(self, length: int, iv: float, bv: tuple, noise: float, tol: float, dx: float, dt: float, w = 1, update_method = "sor", save_dir = devnull) -> None:
        
        self.length = length
        self.iv = iv
        self.bv = bv
        self.noise = noise
        self.tol = tol
        self.dx = dx
        self.dt = dt
        self.w = w

        self.save_dir = save_dir

        self.solver = difference_schemes()

        if "jacobi" == update_method:
            self.update = self.solver.jacobi
        elif "gauss_seidel" == update_method:
            self.update = self.solver.gauss_seidel
        elif "sor" == update_method:
            self.update = self.solver.successive_over_relax

        if devnull is not save_dir:
            # suppress type warnings because save_dir is a pathlib.Path object if not devnull
            self.data_dir = save_dir / "data" #type: ignore
            self.data_dir.mkdir(exist_ok=True)
            self.fig_dir = save_dir / "figures" #type: ignore
            self.fig_dir.mkdir(exist_ok=True)

    def apply_boundary(self, arr: np.ndarray, boundary_vals: tuple) -> np.ndarray:
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

        dim = arr.ndim
        # if 2*dim != len(boundary_vals):
        #     print(f"Arr has invalid dimensions {dim} for provided boundary values {boundary_vals}.")
        #     exit()

        for a in range(dim):
            first = [slice(None)]*dim
            first[a] = slice(0, 1)
            last = [slice(None)]*dim
            last[a] = slice(-1, None)
            # try loops so boundaries can be left open
            try:
                if boundary_vals[2*a] is not None:              
                    arr[tuple(first)] = boundary_vals[2*a]
            except IndexError:
                    continue
            try:
                if boundary_vals[2*a + 1] is not None:
                    arr[tuple(last)] = boundary_vals[2*a + 1]
            except IndexError:
                    continue                    

        return arr
    
    def init_array(self, length: int, dim: int, iv: float, bv = None, noise = 0.) -> np.ndarray:
        """
        Initialize disretised spacial coordinate grid using numpy functions.

        Args:
            length (int): length of one side of the n-dimensional array. All dimensions share the same length.
            dim (int): number of dimensions in the array
            iv (float): initial value for the array sites
            bv (float,): boundary values for the array. Applied after the initial values. i.e.
                         bv = ((axis_0 start, axis_0 end), (axis_1 start, axis_1 end), ...., (axis_n start, axis_n end))
            noise (float): random noise to apply to the initial values
        """

        shape = (length,)*dim
        arr = np.ones(shape)*iv + np.ones(shape)*noise*np.random.uniform(0, 1, shape)

        if bv is not None:

            self.apply_boundary(arr, bv)

        return arr
    
    def plot(self, x: np.ndarray|list, y: np.ndarray|list, xlabel: str, ylabel: str, title: str, fig_label: str) -> None:
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
        ax.set_title(title)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xlabel(xlabel, fontweight="bold")

        # save figure
        if devnull != self.save_dir:
            self.fig_dir.mkdir(exist_ok=True)
            extension = "png"
            file_path = self.fig_dir /  f"{fig_label}-figure.{extension}"
            plt.savefig(file_path, format=extension, dpi=400, bbox_inches = 'tight') 

        # show
        plt.show()
    
    def plot3D(self, sf: np.ndarray, vf: tuple|np.ndarray, titles: tuple, xlabels: tuple, ylabels: tuple, zlabels: tuple) -> None:
        """
        Method which produces the following plots for use with the point_charge and vertical_wire functions:
            - 2D contour plot of plane which cuts the source in half (i-j plane);
            - 2D quiver plot of the field lines in the plane which cuts the current line in half (i-j plane);
            - 3D quiver plot using matplotlib library of the discretised vector field solution;
            - 1D plot of discretised order parameter strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised vector field strength against distance from the centre of the point charge on the same line as above.

        Args:
            sf (numpy.ndarray): scalar field data
            vf (numpy.ndarray): vector field data
            title (tuples(str)): tuple containing plotting labels 
            xlabels (tuples(str)): tuple containing x axis labels 
            ylabels (tuples(str)): tuple containing y axis labels 
            zlabels (tuples(str)): tuple containing z axis labels 
        """

        # constants
        length = sf.shape[0]
        mid = sf.shape[0]//2
        mid_line = mid, slice(None), mid

        # formatting
        vf0, vf1, vf2 = vf

    ####################### Field Slice 2D plot ############################

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        ax1.set_title(titles[0])
        ax1.set_xlabel(xlabels[0], fontweight="bold")
        ax1.set_ylabel(ylabels[0], fontweight="bold")
        ax2.set_title(titles[1])
        ax2.set_xlabel(xlabels[1], fontweight="bold")
        ax2.set_ylabel(ylabels[1], fontweight="bold")
        
        sfp = sf[:, :, mid] # scalar field plane
        x, y = np.meshgrid(
            np.arange(0, length),
            np.arange(0, length),
            indexing='ij'
        )
        c = ax1.contourf(x, y, sfp, cmap="RdBu")
        cbar = fig.colorbar(c, ax=ax1)
        cbar.set_label(label=zlabels[0], fontweight="bold")

        step = length//10
        x, y = np.meshgrid(
            np.arange(0, length, step),
            np.arange(0, length, step),
            indexing='ij'
        )

        vfp0, vfp1 = vf0[::step, ::step, mid], vf1[::step, ::step, mid] # vector field plane 
        q = ax2.quiver(x, y, vfp0, vfp1, scale=5, color="tab:red")

        # save plot
        if self.save_dir != devnull:
            fmt = "png"
            fig_path = self.fig_dir / f"scalar_field_2d.{fmt}"
            plt.savefig(fig_path, format=fmt)

        plt.suptitle("2D Slice Plots", fontweight="bold")
        plt.tight_layout()
        plt.show()

    ####################### Vector Field 3D plot ############################

        fig = plt.figure()
        ax = plt.axes(projection = '3d') # (rows, cols, position to add)
        ax.set_title(titles[2])
        ax.set_xlabel(xlabels[2], fontweight="bold")
        ax.set_ylabel(ylabels[2], fontweight="bold")
        # ax.set_zlabel(zlabels[2], fontweight="bold") doesnt work, dont know why. Following matplotlib docs.

        step = length//10
        x, y, z = np.meshgrid(
        np.arange(0, length, step),
        np.arange(0, length, step),
        np.arange(0, length, step),
        indexing='ij')
        mask = slice(None, None, step), slice(None, None, step), slice(None, None, step)
        q = ax.quiver(x, y, z, vf0[mask], vf1[mask], vf2[mask], length=2*step, color='tab:red')

        # save plot
        if self.save_dir != devnull:
            fmt = "png"
            fig_path = self.fig_dir / f"vector_field_3d.{fmt}"
            plt.savefig(fig_path, format=fmt)

        plt.suptitle("3D Vector Field Plot", fontweight="bold")
        plt.tight_layout()
        plt.show()

    ####################### Field Strength 1D plot ############################
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
        ax1.set_title(titles[3])
        ax1.set_xlabel(xlabels[3], fontweight="bold")
        ax1.set_ylabel(ylabels[3], fontweight="bold")
        ax2.set_title(titles[4])
        ax2.set_xlabel(xlabels[4], fontweight="bold")
        ax2.set_ylabel(ylabels[4], fontweight="bold")

        vfl = np.linalg.norm(vf, axis=0)[mid_line] #vector field line; vf has shape (3, len, len, len)
        sfl = sf[mid_line]
        vfl = vfl[mid:] # function is isotropic, plot positive half only
        sfl = sfl[mid:] 
        x = np.arange(0, len(vfl))

        # fit to 1/r^2 
        model = lambda x, a, b: a*np.pow(x, b)  
        vf_fit, _ = curve_fit(model, x[x!=0], vfl[x!=0], maxfev=10000) # x=0 explodes 1/r^2 
        sf_fit, _ = curve_fit(model, x[x!=0], sfl[x!=0], maxfev=10000, p0=(35, -1)) # and 1/r

        ax1.plot(x, sfl, color="tab:red", label = "Simulated", marker="o", markersize=3, markerfacecolor="black", markeredgecolor="black")
        a, b = sf_fit
        ax1.plot(x, model(x, a, b), color="tab:blue", label = f"a={a:.1f}, b={b:.1f}", linestyle="--")
        ax1.legend()
        ax2.plot(x, vfl, color="tab:red", label = "Simulated", marker="o", markersize=3, markerfacecolor="black", markeredgecolor="black")
        a, b = vf_fit
        ax2.plot(x, model(x, a, b), color="tab:blue", label = f"a={a:.1f}, b={b:.1f}", linestyle="--")
        ax2.legend()

        # save plot
        if self.save_dir != devnull:
            fmt = "png"
            fig_path = self.fig_dir / f"field_strength_1d.{fmt}"
            plt.savefig(fig_path, format=fmt)

        plt.suptitle("Field Strengh Plots", fontweight="bold")
        plt.tight_layout()
        plt.show()
    
    def save_nparray(self, data: list|tuple, data_names: list|tuple) -> None:
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

    def cahn_hilliard(self, N: int) -> None:
        """
        Solve the Cahn Hilliard equation using a forward euler scheme in 2 dimensions.
        If plot is True then a 1D plot of free energy vs iterations is presented.

        The plot is saved at ./runs/{timestamp}/figures as a .png
        Free energy data and iteration no. data are saved in ./runs/{timestamp}/data

        Args:
            N (int): maximum number of iterations to perform
        """    
        solver = self.solver
        lap = solver.laplacian
        grad = solver.grad
        forward_euler = solver.forward_euler
        apply_boundary = self.apply_boundary
        bv = self.bv

        # equations
        a = kappa = 1
        mu = lambda op, dx: -a*op*(1-np.square(op)) - kappa*lap(op, dx) # chemical potential
        fed = lambda op, dx: -(a/2)*np.square(op) + (a/4)*np.pow(op, 4) + (kappa/2)*np.square(grad(op, dx)) # free energy density

        # initial conditions
        dim = 2
        dx = self.dx
        op = self.init_array(self.length, dim, self.iv, None, self.noise) # order parameter

        # compute loop
        free_energy = np.zeros(shape=(N,))
        history = set() # for convergence check
        for i in range(N):
            chemical_potential = mu(op, dx)
            op = apply_boundary(forward_euler(op, chemical_potential, dx, self.dt), bv)
            free_energy[i] = np.sum(fed(op, dx))

            if 0 == i%1000:
                if op.tobytes in history:
                    print(f"Converged after {i} iterations.")
                    break
                else:
                    history.add(op.tobytes())
                

        # save data
        if devnull != self.save_dir:
            arr = (free_energy,)
            name = ("free_energy",)
            self.save_nparray(arr, name)

        # free energy plot
        x_label = "No. Iterations"
        y_label = "Total Free Energy"
        title = r"$\phi_{initial} =$ " + f"{self.iv}, " + r"$\delta x =$ " +f"{dx}, " + r"$\delta t =$ " + f"{self.dt}"
        fig_label = "cahn_hilliard"
        self.plot(np.arange(N), free_energy, x_label, y_label, title, fig_label)
    
    def cahn_hilliard_visual(self, N: int) -> None:
        """
        Solve the Cahn Hilliard equation using a forward euler scheme.
        If plot is True then a 1D plot of free energy vs iterations is presented. 

        The plot is saved at ./runs/{timestamp}/figures as a .png
        Free energy data and iteration no. data are saved in ./runs/{timestamp}/data

        A 2D animation of the order parameter evolution is presented during computaton. The plot is updated every 1000 iterations.

        Args:
            N (int): maximum number of iterations to perform
        """  
        solver = self.solver
        lap = solver.laplacian
        grad = solver.grad
        forward_euler = solver.forward_euler
        apply_boundary = self.apply_boundary
        bv = self.bv

        # equations
        a = kappa = 1
        mu = lambda op, dx: -a*op*(1-np.square(op)) - kappa*lap(op, dx) # chemical potential
        fed = lambda op, dx: -(a/2)*np.square(op) + (a/4)*np.pow(op, 4) + (kappa/2)*np.square(grad(op, dx)) # free energy density

        # initial conditions
        dim = 2
        dx = self.dx
        op = self.init_array(self.length, dim, self.iv, None, self.noise) # order parameter

        #interactive plot
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(op, vmin=-1.0, vmax=1.0, origin="lower", cmap="RdBu")
        ax.set_title(r"$\phi_{initial} =$ " + f"{self.iv}, " + r"$\delta x =$ " +f"{dx}, " + r"$\delta t =$ " + f"{self.dt}")
        ax.set_xlabel("i", fontweight="bold")
        ax.set_ylabel("j", fontweight="bold")
        cbar = plt.colorbar(im)
        cbar.set_label(label=r"Order Parameter, $\phi_{ij}$", fontweight="bold")
        plt.ion()

        # compute loop
        free_energy = np.zeros(shape=(N,))
        history = set()
        for i in range(N):
            chemical_potential = mu(op, dx)
            op = apply_boundary(forward_euler(op, chemical_potential, dx, self.dt), bv)
            free_energy[i] = np.sum(fed(op, dx))

            # update plot
            if i%1000 == 0:
                im.set_data(op)
                plt.pause(0.001)
                if op.tobytes in history:
                    print(f"Converged after {i} iterations.")
                    break
                else:
                    history.add(op.tobytes())

        # save data
        if devnull != self.save_dir:
            arr = (free_energy,)
            name = ("free_energy",)
            self.save_nparray(arr, name)

        # free energy plot
        plt.ioff()
        x_label = "No. Iterations"
        y_label = "Total Free Energy"
        title = r"$\phi_{initial} =$ " + f"{self.iv}, " + r"$\delta x =$ " +f"{dx}, " + r"$\delta t =$ " + f"{self.dt}"
        fig_label = "cahn_hilliard"
        self.plot(np.arange(N), free_energy, x_label, y_label, title, fig_label)
    
    def poisson(self, N_max: int, plot = True, savedata = True) -> int:
        """
        Preset simulation which solves Poisson's equation for the discretised electrostatic potential of a point charge with given initial conditions.
        Returns the number of iterations to converge.
        
        Four plots are produced if the plot argument is true:
            - 2D contour plot of plane which cuts the point charge in half (i-j plane);
            - 2D quiver plot of the field lines in the plane which cuts the current line in half (i-j plane);
            - 3D quiver plot using matplotlib library of the discretised electric field solution;
            - 1D plot of discretised electrostatic potential strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised electric field strength against distance from the centre of the point charge on the same line as above.

        Data is saved in the ./runs/{timestamp}/data as .npy binaries. 
        Plots saved in ./runs/{timestamp}/figures as .png images with a dpi of 150.

        Args:
            N_max (int): maximum number of iterations to attempt for convergence
            plot (bool): plot after convergence
            savedata (bool): saves electric field and electrostatic potential data as .npy binaries after convergence

        Returns:
            Number of iterations to converge (int)
        """
        solver = self.solver
        grad = solver.grad
        update = self.update

        # equations
        electric_field = lambda op, dx: -1*grad(op, dx)
        def point_charge(length, dim, pos, charge):
            cd = np.zeros(shape=(length,)*dim)
            cd[pos] = charge
            return cd
        boundary = self.apply_boundary
        
        # initial conditions
        dim = 3
        dx = self.dx
        bv = self.bv
        tol = self.tol
        op = self.init_array(self.length, dim, self.iv, bv, self.noise) # order parameter
        charge = 500
        cd = point_charge(self.length, dim, (self.length//2,)*3, charge) # charge distribution

        if solver.successive_over_relax == update:
            w = self.w
            args = (dx, w)
        else:
            args = (dx,)


        # compute loop
        iterations = N_max
        for i in range(N_max):
            op_old = op.copy()
            op = boundary(update(op, cd, *args), bv)

            # convergence check
            if np.max(np.abs(op - op_old)) < tol:
                print(f"Converged in {i} iterations")
                iterations = i
                break
            elif i == N_max-1:
                print(f"Did not converge; {N_max} iterations elapsed")
        
        ef = electric_field(op, dx)

        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (op, ef)
            name = ("electrostatic_potential", "electric_field")
            self.save_nparray(arr, name)

        # plot
        if True == plot:
            titles = ("Electrostatic Potential Slice",
                    "Electric Field Slice",
                    r"Electric Field, $\vec{E} = -\vec{\nabla}\varphi$",
                    r"Magnitude of Electrostatic Potential fit, $ar_{ijk}^{b}$",
                    r"Magnitude of Electric field fit, $ar_{ijk}^{b}$")
            xlabels = ("i",
                    "i",
                    "i",
                    r"Displacement From Centre, $r_{ijk}$",
                    r"Displacement From Centre, $r_{ijk}$")
            ylabels = ("j",
                    "j",
                    "j",
                    r"Electrostatic Potential Magnitude, $\|\varphi\|$",
                    r"Electric Field Magnitude, $\|\vec{E}\|$")
            zlabels = (r"Electrostatic potential, $\varphi$",
                    r"Electric Field, $\vec{E}$"
                    "k",
                    None,
                    None)
            solver_name = "jacobi"
            self.plot3D(op, ef, titles, xlabels, ylabels, zlabels)

        return iterations
    
    def maxwell(self, N_max: int, plot = True, savedata = True) -> int:
        """
        Preset simulation which solves Maxwell's equations for the magnetic vector potential of a line of current
        with given initial and boundary conditions.

        Returns the number of iterations to converge.
        
        Four plots are produced if the plot argument is true:
            - 2D contour plot of plane which cuts the current line in half (i-j plane);
            - 2D quiver plot of the field lines in the plane which cuts the current line in half (i-j plane);
            - 3D quiver plot using matplotlib library of the magnetic field solution;
            - 1D plot of the k componenet of the discretised magnetic vector potential strength against distance on a line intersecting the point charge on the (i-j) plane;
            - 1D plot of discretised electric field strength against distance from the centre of the point charge on the same line as above.

        Data is saved in the ./runs/{timestamp}/data as .npy binaries. 
        Plots saved in ./runs/{timestamp}/figures as .png images with a dpi of 150.

        Args:
            N_max (int): maximum number of iterations to attempt for convergence
            plot (bool): plot after convergence
            savedata (bool): saves magnetic field and k component of vector potential as .npy binaries after convergence

        Returns:
            Number of iterations to converge (int)
        """
        solver = self.solver
        curl = solver.curl
        update = self.update

        # equations
        magnetic_field = lambda op, dx: curl(op, dx)
        def line_current(length, dim, pos, current):
            cd = np.zeros(shape=(length,)*dim)
            cd[pos] = current
            return cd
        boundary = self.apply_boundary
        
        # initial conditions
        dim = 3
        dx = 1
        bv = self.bv
        tol = self.tol
        op = self.init_array(self.length, dim, self.iv, bv, self.noise) # order parameter (in k direction only)
        current = 50
        cd = line_current(self.length, dim, (self.length//2, self.length//2, slice(None)), current) # current distribution
        
        if solver.successive_over_relax == update:
            w = self.w
            args = (dx, w)
        else:
            args = (dx,)

        # compute loop
        iterations = N_max
        for i in range(N_max):
            op_old = op.copy()
            op = boundary(update(op, cd, *args), bv)

            # convergence check
            if np.max(np.abs(op - op_old)) < tol:
                print(f"Converged in {i} iterations")
                iterations = i
                break
            elif i == N_max-1:
                print(f"Did not converge; {N_max} iterations elapsed") 

        zero = np.zeros_like(op)
        op = np.array([zero, zero, op])
        b = magnetic_field(op, dx)
        
        # save data
        if devnull != self.save_dir and True == savedata:
            arr = (op, b)
            name = ("magnetic_vector_potential_k", "magnetic_field")
            self.save_nparray(arr, name)

        # plot   
        if True == plot:
            titles = ("Vector Potential Slice",
                    "Magnetic Field Slice",
                    r"Magnetic Field, $\vec{B} = \vec{\nabla}\times\vec{A}$",
                    r"k-component of Vector Potential fit, $ar_{ijk}^{b}$",
                    r"Magnitude of Magnetic Field fit, $ar_{ijk}^{b}$")
            xlabels = ("i",
                    "i",
                    "i",
                    r"Displacement From Centre, $r_{ijk}$",
                    r"Displacement From Centre, $r_{ijk}$")
            ylabels = ("j",
                    "j",   
                    "j",
                    r"k-component of Vector Potential, $\vec{A}\cdot\vec{e_{k}}$",
                    r"Magnetic Field Magnitude, $\|\vec{B}\|$")
            zlabels = (r"k-componenet of Vector Potential, $\vec{A}\cdot\vec{e_{k}}$",
                    r"Magnetic Field, $\vec{B}$"   
                    "k",
                    None,
                    None)
            solver_name = "jacobi"
            self.plot3D(op[2], b, titles, xlabels, ylabels, zlabels)

        return iterations
    
    def optimal_w(self, w_min: float, w_max: float, res: int, sim, N_max: int) -> None:
        """
        Computes a range of preset simluations, sim, to deteminre the optimal relaxation parameter when sim is solved using the
        successive over relaxation scheme. A 1D plot is presented with the no. of iterations to converge against relaxation parameter w.

        The optimal value is printed to stdout and marked on the plot with a vertical line.

        Relaxation parameter and iteration data are saved at ./runs/{timestamp}/data in .npy binaries.
        
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        Intermediate data for sims runs are NOT saved 
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        sim must have the function signature (N_max, w=1, plot=True, savedata=True)

        Args:
            w_min (float): starting value of w to define the sample space w_vals
            w_max (float): maximum value of w to define the sample space w_vals
            res (int): number of w points to sample
            sim (function): preset simulation to solve; i.e. self.poisson or self.maxwell
            N_max (int): maximum number of iterations to attempt for convergence

        Returns:
            Optimal value of relaxation parameter (float)
        """
        
        # force SOR solver
        self.update = self.solver.successive_over_relax

        w_vals = np.linspace(w_min, w_max, res)
        print(f"w values: [{w_vals[0]:.2f}, {w_vals[1]:.2f}, {w_vals[2]:.2f}, ..., {w_vals[-2]:.2f}, {w_vals[-1]:.2f}]")

        # compute loop
        iterations = np.zeros_like(w_vals)
        for i, w in enumerate(w_vals):

            # will converge instantly but will not give correct solution
            if 0. == w:
                continue
            
            self.w = w
            print(f"w = {w:.2f} -> ", end="")
            iterations[i] = sim(N_max, False, False)

        # save data
        if devnull != self.save_dir:
            arr = (w_vals, iterations)
            name = ("relaxation_parameter", "iterations")
            self.save_nparray(arr, name)

        # optimal w
        min_ind = np.argmin(iterations)
        optimal_w = w_vals[min_ind]
        print(f"Optimal w is {optimal_w:.2f}.")

        # plot
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
