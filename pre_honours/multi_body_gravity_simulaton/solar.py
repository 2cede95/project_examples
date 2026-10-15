import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from numpy.linalg import norm
from time import sleep
from planet import Planet
from planet_euler import Planet_Euler

class Solar(object):
    """
    Class to run the orbital simulation
    """ 
    def __init__(self, Euler=False):
   
        # introduced checks so that only values which are needed for the current experiment are recorded to avoid lag.
        self.experiment_1_check = False
        self.experiment_2_check = False
        self.experiment_3_check = False
        self.time_alignments_list = [0]
        self.delta_time = [0]
        self.recent_align = 200
        self.check_counter = self.recent_align

        # lists which are used to plot the total energies against time
        # time is recorded in Earth years
        self.method_euler = Euler
        self.method_beeman = not Euler 
        self.total_energy_list = []
        self.time_list = []


        # Read data for planets from parameters-solar file given in template
        input_data = []
        filename = "parameters-solar.txt"
        file_in = open(filename, "r")
        for line in file_in.readlines():
            if (not line.startswith("#")):
                input_data.append(line)
        file_in.close()
        
        # simulation parameters
        self.niter = int(input_data[0])
        self.dt = float(input_data[1])
        self.G = float(input_data[2])

        # list for sun and planets
        self.bodies = []
        
        # rest of input data is sun and planets
        # first entry must be sun
        for i in range(3, len(input_data)-3, 5):
            name = input_data[i]
            mass = float(input_data[i+1])
            orbit = float(input_data[i+2])
            colour = input_data[i+3]
            period = float(input_data[i+4])
            if self.method_euler:
                self.bodies.append(Planet_Euler(name, mass, orbit, colour, period))
            else:
                self.bodies.append(Planet(name, mass, orbit, colour, period))
                                
        # set initial positions and velocities relative to sun
        # sum must be first element in bodies list!
        for i in range(0, len(self.bodies)):
            self.bodies[i].initialise(self.G, self.bodies[0])

    def init(self):
        """
        Method to return initialiser for animator
        """
        return self.patches

    def animate(self, i, lim):
        """
        Method to be animated, appends relevant information to lists for experiments.
        """
        # keep track of time in earth years
        time = (i+1)*self.dt

       # open file to record total energy
        if self.method_beeman:
            energy_list=open("energy_list_beeman.txt", "a")
        else:
            energy_list=open("energy_list_euler.txt", "a")
         
        # update positions
        for j in range(0, len(self.bodies)):
            self.bodies[j].update_pos(self.G, self.dt)
            self.patches[j].center = self.bodies[j].r
            
        # then update velocities
        for j in range(0, len(self.bodies)):
            for k in range(0, len(self.bodies)):
                if (j != k):
                    self.bodies[j].update_vel(self.G, self.dt, self.bodies[k])

        # check year and print year if new year for any planet
        # do not do this in experiment 3 so it does not lag
        if not self.experiment_3_check: 
            for j in range(0, len(self.bodies)):
                if (self.bodies[j].new_year()):
                    print (f"{self.bodies[j].name.strip()} "
                           f"{self.bodies[j].year} years = {time} earth years.")
                    #append list of obital periods with the value for this full rotation
                    if self.experiment_1_check:
                        self.bodies[j].period_list.append(time)

                    #if new year is earth year, also print total energy
                    if (self.bodies[j].name.strip() == "earth"):
                        # need to convert from earth masses AU^2 yr^-2 to kg m^2 s-2 (J)
                        # 1 earth mass = 5.97219e24 kg
                        # 1 AU = 1.496e+11 m
                        c =(5.97219e+24*1.496e+11*1.496e+11)/(3.154e+7*3.154e+7)
                        energy = self.energy()*c
                        print(f"Time = {time} earth years. "
                              f"Total energy = {energy:.3e} J.")
                        #append total energy at the end of each earth year
                        energy_list.write(f"{energy}\n")
                        if self.experiment_2_check:
                            self.time_list.append(time)
                            self.total_energy_list.append(energy)
        
        # checking if planets are aligned for experiment 3
        # if timestep is too low ( <0.04) then alignments are double counted and the average becomes inaccurate. 

        #stop counting when alignment detected for clarity and to avoid double counting, chose 200 timesteps because from observation of the simulation an alignment wil never happen within 200 timesteps of another
        

        if self.recent_align%self.check_counter != 0:
            self.recent_align+=1


        if self.experiment_3_check:
            alignments = len(self.time_alignments_list)
            if self.check_alignnment() and self.recent_align%self.check_counter==0:
                #stop counting when alignment detected for clarity and to avoid double counting, chose 200 timesteps because from observation of the simulation an alignment wil never happen within 200 timesteps of another
                self.recent_align += 1
                self.time_alignments_list.append(time)
                alignments = len(self.time_alignments_list) 
                print("Planets are aligned!")
                time_diff = time-self.time_alignments_list[-2]
                self.delta_time.append(time_diff)

            if time%100==0:
                if alignments > 1:
                    print(f"Time = {time} Earth years; {alignments-1} alignments; time since last alignment = {time - self.time_alignments_list[alignments-1]} Earth years")
                else:
                    print(f"Time = {time} Earth years; {alignments-1} alignments")
         
        energy_list.close()

        # making plot follow sun so the system is still visible at large times
        # update every 5000 iterations otherwise it causes lag
        if i%5000==0:
            sun_x = self.bodies[0].r[0]
            sun_y = self.bodies[0].r[1] 
            plt.xlim(sun_x - lim, sun_x + lim)
            plt.ylim(sun_y - lim, sun_y + lim)
            plt.draw()

        # close animation plot once required number of iterations has passed.
        if i == self.niter-1:
            plt.close()
        
        return self.patches

    def energy(self):
        """
        Method to return the sum of the kinetic energy of the bodies in the system and their potential energies.
        """
        ke = 0.0
        pe = 0.0
        for j in range(0, len(self.bodies)):
            ke += self.bodies[j].kinetic_energy()
            for k in range(0, len(self.bodies)):
                if (k != j):
                    r = norm(self.bodies[k].r - self.bodies[j].r)
                    pe -= self.G*self.bodies[j].m*self.bodies[k].m / r
        # divide pe by two to avoid double counting
        pe = pe / 2
        tot_energy = ke + pe
        return tot_energy

    def run(self):
        """
        Method to run the animation.
        """
        # set up the plot components        
        fig = plt.figure()
        ax = plt.axes()

        # create an array for patches (planet and moons)
        self.patches = []

        # get orbital radius of outermost body to set size of
        # orbiting bodies and of plot
        # hacky - should really check to see which body is outermost
        max_orb = math.sqrt(np.dot(self.bodies[-1].r, self.bodies[-1].r))

        # add the planet and moons to the Axes and patches
        for i in range(0, len(self.bodies)):
            if (i == 0):
                self.patches.append(
                    ax.add_patch(plt.Circle(self.bodies[i].r, 0.05*max_orb,
                                            color=self.bodies[i].c, animated=True)))
            else:
                self.patches.append(
                    ax.add_patch(plt.Circle(self.bodies[i].r, 0.02*max_orb,
                                            color=self.bodies[i].c, animated=True)))

        # set up the axes
        # scale axes so circle looks like a circle and set limits
        # with border b for prettier plot

        b = 1.2
        lim = max_orb*b
        ax.axis("scaled")
        sun_x = self.bodies[0].r[0] 
        sun_y = self.bodies[0].r[1] 
        ax.set_xlim(sun_x - lim, sun_x + lim)
        ax.set_ylim(sun_y - lim, sun_y + lim)

        anim = FuncAnimation(
            fig, self.animate, init_func=self.init,
            frames=self.niter, repeat=False, interval=1, blit=True, fargs=(lim,))

        plt.show()
    
    def check_alignnment(self):
        """
        Method to check if all planets are aligned.
        """
        threshold = (2*math.pi)*0.05
        aligned = 0
        # planets aligned only if they are in the same quadrant and have an angle between themselves and the sun within the threshold
        # start counting at second body because need to have a reference angle, else no angle to compare to 
        for i in range(1, len(self.bodies)-1):
            if abs(self.angle(i+1) - self.angle(i)) < threshold:
                aligned += 1
        if aligned == len(self.bodies)-2:
            return True
        return False

    def angle(self, i):
        """
        returns angle between two bodies
        """
        # calculates angle between vectors (with sun as origin) using dot product
        r_rel = self.bodies[i].r - self.bodies[0].r
        squares = np.power(r_rel, 2)
        magnitude = np.sqrt(squares[0] + squares[1])
        # measuring angle from the positive x-axis - therefore using unit vector in that direction 
        dot_product = np.dot([1, 0], r_rel)
        angle = np.arccos(dot_product/magnitude)
        # ensuring vectors are all in the same quadrant, otherwise they are not aligned
        if r_rel[1] < 0:
            angle = 2*math.pi - angle
        return angle
    
    def mean_delta(self, l):
        """
        calculates mean value of time between alignments.
        """
        # omitting first item in list because it is only there so calculations do not break
        sum_of_items = 0
        for i in range(1, len(l)):
            sum_of_items += l[i]
        mean = sum_of_items/(len(l)-1)
        return mean

    def print_message(self, message):
        """
        Method to print the message at the beginning of each experiment.
        """
        print(message)
        sleep(5)
        print("-"*50 + "\nSimulation running, please wait...\n" + "-"*50)

    def experiment_1(self):
        """
        Method to compare the orbital periods of the planets in the simulation to values measured by NASA. The actual real measured values are compared to the last recorded measurements in the simulation.
        """
        self.experiment_1_check = True
        self.dt = 0.005
        self.niter = 32000
        simulation_time = self.dt*self.niter
        message = f"The simulation will run for {simulation_time} Earth years. Once the simulation has completed the difference in orbital period will be printed to the terminal for each body in the system. "
        self.print_message(message)
        #run the simulation, time in earth years
        self.run()
        # skipping the 0th item because that is the sun
        print("\n")
        for i in range(1, len(self.bodies)):
            #simulated period is the average time between full rotations
            #data from NASA in earth days so must be converted to earth years
            simulated_period = (self.bodies[i].period_list[-1] - self.bodies[i].period_list[-2])
            measured_period_years = self.bodies[i].orbital_period*1/365
            # how much the simulated value differs from the actual orbital period
            difference = abs(simulated_period - measured_period_years)
            percent_difference = (difference/simulated_period)*100
            print(f"{self.bodies[i].name}Simulated orbital period = {simulated_period} Earth years\nActual measured orbital period = {measured_period_years} Earth years\nDifference = {difference} Earth years\nDifference as a percentage = {percent_difference}%\n")


    def experiment_2(self):
        """
        Method to compare the total energies of the simulation using the Beeman integration method and the Euler integration method.
        """
        # clearing files storing energy values so that only values from this iteration are stored
        for i in ["energy_list_beeman.txt", "energy_list_euler.txt"]:
            file_in = open(i, "w")
            file_in.write("#Energies are measured in Joules\n")
            file_in.close()
        euler = Solar(Euler=True)
        #defining time that simulation should run for in Earth years
        timestep = 0.02
        niter = 20000
        systems = [self, euler]
        for i in systems:
            i.dt, i.niter, i.experiment_2_check = timestep, niter, True
        #beeman and self will have same timestep and number of iterations
        #duration in earth years
        simulation_time = self.dt*self.niter
        message = f"The simulation will run for {simulation_time} Earth years. Once the simulation has completed a graph will be plotted with the total energy against time elapsed.The plot will include both the three-step Beeman and Euler direct step numerical integration methods."
        self.print_message(message)
        self.run()
        euler.run()

        x1, x2, y1, y2 = euler.time_list, self.time_list, euler.total_energy_list, self.total_energy_list

        plt.plot(x1, y1, "r", label="Euler method")
        plt.plot(x2, y2, "b", label="Beeman method")
        plt.xlabel("Time (Earth years)")
        plt.ylabel("Total Energy (Joules)")
        plt.title("Comparison of total energy against time for the Beeman and Euler numerical integration methods") 
        plt.legend(loc="upper left")
        plt.show()

    def experiment_3(self):
        """
        Method to plot a graph of the time between planetary alignments vs total elapsed time and print the mean time between planetary alignments.
        """
        self.experiment_3_check = True
        self.niter = 75000*2
        self.dt = 0.02
        simulation_time = self.niter*self.dt
        message = f"Simulation will run for {simulation_time} Earth years. Once the simulation has completed a graph of the time between collisions vs time elapsed will be plotted.The mean value of time between collisions will also be printed."
        self.print_message(message)
        self.run()
        y = self.delta_time
        x = self.time_alignments_list
        mean = self.mean_delta(y)
        print(50*"*" + f"\nAverage alignment seperation is {mean} Earth years\n" + 50*"*")
        plt.plot(x, y, "r")
        plt.title("Time between alignments against total time of simulation")
        plt.ylabel("Time between alignments (Earth years)")
        plt.xlabel("Time (Earth years)")
        plt.show()
        
