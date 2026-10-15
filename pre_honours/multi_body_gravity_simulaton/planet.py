import math
import matplotlib
matplotlib.use("TKAgg")
import numpy as np
from numpy.linalg import norm

class Planet(object):
    """
    Planet class which uses beeman numerical integration method to update its position and velocity.
    """

    def __init__(self, name, mass, orbit, colour, period):
        self.name = name
        # mass in kg
        self.m = mass
        # orbital radius in m
        self.orbit = orbit
        # colour - need to strip trailing line return!
        self.c = colour.strip()
        # set year to zero
        self.year = 0
        # initialize list that contains measured orbital periods
        self.period_list = [0]
        # value of actual period measured in real life
        self.orbital_period = period 

    def initialise(self, G, p):
        """
        Method to initialize the planets physical properties
        """
        # inital position, initial coords = (orbit radius, 0)
        self.r = np.array([self.orbit, 0])
        # inital velocity, tangential to position
        # speed = sqrt(G*marsmass/r)
        if (self.orbit == 0.0):
            self.v = np.array([0, 0])
        else:
            vel = math.sqrt(G*p.m/self.orbit)
            self.v = np.array([0, vel])
        # intial accelatation, using gravitational force law
        if (self.orbit == 0.0):
            self.a = np.array([0, 0])
        else:
            self.a = self.update_acc(G, p)
        # set acc_old = acc to start Beeman
        self.a_old = self.a

    def update_pos(self, G, dt):
        """
        Method to update the bodies position using the Beeman method.
        """
        # keep old position to check for year
        self.r_old = self.r
        
        # update position first: Beeman
        self.r = self.r + self.v*dt + (4*self.a - self.a_old)*dt*dt/6.0
        
    def update_vel(self, G, dt, p):
        """
        Method to update the bodies velocity using the Beeman method.
        """
        # update velocity second: Beeman
        a_new = self.update_acc(G, p)
        self.v = self.v + + (2*a_new + 5*self.a - self.a_old)*dt/6.0
        # now update acc ready for next iteration
        self.a_old = self.a
        self.a = a_new

    def update_acc(self, G, p):
        """
        Method to update the bodies acceleration using the gravitational force law
        """
        # update acc (gravitational force law)
        pos = self.r - p.r
        a = -G*p.m*pos/math.pow(norm(pos),3)
        return a

    def new_year(self):
        """
        Method to detected when the body completes one full revolution around the sun, (one year)
        """
        # update the year when the planet passes the +x axis
        if (self.r_old[1] < 0.0 and self.r[1] >= 0.0):
            self.year +=1
            return True
        else:
            return False

    # determine kinetic energy
    def kinetic_energy(self):
        """
        Method to return the kinetic energy of the body.
        """
        # ke in J
        ke = (np.dot(self.v, self.v))*self.m/2
        return ke

