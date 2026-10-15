from planet import Planet

class Planet_Euler(Planet):
    """
    Planet class ammended to use euler direct step numerical integration method.
    """
    def update_pos(self, G, dt):
        """
        Method to update the position of the body after a small timestep using the Euler method.
        """
        # keep old position to check for year
        self.r_old = self.r
        
        # update position first: Euler
        self.r = self.r + self.v*dt 

    def update_vel(self, G, dt, p):
        """
        Method to update the velocity of the body after a small timestep using the Euler method.
        """
        # update velocity second: Euler
        a_new = self.update_acc(G, p)
        self.v = self.v + a_new*dt 
        # now update acc ready for next iteration
        self.a_old = self.a
        self.a = a_new

