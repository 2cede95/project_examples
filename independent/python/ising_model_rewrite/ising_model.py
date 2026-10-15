from physics_tools import *

def main():

    ising_lattice = SquareIsingLattice(50)
    simulation = Simulation(sweeps=1000,
                            correlation_gap=10,
                            ising_lattice=ising_lattice
                            )
    simulation.compute_observables_and_plot_glauber(min_beta=0.1, max_beta=3.0, beta_step=0.1)
    # simulation.compute_observables_and_plot_kawasaki(min_beta=0.1, max_beta=3.0, beta_step=0.1)

main()