import matplotlib.pyplot as plt
import numpy as np
from helpers.compartmental import Simulate, NaiveStepper, SDEStepper, State

#print("hi")

class InfectedRate():
    """
        Dependency Injection for the Rate Class. Implement the rate of a\
        moving one from susceptible to infected.
    """

    def __init__(self, b:float):
        """
            Initialisation of the b.
        """
        self.b = b

    def computerate(self, curstate:State)->float:
        """
            Compute the rate of the moving a susceptible to infected
        """
        x = curstate.x
        rate = (self.b*x[0]*x[1])/np.sum(x)
        print(f"Infection state: {x} rate: {rate} ")
        return rate

class RecoveredRate():
    """
        Dependency Injection for the Rate Class. Implement the rate of a\
        moving one from infected to recovered.
    """
    def __init__(self, gamma) -> None:
        """
            Initialise the Recovered rate class

            Parameters
            ----------
            gamma: float
                The recovery rate.
        """
        self.gamma:float = gamma

    def computerate(self, curstate:State)->float:
        """
            Compute the rate of infected person to be recovered
        """
        x = curstate.x
        rate = self.gamma*x[1]/np.sum(x)
        print(f"Recovery state: {x} rate: {rate} ")
        return rate

if __name__ == "__main__":

    b = 2
    gamma = 30
    dt = 1
    steps = 200
    rates = [
            InfectedRate(b),
            RecoveredRate(gamma)
    ]

    increments = [
            np.array( (-1, 1, 0) ),
            np.array( (0 , -1, 1) )
    ]

    stepper:NaiveStepper = NaiveStepper(
        rates= rates,
        increments = increments,
        ncompartments=3,
            dt =dt
    )
    s = 10000
    i = 20
    initstatex = np.array((s, i, 0))
    initstate = State( x= initstatex, _n_compartments = 3)
    simulator:Simulate = Simulate(
        initstate= initstate,
        stepper=stepper
    )
    simul, times = simulator.simulate_and_return(steps=steps,endcondition = np.array((0, 0, s+i)))
    print(simul.shape)
    print(times.shape)
    plt.plot(simul.T)
    plt.show()
