import matplotlib.pyplot as plt
import numpy as np
from helpers.compartmental import Simulate, NaiveUrnStepper, SDEStepper, State

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
#        print(f"Infection state: {x} rate: {rate} ")
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
        rate = self.gamma*x[1]#/np.sum(x)
#        print(f"Recovery state: {x} rate: {rate} ")
        return rate

if __name__ == "__main__":
    plt.rcParams['font.family'] = 'serif'
    #Simulation Parameters
    b = 2
    s = 1000
    inf = 20
    gamma = 1
    steps = 20000
    total_n = s+inf
    # Number of the 
    alpha_ssa = 0.05
    trials = 100

    infectedfn:InfectedRate= InfectedRate(b)
    recoveredfn:RecoveredRate= RecoveredRate(gamma)

    # Define the maximum rates to get dt
    maxinfected = infectedfn.computerate(State(np.array((inf+s, inf+s, 0)), 3))
    maxrecover = recoveredfn.computerate( State(np.array((0, inf+s, 0)), 3) )
    dt = 1/(maxinfected + maxrecover)

    print( f' maxrecover: {maxrecover} \n maxinfected: {maxinfected} \n dt: {dt}')

    rates = [
        infectedfn,
        recoveredfn
    ]

    increments = [
            np.array( (-1, 1, 0) ),
            np.array( (0 , -1, 1) )
    ]

    stepper:NaiveUrnStepper = NaiveUrnStepper(
        rates= rates,
        increments = increments,
        ncompartments=3,
        dt =dt
    )
    av = np.zeros((3, steps+1))

    for i in range(trials):
        initstatex = np.array((s, inf, 0))
        initstate = State( x= initstatex, _n_compartments = 3)
        simulator:Simulate = Simulate(
            initstate= initstate,
            stepper=stepper
        )
        simul, times = simulator.simulate_and_return(steps=steps)
        av+=simul
        print(simul.shape)
        print(times.shape)
        plt.plot(times, simul[0]/total_n, alpha = alpha_ssa, color = 'orange')
        plt.plot(times, simul[1]/total_n, alpha = alpha_ssa, color = 'red')
        plt.plot(times, simul[2]/total_n, alpha = alpha_ssa, color = 'green')
    # just label
    initstatex = np.array((s, inf, 0))
    initstate = State( x= initstatex, _n_compartments = 3)
    simulator:Simulate = Simulate(
        initstate= initstate,
        stepper=stepper
    )
    simul, times = simulator.simulate_and_return(steps=steps)
    print(simul.shape)
    print(times.shape)
    av +=simul
    plt.plot( times, simul[0]/total_n, alpha = alpha_ssa, color = 'orange')
    plt.plot( times, simul[1]/total_n, alpha = alpha_ssa, color = 'red')
    plt.plot( times, simul[2]/total_n, alpha = alpha_ssa, color = 'green')
    av=av/(trials+1)

    print(av[:,0], '\n here \n:', simul[:,0])
    plt.plot( times, av[0]/total_n, alpha = 1, label =r' $s_{mean}$', color = 'orange')
    plt.plot( times, av[1]/total_n, alpha = 1, label =r' $i_{mean}$', color = 'red')
    plt.plot( times, av[2]/total_n, alpha = 1, label =r' $r_{mean}$', color = 'green')
    
    
    #######################################
    #######################################
    ####### Mean Field Model ##############
    #######################################
    #######################################
    dt_new = dt
    # print(euler_states[0])
    steps = int(steps*dt/dt_new) +1
    dt = dt_new
    euler_states = np.zeros((steps,3))
    euler_states[0, :] = initstatex/(total_n)
    # steps = 5
    gamma_sir = gamma#/(s+inf)
    b_sir = b
    for i in range(1,steps):
        euler_states[i,0] = euler_states[i-1,0] - (b_sir*euler_states[i-1,0]*euler_states[i-1,1])*dt
        euler_states[i,1] = euler_states[i-1,1] + (b_sir*euler_states[i-1,0]*euler_states[i-1,1] -gamma_sir*euler_states[i-1,1])*dt
        euler_states[i,2] = euler_states[i-1,2] + gamma_sir*euler_states[i-1,1]*dt
        if euler_states[i,0]<=0:
            euler_states[i,0] = 0
        if euler_states[i,1]<=0:
            euler_states[i,1] = 0
        if euler_states[i,2]<=0:
            euler_states[i,2] = 0
    times_euler = np.arange(0, steps, 1)*dt_new
    plt.plot(
        times_euler,
        euler_states[:,0],
        marker = 'x',
        linestyle = 'dashdot',
        label = r'$s_{euler}$',
        color = 'orange',
        markevery = 1000
    )
    plt.plot(
        times_euler,
        euler_states[:,1],
        marker = 'x',
        linestyle = 'dashdot',
        label = r'$i_{euler}$',
        color = 'red',
        markevery = 1000
    )
    plt.plot(
        times_euler,
        euler_states[:,2],
        marker = 'x',
        linestyle = 'dashdot',
        label = r'$r_{euler}$',
        color = 'green',
        markevery = 1000
    )


    plt.title(f'Simulating {trials} trials for the URN Model for the SIR')
    plt.xlabel('Time')
    plt.ylabel('Fraction of Population')
    plt.subplots_adjust(right=0.80)
    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5)
    )
    plt.savefig(f'urnmodel_naive.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
