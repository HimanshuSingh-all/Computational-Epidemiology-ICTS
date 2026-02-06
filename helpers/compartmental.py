import numpy as np
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Any, Protocol


@dataclass
class State:
    """
        Represents the state of the compartment.
        
        Parameters
        ----------
        x: NDArray
            A one dimesional array of integers.
        n_compartments: int
            The number of _compartments_ or _species_.

        Attributes
        ----------
        x: NDArray
            A one dimesional array of integers.
        n_compartments: int
            The number of _compartments_ or _species_.
    """
        

    x:NDArray
    _n_compartments: int

    def __post_init__(self):
        if self.x.ndim!=1:
            raise ValueError("x should be a 1 Dimensional array")

        if self.x.shape[0]!=self._n_compartments:
            raise ValueError

        if not np.issubdtype(self.x.dtype, np.integer):
            raise ValueError("The state should be of a type integers")

    def update_state(self, newstate:NDArray)->None:
        """
            Update the state of the system.

            Parameters:
            -----------
            newstate: NDArray
                The new state with which to update the state. `numpy.subtypte(numpy.integer, newstate.dtype)` must be True
        """

        if newstate.ndim!=1:
            raise ValueError("The updating state should be a 1 dimensional array of integers")

        if not np.issubdtype(newstate.dtype, np.integer):
            raise ValueError("The updating state should be of a type integers")

        if newstate.shape[0]!=self._n_compartments:
            raise ValueError('The update state should have the same size as the number of the compartments')
        self.x = newstate
            # What happens if I do `del newstate` after

    @property
    def get_compartments(self):
        return self._n_compartments

    def get_state_copy(self):
        """
            Returns a copy of the state.

            Parameters
            ----------
            None
        """
        return np.copy(self.x)


class Rate(Protocol):
    """
    Rate function protocol opreator. Computes the rate of the operation given the state

    Return the rate associated with a reaction given the :class:`State` object.
    """
    
    def computerate(self, curstate:State)->float:
        ...

class SDEStepper(Protocol):
    """

    
    """
    def step(self, curstate:State)->tuple[NDArray, float]:
        ...


class NaiveUrnStepper:
    """
    Simulate a stochastic sytem given the rates, as an urn model.
    """
    def __init__(
            self,
            rates: list[Rate],
            increments: list[NDArray],
            ncompartments:int,
            dt:float,
    )-> None:
        """
            Assume that the increments are not a functions of the state,
            meaning that the same reaction will give the same change in the
            compartment states.
        """
        for inc in increments:

            if inc.ndim!=1:
                raise ValueError

            if inc.shape[0]!=ncompartments:
                raise ValueError

        self.rates = rates
        self.increments = increments
        self.dt = dt


    def step(self, curstate:State)->tuple[NDArray, float]:
        """
            Do the naive stepping, return the increment related to the step and the time increment

            Parameters
            ----------
                curstate: State
                    The :class:`State` object representing the current state object.
            Returns
            -------
            increment

            dt

        """
        randomnum = np.random.uniform(0,1)
        # The length of the array is the number of rections plus 1 (for no reaction happening)
        probs = np.zeros((len(self.rates)+1))

        for i, ratefunc in enumerate(self.rates):
            probs[i] = ratefunc.computerate(curstate=curstate)*self.dt

        
        if np.all(probs==0):
            #return no reaction increment
#            print('All Zero')
            increment = np.zeros_like(curstate.x)
            return increment, self.dt

        # print("Probability: ", np.sum(probs))
        #assert np.isclose(np.sum(probs), 1.0)
        # TODO: Try to implement a binary search here, currently using a linear search.
        # Is the performance for low array size 
        # probability of no reaction happening
        probnoreact = 1-np.sum(probs)
        probs[-1] = probnoreact 
        probssum = np.cumsum(probs)
        assert np.isclose(probssum[-1], 1.0)
#       print(probs)
#       print(f"The no reaction probability is {probnoreact}")
#       print(probssum)
#       print(f"The random number {randomnum
        if probssum[0]>randomnum:
            # print(f"Increments: {self.increments[0]}")
            return (self.increments[0], self.dt)
        for i in range(1,probssum.shape[0]-1):
            if probssum[i-1]<randomnum and probssum[i]>=randomnum:
                # print(f"Increments: {self.increments[i]}")
                return (self.increments[i], self.dt)

        increment = np.zeros_like(curstate.x)
        return increment, self.dt

class Simulate:

    def __init__(
        self,
        initstate:State,
        stepper:SDEStepper,
    )->None:
        """
            Initialise the simulator class to simulate and store the the values for the compartmental model

            Parameters
            ----------
                state: :class:`State` object representing the initial state of the compartments.
                stepper: :class:`SDEStepper` object representing how the stepper would work
        """
        self.state = initstate
        self.stepper = stepper

    def simulate_and_return(
        self,
        steps:int,
        endcondition =None
    )->tuple[NDArray, NDArray]:
        """
            Simulate and run a model for `steps` number of steps.
            
            Parameters
            ---------- 
                steps: int
                    The number of steps to simulate for.
        """
        times = np.zeros(steps+1)
        traj = np.zeros( (self.state.get_compartments, steps+1) )
        traj[:,0] = self.state.get_state_copy()
        for i in range(1,steps+1):
            x, dt = self.stepper.step(self.state)
            if dt == -1:
                raise ValueError
            self.state.update_state(x+self.state.x)
            traj[:,i] = self.state.x
            times[i] = times[i-1] +dt
            if endcondition is not None and np.all(endcondition == self.state.x):
                return (traj[:i], times[:i])

        return (traj, times)
        

if __name__ == '__main__':
    print("hello world")
