An old Python package to simulate 2D Ising models using the Monte Carlo Metropolis algorithm.

Methods:

  IsingSquare/IsingTriangle/IsingHexagon(order, interactionVal=1, magMoment=1)
  
  Description:
    Initialises an Ising model with the corresponding shape and values.
  Parameters:
    order (int): the order of the model. For a square and triangle, order is the number of units along each side. For a hexagon, order is the number of hexagons from the centre to the edge, including the centre. For example, an order of 2 is a single hexagon surrounded by six more hexagons.
    interactionVal (float): the interaction value of the model. This is equal to the coefficient of the pairwise energies in the Hamiltonian.
    magMoment (float): the magnetic field value. This is equal to the coefficient of the singular spin in the Hamiltonian.
    
  basicIter(iters=1000000, temp=1, plot=False)
  
  Description:
    Iterates the model for a given number of iterations and plots the initial and final spin configuration.
  Parameters:
    iters (int): the number of iterations.
    temp (float): the temperature. Must be greater than zero.
    plot (Boolean): if true, plots initial and final spin configuration.
  
  tempRangeIter(self, tempRange=np.arange(start=0.8, stop=3.2, step=0.05), itersPerTemp=100000, plotProperties=False)
  
  Description:
    Iterates the model for a given number of iterations over a given temperature range, and plots the properties at each temperature.
  Parameters:
    tempRange (np.array): the range to iterate the temperature over.
    itersPerTemp (int): the number of iterations per temperature.
    plotProperties (Boolean): if true, plots the total energy, total magnetic spin, specific heat capacity and susceptibility at each temperature.
