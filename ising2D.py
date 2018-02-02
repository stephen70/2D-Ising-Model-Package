import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import colors
import copy

class IsingSquare:

    # initialise a spin lattice and populate with random spins
    def __init__(self, order, interactionVal=1, magMoment=1):

        if order < 3:
            raise ValueError('Order number needs to be greater than 2.')

        self.temp = 0.0
        self.beta = 0.0
        self.boltzmann = 1.38064852 * (10 ** -23)
        self.order = order

        self.J = float(interactionVal)
        self.h = float(magMoment)

        self.magList = []
        self.specHeatList = []
        self.energyList = []
        self.suscepList = []

        self.spins = []
        self.__resetSpins()

    # reset the spin lattice to a random configuration
    def __resetSpins(self):
        vals = np.array([-1, 1])

        self.spins = np.random.choice(vals, size=(self.order, self.order))

    # returns an array of an atom's 4 nearest neighbours
    def __neighbours(self, row, col):

        return np.asarray([self.spins[row][col - 1],                 #left
                           self.spins[row][(col + 1) % self.order],  #right
                           self.spins[row - 1][col],                 #up
                           self.spins[(row + 1) % self.order][col]]) #down

    # calculates the energy of a single atom, using the Hamiltonian
    def __singleEnergy(self, row, col):

        neighbours = self.__neighbours(row, col)
        selfSpin = self.spins[row][col]
        return self.J * selfSpin * np.sum(np.sum(neighbours)) - self.h * selfSpin

    # calculates the magnitude of the entire energy of the lattice
    def __totalEnergy(self):

        energy = 0.0
        for i in np.arange(self.order):
            for j in np.arange(self.order):
                energy += self.__singleEnergy(i, j)

        # to avoid counting pairs twice, divide by two
        # divide by maximum possible energy to normalise
        return math.fabs(energy) / (self.order * self.order * (-4 * self.J - self.h) )

    # calculates the magnitude of the residual magnetic spin of the lattice
    # normalise by dividing by order of lattice squared
    def __totalMag(self):
        return math.fabs(np.sum(np.sum(self.spins)) / (self.order ** 2))

    def __specHeat(self, energy, energySquared, temp):
        return (energySquared - energy ** 2) * (1 / (self.order * self.order * 2 * temp * temp))

    def __suscep(self, mag, magSquared, temp):
        return self.J * (magSquared - mag ** 2) * (1 / (self.order * self.order * 2 * temp))
    # attempts to flip a random spin using the metropolis algorithm and the Boltzmann distribution
    def __tryFlip(self, row, col):
        # energy change = -2 * E_initial
        # so accept change if E_initial >= 0

        energy = self.__singleEnergy(row, col)

        if energy <= 0 or np.random.random() <= math.exp(-self.beta * 2 * energy):
            self.spins[row][col] *= -1

    # closes plot window
    def __close_event(self):
        plt.close()  # timer calls this function after 3 seconds and closes the window

    # plots a meshgrid of the initial and final spin lattices
    def __plotStartEndSpins(self, spinsList, iters=1000000):

        cmap = colors.ListedColormap(['red', 'yellow'])
        bounds = [-1, 0, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.subplots(nrows=1, ncols=2)
        plt.tight_layout()

        plt.subplot(1,2,1)
        plt.imshow(spinsList[0], cmap=cmap, norm=norm)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Initial Configuration')

        plt.subplot(1, 2, 2)
        plt.imshow(spinsList[1], cmap=cmap, norm=norm)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Final Configuration')

        title = "Temperature (J/K_B) = {0}, J = {1}, h = {2}, Iterations = {3}".format(self.temp, self.J, self.h, iters) + "\n" + "Order: {0} x {1}".format(self.order, self.order)
        plt.suptitle(title)

        # timer = fig.canvas.new_timer(
        #     interval=graphInterval)  # creating a timer object and setting an interval of 3000 milliseconds
        # timer.add_callback(self.__close_event)
        # timer.start()
        plt.show()

    # simulates the lattice at a constant temperature temp, for iters iterations, plots the resulting lattices, and returns the spin configurations
    def basicIter(self, iters=1000000, temp=1, plot=False):

        self.__resetSpins()
        spinsList = [copy.deepcopy(self.spins)]

        self.temp = temp
        self.beta = 1.0 / self.temp

        for i in np.arange(iters + 1):
            row, col = np.random.randint(self.order), np.random.randint(self.order)
            self.__tryFlip(row, col)

        spinsList.append(self.spins)

        if plot:
            self.__plotStartEndSpins(spinsList, iters)
        else:
            for i in np.arange(len(spinsList[0])):
                spinsList[0][i] = np.asarray(spinsList[0][i])

            for i in np.arange(len(spinsList[1])):
                spinsList[1][i] = np.asarray(spinsList[1][i])

            spinsList = np.array(spinsList)

        return spinsList

    # simulates the lattice oer a temperature range tempRange, with itersPerTemp iterations per temperature
    # plotProperties: plot the residual spin, total energy, susceptibility and specific heat
    def tempRangeIter(self, tempRange=np.arange(start=0.8, stop=3.2, step=0.05), itersPerTemp=100000, plotProperties=False):

        self.__resetSpins()

        # store the averages here
        energyList = []
        magList = []
        specHeatList = []
        suscepList = []

        for temp in tempRange:
            self.beta = 1.0 / temp
            print("Calculating temp:", temp)

            # allow to reach equilibrium
            for i in np.arange(itersPerTemp + 1):
                row, col = np.random.randint(0, self.order), np.random.randint(0, self.order)
                self.__tryFlip(row, col)

            #do a further thousand iterations to get average, and every hundred iterations, store the properties
            if plotProperties:

                #store the values used to calculate averages here
                magListEquilib = []
                energyListEquilib = []

                for i in np.arange(500000):

                    if i % 5000 == 0:

                        energy = self.__totalEnergy()
                        mag = self.__totalMag()

                        energyListEquilib.append(energy)
                        magListEquilib.append(mag)

                    row, col = np.random.randint(0, self.order), np.random.randint(0, self.order)
                    self.__tryFlip(row, col)

                energyAvg = np.average(energyListEquilib)
                energySquaredAvg = np.average(np.square(energyListEquilib))
                magAvg = np.average(magListEquilib)
                magSquaredAvg = np.average(np.square(magListEquilib))

                energyList.append(energyAvg)
                magList.append(magAvg)
                specHeatList.append(self.__specHeat(energyAvg, energySquaredAvg, temp))
                suscepList.append(self.__suscep(magAvg, magSquaredAvg, temp))

            # reset the spins for the next temperature
            self.__resetSpins()

        if plotProperties:

            plt.tight_layout()

            plt.subplot(2, 2, 1)
            plt.plot(tempRange, energyList)
            plt.title("Total Energy")
            plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])

            plt.subplot(2, 2, 2)
            plt.plot(tempRange, magList)
            plt.title("Residual Spin")
            plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(tempRange, specHeatList)
            plt.title("Specific Heat Capacity")
            plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(tempRange, suscepList)
            plt.title("Susceptibility")
            plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.show()

class IsingTriangle:

    # initialise a spin lattice and populate with random spins
    def __init__(self, order, interactionVal=1, magMoment=1):

        if order < 4:
            raise ValueError('Order number needs to be greater than 3.')

        self.temp = 0.0
        self.beta = 0.0
        self.boltzmann = 1.38064852 * (10 ** -23)
        self.order = order
        self.J = float(interactionVal)
        self.h = float(magMoment)

        self.magList = []
        self.specHeatList = []
        self.energyList = []
        self.suscepList = []

        self.spins = []
        self.__resetSpins()

    # reset the spin lattice to a random configuration
    def __resetSpins(self):

        self.spins = []

        vals = np.array([-1, 1])

        for i in np.arange(self.order):
            self.spins.append(list(np.random.choice(vals, size=i + 1)))

        self.spins = np.array(self.spins)

    # returns an array of an atom's 6 nearest neighbours
    def __neighbours(self, row, col):

        # centre atoms
        if 1 < row < self.order - 1 and 0 < col < row:
            return np.asarray([self.spins[row - 1][col - 1],
                              self.spins[row - 1][col],

                              self.spins[row][col - 1],
                              self.spins[row][col + 1],

                              self.spins[row + 1][col],
                              self.spins[row + 1][col + 1]])

        # left side central
        elif 0 < row < self.order - 1 and col == 0:
            return np.asarray([self.spins[row - 1][0],
                              self.spins[row][1],

                              self.spins[row + 1][0],
                              self.spins[row + 1][1],

                              self.spins[row][row],
                              self.spins[row - 1][row - 1]])

        # right side central
        elif 0 < row < self.order - 1 and col == row:
            return np.asarray([self.spins[row - 1][row - 1],
                              self.spins[row - 1][0],

                              self.spins[row][row - 1],
                              self.spins[row][0],

                              self.spins[row + 1][row],
                              self.spins[row + 1][row + 1]])

        # bottom side central
        elif row == self.order - 1 and 0 < col < row:
            return np.asarray([self.spins[row - 1][col - 1],
                               self.spins[row - 1][col],

                               self.spins[row][col - 1],
                               self.spins[row][col + 1],

                               self.spins[0][0],
                               self.spins[0][0]])

        # very top
        elif row == 0:
            return np.asarray([self.spins[1][0],
                              self.spins[1][1],

                              self.spins[self.order - 1][0],
                              self.spins[self.order - 1][self.order - 1],

                              self.spins[self.order - 1][1],
                              self.spins[self.order - 1][self.order - 2]])
        # bottom left
        elif row == self.order - 1 and col == 0:
            return np.asarray([self.spins[row - 1][0],
                               self.spins[row - 1][row - 1],

                               self.spins[row][1],
                               self.spins[row][row],

                               self.spins[0][0],
                               self.spins[0][0]])

        # bottom right
        elif row == self.order - 1 and (col == row):
            return np.asarray([self.spins[row - 1][0],
                               self.spins[row - 1][row - 1],

                               self.spins[row][0],
                               self.spins[row][row - 1],

                               self.spins[0][0],
                               self.spins[0][0]])

    # calculates the energy of a single atom, using the Hamiltonian
    def __singleEnergy(self, row, col):

        neighbours = self.__neighbours(row, col)
        selfSpin = self.spins[row][col]
        return self.J * selfSpin * np.sum(np.sum(neighbours)) - self.h * selfSpin

    # calculates the magnitude of the entire energy of the lattice
    def __totalEnergy(self):

        energy = 0.0
        for i in np.arange(self.order):
            for j in np.arange(len(self.spins[i])):
                energy += self.__singleEnergy(i, j)

        # to avoid counting pairs twice, divide by two
        # divide by maximum possible energy to normalise
        return -math.fabs(energy / (self.order * self.order * (-6 * self.J - self.h)))

    # calculates the magnitude of the residual magnetic spin of the lattice
    # normalise by dividing by order of lattice squared
    def __totalMag(self):
        return math.fabs((np.sum(np.sum(self.spins)) * 2) / (self.order ** 2 + self.order))

    def __specHeat(self, energy, energySquared, temp):
        return (energySquared - energy ** 2) * (1 / (self.order * self.order * 2 * temp * temp))

    def __suscep(self, mag, magSquared, temp):
        return self.J * (magSquared - mag ** 2) * (1 / (self.order * self.order * 2 * temp))
    # attempts to flip a random spin using the metropolis algorithm and the Boltzmann distribution
    def __tryFlip(self, row, col):
        # energy change = -2 * E_initial
        # so accept change if E_initial <= 0

        energy = self.__singleEnergy(row, col)

        if energy <= 0 or np.random.random() <= math.exp(-self.beta * 2 * energy):
            self.spins[row][col] *= -1

    # closes plot window
    def __close_event(self):
        plt.close()  # timer calls this function after 3 seconds and closes the window

    # plots a meshgrid of the initial and final spin lattices
    def __plotStartEndSpins(self, spinsList, iters=1000000):

        for i in np.arange(self.order):

            for j in np.arange(self.order - i - 1):

                spinsList[0][i].append(8)
                spinsList[1][i].append(8)

        cmap = colors.ListedColormap(['red', 'yellow', 'white'])
        bounds = [-1, 0, 2, 10]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.subplots(nrows=1, ncols=2)
        plt.tight_layout()
        for i in np.arange(len(spinsList[0])):
            spinsList[0][i] = np.asarray(spinsList[0][i])

        for i in np.arange(len(spinsList[1])):
            spinsList[1][i] = np.asarray(spinsList[1][i])

        spinsList = np.array(spinsList)

        plt.subplot(1,2,1)
        plt.imshow(spinsList[0], cmap=cmap, norm=norm)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Initial Configuration')

        plt.subplot(1, 2, 2)
        plt.imshow(spinsList[1], cmap=cmap, norm=norm)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Final Configuration')

        title = "Temperature (J/K_B) = {0}, J = {1}, h = {2}, Iterations = {3}".format(self.temp, self.J, self.h, iters) + "\n" + "Order: {0}".format(self.order,)
        plt.suptitle(title)

        # timer = fig.canvas.new_timer(
        #     interval=graphInterval)  # creating a timer object and setting an interval of 3000 milliseconds
        # timer.add_callback(self.__close_event)
        # timer.start()
        plt.show()

    # simulates the lattice at a constant temperature temp, for iters iterations, and returns the spin configurations
    def basicIter(self, iters=1000000, temp=1, plot=False):

        self.__resetSpins()
        spinsList = [copy.deepcopy(self.spins)]

        self.temp = temp
        self.beta = 1.0 / self.temp

        for i in np.arange(iters + 1):
            row = np.random.randint(self.order)
            col = np.random.randint(row + 1)
            self.__tryFlip(row, col)

        spinsList.append(self.spins)

        if plot:
            self.__plotStartEndSpins(spinsList, iters)
        else:
            for i in np.arange(len(spinsList[0])):
                spinsList[0][i] = np.asarray(spinsList[0][i])

            for i in np.arange(len(spinsList[1])):
                spinsList[1][i] = np.asarray(spinsList[1][i])

            spinsList = np.array(spinsList)

        return spinsList

    # simulates the lattice oer a temperature range tempRange, with itersPerTemp iterations per temperature
    # plotProperties: plot the residual spin, total energy, susceptibility and specific heat
    def tempRangeIter(self, tempRange=np.arange(start=1, stop=5, step=0.2), itersPerTemp=100000, plotProperties=False):

        self.__resetSpins()

        # store the averages here
        energyList = []
        magList = []
        specHeatList = []
        suscepList = []

        for temp in tempRange:
            self.beta = 1.0 / temp
            print("Calculating temp:", temp)

            # allow to reach equilibrium
            for i in np.arange(itersPerTemp + 1):
                row = np.random.randint(self.order)
                col = np.random.randint(row + 1)
                self.__tryFlip(row, col)

            #do a further ten thousand iterations to get average, and every two hundred iterations, store the properties
            if plotProperties:

                #store the values used to calculate averages here
                magListEquilib = []
                energyListEquilib = []

                for i in np.arange(10000):

                    if i % 200 == 0:

                        energy = self.__totalEnergy()
                        mag = self.__totalMag()

                        energyListEquilib.append(energy)
                        magListEquilib.append(mag)

                    row = np.random.randint(self.order)
                    col = np.random.randint(row + 1)
                    self.__tryFlip(row, col)

                energyAvg = np.average(energyListEquilib)
                energySquaredAvg = np.average(np.square(energyListEquilib))
                magAvg = np.average(magListEquilib)
                magSquaredAvg = np.average(np.square(magListEquilib))

                energyList.append(energyAvg)
                magList.append(magAvg)
                specHeatList.append(self.__specHeat(energyAvg, energySquaredAvg, temp))
                suscepList.append(self.__suscep(magAvg, magSquaredAvg, temp))

            # reset the spins for the next temperature
            self.__resetSpins()

        if plotProperties:

            plt.tight_layout()

            plt.subplot(2, 2, 1)
            plt.plot(tempRange, energyList)
            plt.title("Total Energy")
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])

            plt.subplot(2, 2, 2)
            plt.plot(tempRange, magList)
            plt.title("Residual Spin")
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(tempRange, specHeatList)
            plt.title("Specific Heat Capacity")
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(tempRange, suscepList)
            plt.title("Susceptibility")
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.show()

class IsingHexagon:

    # initialise a spin lattice and populate with random spins
    def __init__(self, order, interactionVal=1, magMoment=1):

        if order < 2:
            raise ValueError('Order number needs to be greater than 3.')

        self.temp = 0.0
        self.beta = 0.0
        self.boltzmann = 1.38064852 * (10 ** -23)
        self.order = order
        self.J = float(interactionVal)
        self.h = float(magMoment)

        self.magList = []
        self.specHeatList = []
        self.energyList = []
        self.suscepList = []

        self.__resetSpins()

    # reset the spin lattice to a random configuration
    def __resetSpins(self):
        self.spins = []

        vals = np.array([1, -1])

        if self.order == 1:
            self.spins.append(list(np.random.choice(vals, size=2)))
            self.spins.append(list(np.random.choice(vals, size=2)))
            self.spins.append(list(np.random.choice(vals, size=2)))
            self.spins = np.array(self.spins)
            return

        # top layers
        iter = 2
        while self.order >= iter / 2.0 and not iter == 2 * self.order:
            self.spins.append(list(np.random.choice(vals, size=iter)))
            iter += 2
        # middle layers
        for i in np.arange(5 + 2 * (self.order - 2)):
            self.spins.append(list(np.random.choice(vals, size=2 * self.order)))
        # bottom layers
        iter = 2 * self.order - 2
        while iter > 0:
            self.spins.append(list(np.random.choice(vals, size= iter)))
            iter -= 2

    # returns the nearest neighbours for centre atoms, used in the main nearest neighbour function
    def centreReturn(self, left, row, col): # left is boolean, when true, use atom to left, otherwise, use atom to right
        if left:
            return np.asarray([self.spins[row - 1][col],
                          self.spins[row][col - 1],
                          self.spins[row + 1][col]])
        else:
            return np.asarray([self.spins[row - 1][col],
                              self.spins[row][col + 1],
                              self.spins[row + 1][col]])

    # returns an array of an atom's 3 nearest neighbours
    def __neighbours(self, row, col):
        # centre atoms
        if 1 < row < 4 * self.order - 3:
            if self.order - 1 < row < 3 * self.order - 1 and 0 < col < len(self.spins[row]) - 1:
                # handles centre atoms
                if col % 2 == 0:
                    if self.order % 2 == 0:
                        if row % 2 == 0:
                            return self.centreReturn(True, row, col)
                        else:
                            return self.centreReturn(False, row, col)
                    else:
                        if row % 2 == 0:
                            return self.centreReturn(False, row, col)
                        else:
                            return self.centreReturn(True, row, col)
                else:
                    if self.order % 2 == 0:
                        if row % 2 == 0:
                            return self.centreReturn(False, row, col)
                        else:
                            return self.centreReturn(True, row, col)
                    else:
                        if row % 2 == 0:
                            return self.centreReturn(True, row, col)
                        else:
                            return self.centreReturn(False, row, col)

            elif (row < self.order or row > 3 * self.order - 2) and 1 < col < len(self.spins[row]) - 2:
                # handles centre atoms
                if col % 2 == 0:
                    if self.order % 2 == 0:
                        if row % 2 == 0:
                            return self.centreReturn(True, row, col)
                        else:
                            return self.centreReturn(False, row, col)
                    else:
                        if row % 2 == 0:
                            return self.centreReturn(False, row, col)
                        else:
                            return self.centreReturn(True, row, col)
                else:
                    if self.order % 2 == 0:
                        if row % 2 == 0:
                            return self.centreReturn(False, row, col)
                        else:
                            return self.centreReturn(True, row, col)
                    else:
                        if row % 2 == 0:
                            return self.centreReturn(True, row, col)
                        else:
                            return self.centreReturn(False, row, col)
        # left
        if 0 < row < (4 * self.order - 2) and col < 2:
            if row % 2 == 0:
                return np.asarray([self.spins[row - 1][0],
                                   self.spins[row + 1][0],
                                   self.spins[row][len(self.spins[row]) - 1]])
            else:
                return np.asarray([self.spins[row][len(self.spins[row]) - 1],
                                  self.spins[row][1],
                                  self.spins[row + 1][0]])

        # right
        elif 0 < row < (4 * self.order - 2) and col > len(self.spins[row]) - 3:
            if row % 2 == 0:
                return np.asarray([self.spins[row - 1][0],
                                   self.spins[row + 1][0],
                                   self.spins[row][len(self.spins[row]) - 1]])
            else:
                return np.asarray([self.spins[row - 1][len(self.spins[row - 1]) - 1],
                                  self.spins[row][col - 1],
                                  self.spins[row][0]])

        # top
        elif row == 0:
            if col == 0:
                return np.asarray([self.spins[0][1],
                                   self.spins[1][1],
                                   self.spins[4 * self.order - 2][0]])
            else:
                return np.asarray([self.spins[0][0],
                                   self.spins[1][2],
                                   self.spins[4 * self.order - 2][1]])

        # bottom
        elif row == 4 * self.order - 2:
            if col == 0:
                return np.asarray([self.spins[row][1],
                                   self.spins[row - 1][1],
                                   self.spins[0][0]])
            else:
                return np.asarray([self.spins[row][0],
                                   self.spins[1][2],
                                   self.spins[0][1]])

    # calculates the energy of a single atom, using the Hamiltonian
    def __singleEnergy(self, row, col):

        neighbours = self.__neighbours(row, col)
        selfSpin = self.spins[row][col]
        return self.J * selfSpin * np.sum(np.sum(neighbours)) - self.h * selfSpin

    # calculates the magnitude of the entire energy of the lattice
    def __totalEnergy(self):

        energy = 0.0
        for i in np.arange(len(self.spins)):
            for j in np.arange(len(self.spins[i])):
                energy += self.__singleEnergy(i, j)
        # to avoid counting pairs twice, divide by two
        # divide by maximum possible energy to normalise
        return -math.fabs(energy / ((3 * self.J + self.h) * (6 * self.order * self.order))) #( * (-3 * self.J - self.h)

    # calculates the magnitude of the residual magnetic spin of the lattice
    # normalise by dividing by order of lattice squared
    def __totalMag(self):
        sum = 0
        for i in np.arange(len(self.spins)):
            sum += np.sum(self.spins[i])

        return math.fabs(float(sum) / (6 * self.order ** 2))

    def __specHeat(self, energy, energySquared, temp):
        return (energySquared - energy ** 2) * (1 / (self.order * self.order * 2 * temp * temp))

    def __suscep(self, mag, magSquared, temp):
        return self.J * (magSquared - mag ** 2) * (1 / (self.order * self.order * 2 * temp))
    # attempts to flip a random spin using the metropolis algorithm and the Boltzmann distribution
    def __tryFlip(self, row, col):
        # energy change = -2 * E_initial
        # so accept change if E_initial <= 0

        energy = self.__singleEnergy(row, col)

        if energy <= 0 or np.random.random() <= math.exp(-self.beta * 2 * energy):
            self.spins[row][col] *= -1

    # closes plot window
    def __close_event(self):
        plt.close()  # timer calls this function after 3 seconds and closes the window

    # plots a meshgrid of the initial and final spin lattices
    def __plotStartEndSpins(self, spinsList, iters=1000000):

        for i in np.arange(self.order):

            for j in np.arange(self.order - i - 1):

                spinsList[0][i].append(8)
                spinsList[1][i].append(8)

        cmap = colors.ListedColormap(['red', 'yellow', 'white'])
        bounds = [-1, 0, 2, 10]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.subplots(nrows=1, ncols=2)
        plt.tight_layout()
        for i in np.arange(len(spinsList[0])):
            spinsList[0][i] = np.asarray(spinsList[0][i])

        for i in np.arange(len(spinsList[1])):
            spinsList[1][i] = np.asarray(spinsList[1][i])

        spinsList = np.array(spinsList)

        plt.subplot(1,2,1)
        plt.imshow(spinsList[0], cmap=cmap, norm=norm)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Initial Configuration')

        plt.subplot(1, 2, 2)
        plt.imshow(spinsList[1], cmap=cmap, norm=norm)
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title('Final Configuration')

        title = "Temperature (J/K_B) = {0}, J = {1}, h = {2}, Iterations = {3}".format(self.temp, self.J, self.h, iters) + "\n" + "Order: {0}".format(self.order,)
        plt.suptitle(title)

        # timer = fig.canvas.new_timer(
        #     interval=graphInterval)  # creating a timer object and setting an interval of 3000 milliseconds
        # timer.add_callback(self.__close_event)
        # timer.start()
        plt.show()

    # simulates the lattice at a constant temperature temp, for iters iterations, and returns the spin configurations
    def basicIter(self, iters=1000000, temp=1, plot=False):

        self.__resetSpins()
        spinsList = [copy.deepcopy(self.spins)]

        self.temp = temp
        self.beta = 1.0 / self.temp

        for i in np.arange(iters + 1):
            row = np.random.randint(4 * self.order - 1)
            col = np.random.randint(len(self.spins[row]))
            self.__tryFlip(row, col)

        spinsList.append(self.spins)

        if plot:
            self.__plotStartEndSpins(spinsList, iters)
        else:
            for i in np.arange(len(spinsList[0])):
                spinsList[0][i] = np.asarray(spinsList[0][i])

            for i in np.arange(len(spinsList[1])):
                spinsList[1][i] = np.asarray(spinsList[1][i])

            spinsList = np.array(spinsList)

        return spinsList

    # simulates the lattice oer a temperature range tempRange, with itersPerTemp iterations per temperature
    # plotProperties: plot the residual spin, total energy, susceptibility and specific heat
    def tempRangeIter(self, tempRange=np.arange(start=1, stop=5, step=0.2), itersPerTemp=100000, plotProperties=False):

        self.__resetSpins()

        # store the averages here
        energyList = []
        magList = []
        specHeatList = []
        suscepList = []

        for temp in tempRange:
            self.beta = 1.0 / temp
            print("Calculating temp:", temp)

            # allow to reach equilibrium
            for i in np.arange(itersPerTemp + 1):
                row = np.random.randint(4 * self.order - 1)
                col = np.random.randint(len(self.spins[row]))
                self.__tryFlip(row, col)

            #do a further thousand iterations to get average, and every hundred iterations, store the properties
            if plotProperties:

                #store the values used to calculate averages here
                magListEquilib = []
                energyListEquilib = []

                for i in np.arange(20000):

                    if i % 400 == 0:

                        energy = self.__totalEnergy()
                        mag = self.__totalMag()

                        energyListEquilib.append(energy)
                        magListEquilib.append(mag)

                    row = np.random.randint(4 * self.order - 1)
                    col = np.random.randint(len(self.spins[row]))
                    self.__tryFlip(row, col)

                energyAvg = np.average(energyListEquilib)
                energySquaredAvg = np.average(np.square(energyListEquilib))
                magAvg = np.average(magListEquilib)
                magSquaredAvg = np.average(np.square(magListEquilib))

                energyList.append(energyAvg)
                magList.append(magAvg)
                specHeatList.append(self.__specHeat(energyAvg, energySquaredAvg, temp))
                suscepList.append(self.__suscep(magAvg, magSquaredAvg, temp))

            # reset the spins for the next temperature
            self.__resetSpins()

        if plotProperties:

            plt.tight_layout()

            plt.subplot(2, 2, 1)
            plt.plot(tempRange, energyList)
            plt.title("Total Energy")
            #plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])

            plt.subplot(2, 2, 2)
            plt.plot(tempRange, magList)
            plt.title("Residual Spin")
            #plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.subplot(2, 2, 3)
            plt.plot(tempRange, specHeatList)
            plt.title("Specific Heat Capacity")
            #plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.plot(tempRange, suscepList)
            plt.title("Susceptibility")
            #plt.axvline(x=2.269185, c='r', linestyle='--')
            plt.tick_params(axis="x", direction="in")
            plt.tick_params(axis="y", direction="in")
            plt.xlim(tempRange[0], tempRange[len(tempRange) - 1])
            plt.legend()

            plt.show()
