import ising2D as isi
import time
import numpy as np

p = print

t = time.time()

iss = isi.IsingSquare(order=4)

output = open('training1.csv', 'w')

for i in np.arange(5):
    spins = iss.basicIter(temp=5, iters=100)
    spin0 = spins[0]
    spin1 = spins[1]
    print(spins[0].tolist())

    output.write("" + str(spins[0].tolist()).replace("[[","[").replace("]]","]").replace("[","").replace("]","") + ",")
    output.write("" + str(spins[1].tolist()).replace("[[","[").replace("]]","]").replace("[","").replace("]","") + "\n")

p("Time:",np.round(time.time() - t), "s")