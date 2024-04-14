import numpy as np
import matplotlib.pyplot as plt
import sys

f = open("wolves.txt","r")

num_packs = int(sys.argv[1])
wolves_per_pack = int(sys.argv[2])

wolves = np.zeros((num_packs,wolves_per_pack,2))

for pack in range(num_packs):
    for wolf in range(wolves_per_pack):
        s = f.readline()
        s = s.split()
        try:
            s = list(map(float,s))
            wolves[pack,wolf] = s
        except:
            continue

fig,ax = plt.subplots()

for pack in range(num_packs):
    ax.scatter(wolves[pack,:,0],wolves[pack,:,1])

fig.savefig("hehe.png",dpi=800)