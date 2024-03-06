import matplotlib.pyplot as plt 
import numpy as np
import time

from game import BoardState

# test = BoardState()

# player_0 = True
# i_row, i_col = zip(*test.decode_state)

# pos = plt.scatter(i_row,i_col)

# plt.grid()
# plt.show()

plt.axis([0, 10, 0, 1])

for i in range(100):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

plt.show()