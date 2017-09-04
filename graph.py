"""
Graphs data
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

def graph(file):
    wins, loses, draws = 0, 0, 0

    with open(file, 'rb') as f:
        wins, loses, draws = pickle.load(f)

    plt.plot(wins)
    plt.plot(loses, c='r')
    plt.plot(draws)
    plt.legend(['Wins', 'Loses', 'Draws'])

    print("Win mean: {}".format(np.mean(wins)))
    print("Lose mean: {}".format(np.mean(loses)))
    print("Draw mean: {}".format(np.mean(draws)))

    plt.show()
