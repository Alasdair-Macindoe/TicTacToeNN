"""
Reads in the JSON for training my Neural Network
"""
import json
import pickle
from neuralnetwork import NeuralNetwork, Layer, Neuron
from game import Game
import graph
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, help='The Network to be loaded in')
parser.add_argument('--pickle', type=int, default=1, help='Is the network a Pickle file?')
parser.add_argument('--graph', type=int, default=0, help='Is the file loaded actually graph data? (Pickle only)')
parser.add_argument('--game', type=int, default=0, help='Do you want to play a game?')
parser.add_argument('--training', type=str, default=None, help='Do you wish to give a training set? (Pickle only)')
parser.add_argument('--train', type=int, default=0, help='Do you wish to train this network?')
parser.add_argument('--random', type=int, default=0, help='Play the network against random')

args = parser.parse_args()

#Graph data
if args.graph:
    graph.graph(args.file)
else:
    #Create network
    network = None
    g = Game()
    if args.pickle:
        with open(args.file, 'rb') as f:
            network = pickle.load(f)
            g._nn1 = network
    else:
        with open(args.file, 'r') as f:
            text = json.loads(f.read())
            network = NeuralNetwork(text['layers'], text['alphas'])
            g._nn1 = network

    #Training set
    if args.train:
        ex = g.create_examples(args.train)
        with open('data/user-made.pickle', 'wb') as f:
            pickle.dump(ex, f)
            print("Written to: {}".format('data/user-made.pickle'))

    #If trains
    if args.training != None:
        with open(args.training, 'rb') as f:
            g._train(pickle.load(f))

    #If play
    if args.game:
        g.start_game()

    if args.random:
        g.test_against_random(args.random)
