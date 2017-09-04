import pytest
from neuralnetwork import Neuron
import math


def test_weights_alpha_establish():
    n = Neuron(1, [1])
    assert n._alpha == 1
    assert n._weights == [1]

def test_weighted_sum_basic():
    n = Neuron(1, [1, 1, 1])
    assert n._weights == [1, 1, 1]
    assert n._input([1, 1, 1]) == 3

def test_weighted_sum_complex():
    n = Neuron(1, [1, 0.5, 0, 3])
    assert n._weights == [1, 0.5, 0, 3]
    assert n._input([1, 2, 3, 4]) == (1*1 + 0.5*2 + 0*3 + 3*4)

def test_step_a0_with_0():
    n = Neuron(0, [1])
    assert n._g(0) == 0.5

def test_step_a0_with_1():
    n = Neuron(0, [1])
    assert n._g(1) == 0.5

def test_step_a1_with_1():
    n = Neuron(1, [1])
    assert abs(n._g(1) - (1.0 / (1 + math.exp(-1)))) < 0.01

def test_results():
    n = Neuron(2, [0.5, 0.12, 1])
    assert abs(n.output([1, 2, 0.05]) - (1.0 / (1 + math.exp(-2*0.79)))) < 0.01
