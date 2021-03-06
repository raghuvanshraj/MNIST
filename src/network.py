#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 19:03:48 2018

@author: raghuvansh
"""

import numpy as np
from random import shuffle
import pickle

class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        
    def feedforward(self, x):
        a = x
        zs = []
        activations = [x]
        for w,b in zip(self.weights, self.biases):
            z = np.dot(w,a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)
            
        cache = (zs, activations)
        
        return (a, cache)
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data: n_test = len(test_data)
        n_train = len(training_data)
        for j in range(epochs):
            shuffle(training_data)
            mini_batches = [
                    training_data[k:k + mini_batch_size]
                    for k in range(0, n_train, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaluate(test_data), n_test))
            else:
                print('Epoch {0}'.format(j))
                
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            a, cache = self.feedforward(x)
            delta_nabla_b, delta_nabla_w = self.backprop(cache, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [w - (eta/len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, cache, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        zs, activations = cache
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(self.weights[-l+1].T, delta) * sigmoid_derivative(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            
        return (nabla_b, nabla_w)
        
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)[0]), y) for (x,y) in test_data]
        return sum(int(x == y) for (x,y) in test_results)
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    def predict(self, x):
        vectorized_predictions = [self.feedforward(sample)[0] for sample in x]
        predictions = [np.argmax(prediction) 
                        for prediction in vectorized_predictions]
            
        return predictions  
    
    def save_network(self, filename):
        matrix = (self.weights, self.biases)
        pickle_out = open('/home/raghuvansh/DL/MNIST/saved_network/' + filename, 'wb')
        pickle.dump(matrix, pickle_out)
        pickle_out.close()
        
    def load_network(self, filename):
        try:
            pickle_in = open('/home/raghuvansh/DL/MNIST/saved_network/' + filename, 'rb')
        except FileNotFoundError:
            print('no saved networks available')
            return
        
        self.weights, self.biases = pickle.load(pickle_in)
        pickle_in.close()
        
def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))