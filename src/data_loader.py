#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:58:55 2018

@author: raghuvansh
"""

import numpy as np

class DataLoader(object):
    
    def __init__(self):
        self.curr_directory = '/home/raghuvansh/DL/MNIST/data/ubyte/'
        
    def load_data(self,
                  train_images='train-images-idx3-ubyte', 
                  test_images='t10k-images-idx3-ubyte', 
                  train_labels='train-labels-idx1-ubyte', 
                  test_labels='t10k-labels-idx1-ubyte',
                  train_count=60000,
                  test_count=10000):
        
        training_images = open(self.curr_directory + train_images, 'rb')
        testing_images = open(self.curr_directory + test_images, 'rb')
        training_labels = open(self.curr_directory + train_labels, 'rb')
        testing_labels = open(self.curr_directory + test_labels, 'rb')
        
        training_images.read(16)
        testing_images.read(16)
        training_labels.read(8)
        testing_labels.read(8)
        
        images = []
        labels = []
        for i in range(train_count):
            image = []
            for j in range(784):
                image.append(ord(training_images.read(1)))
            
            images.append(np.reshape(np.array(image), (784,1)))
            labels.append(vectorized_result(ord(training_labels.read(1))))
        
        training_data = list(zip(images, labels))
                    
        images = []
        labels = []
        for i in range(test_count):
            image = []
            for j in range(784):
                image.append(ord(testing_images.read(1)))
                
            images.append(np.reshape(np.array(image), (784,1)))
            labels.append(ord(testing_labels.read(1)))
            
        testing_data = list(zip(images, labels))
        
        return (training_data, testing_data)

def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1
    
    return e