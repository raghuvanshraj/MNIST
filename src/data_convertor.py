#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 23:59:43 2018

@author: raghuvansh
"""

# 60000 training
# 10000 testing

def convert(imgf, labelf, n):
    curr_directory = '/home/raghuvansh/Desktop/DL/MNIST/data/ubyte/'
    f = open(curr_directory + imgf, "rb")
    l = open(curr_directory + labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    f.close()
    l.close()
    
    return images