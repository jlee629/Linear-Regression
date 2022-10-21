# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:22:11 2022

@author: Jungyu Lee, 301236221
Exercise#1: Sampling and noise
"""
# a - numpy 100 samples 
import numpy as np
x=np.random.uniform(-100,101,100)
print(x)

# b - seed
np.random.seed(21)

# c - y data
y = 12*x - 4

# d - matplotlib
import matplotlib.pyplot as plt
plt.plot(x,y,'ro', alpha=0.5)
plt.title('sampling & noise: before injecting noise')
plt.xlabel('sampling')
plt.ylabel('12 * x - 4')

# e - add noise
y = 12 * x - 4 + np.random.normal(-100, 101, 100)

# f - reproduce the plot & change the title
plt.plot(x,y,'ro', alpha=0.5)
plt.title('sampling & noise: after injecting noise')
plt.xlabel('sampling')
plt.ylabel('12 * x - 4 + noise')

# g - analysis
