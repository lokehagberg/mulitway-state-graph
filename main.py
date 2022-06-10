

import math as ma
import numpy as np
import matplotlib.pyplot as plt 
from numpy.random import choice


class Configuration_path:
    
    def __init__(self,
                 config_length:         int = 100,
                 in_sum:                int = 100,
                 path_length:           int = 5,
                 start_configuration:   int = 0):
        self.config_length = config_length
        self.in_sum = in_sum
        self.path_length = path_length 
        self.start_configuration = start_configuration

    
    def partition(self, in_sum, length, depth=0):
        if length == depth:
            return [[]]
        return [
            item + [i]
            for i in range(in_sum+1)
            for item in self.partition(in_sum-i, length, depth=depth+1)
            ]


#TODO put into the class
configuration_path = Configuration_path(config_length=4, in_sum=4, path_length=6, start_configuration=0)
lst = [[configuration_path.in_sum-sum(p)] + p for p in configuration_path.partition(configuration_path.in_sum, configuration_path.config_length-1)]
lst_indx = np.arange(len(lst))


similarity = [] 
for i in range(len(lst)):
    similarity_local = []
    for j in range(len(lst)):
        if (i == j):
            similarity_local.append(0.0)
        else:
            numerator = np.dot(lst[i], lst[j])
            proto_denom1 = 0
            proto_denom2 = 0
            for k in range(configuration_path.config_length):
                proto_denom1 = proto_denom1 + lst[i][k]**2
                proto_denom2 = proto_denom2 + lst[j][k]**2
            denominator = (ma.sqrt(proto_denom1) * ma.sqrt(proto_denom2))
            similarity_local.append(numerator/denominator)
    similarity.append(similarity_local)


for i in range(len(lst)):
    difference_to_one = 1
    counter = 0
    summation = 0
    for j in range(len(lst)):
        summation = summation + similarity[i][j]
    for j in range(len(lst)):
        similarity[i][j] = (similarity[i][j]/summation)
        similarity[i][j] = round(similarity[i][j], 4)
        difference_to_one = difference_to_one - similarity[i][j]
        if (similarity[i][j] > 0):
            counter = counter + 1 
    for j in range(len(lst)):
        if (similarity[i][j] > 0):
            similarity[i][j] = similarity[i][j] + (difference_to_one/counter)

for i in range(len(lst)):
    similarity[i][configuration_path.start_configuration] = 0.0
probability_dist = similarity[configuration_path.start_configuration]


path = []
path_indx = []
for i in range(0,configuration_path.path_length):
    
    draw = choice(lst_indx, 1, p=probability_dist)
    path.append(lst[draw[0]])
    path_indx.append(draw[0])
    
    for k in range(len(lst)):
        similarity[k][draw[0]] = 0.0
        temp_counter = 0
        temp_difference_to_one = 1
        for j in range(len(lst)):
            if (similarity[k][j] > 0):
                temp_counter = temp_counter + 1
                temp_difference_to_one = temp_difference_to_one - similarity[k][j]
        for j in range(len(lst)):
            if (similarity[k][j] > 0):
                similarity[k][j] = similarity[k][j] + (temp_difference_to_one/temp_counter)
    
    probability_dist = similarity[draw[0]]
    

print(path_indx)
print(path)