import math as ma
import numpy as np
# import matplotlib.pyplot as plt 
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


#TODO put into the class, enter starting configuration if it is used
config_length = 3
configuration_path = Configuration_path(config_length=config_length, in_sum=3, path_length=5, start_configuration=3)
configurations = [np.array([configuration_path.in_sum-sum(p)] + p) for p in configuration_path.partition(configuration_path.in_sum, configuration_path.config_length-1)]
configuration_indicies = np.arange(len(configurations))

#Initialization of the Matrix

probabilityMatrix = [[1/len(configuration_indicies) for i in range(len(configuration_indicies))] for j in range(len(configuration_indicies))]
weight = 0.9
highest_entropy_configuration = np.array([0.5 for i in range(config_length)])
for i in range(len(configurations)):

    pertubation_similarity = 0
    entropy_similarity = 0
    for z in range(len(configurations)):
        if (z!=i):
            pertubation_similarity += 2 / (np.sum(np.abs(configurations[i] - configurations[z])))
            entropy_similarity += 2 / (np.sum(np.abs(highest_entropy_configuration - configurations[z])))

    for j in range(len(configurations)):
        if (j!=i):
            probabilityMatrix[i][j] = (weight*(2 / np.sum(np.abs(configurations[i] - configurations[j]))))/pertubation_similarity + (1 - weight)*((2/np.sum(np.abs(highest_entropy_configuration - configurations[j])))/entropy_similarity)
        else:
            probabilityMatrix[i][j] = 0



#Likhetsmått: |v1 - v2|/2. Hög likhet -> Högre sannolikhet
# likhetsmått = np.abs(np.array(configurations[0]) - np.array([configurations[1]]))/2
#Entropimått: [3,0,0] = Hög entropi, [1,1,1] = Låg entropi. Min[Sum[Ceil[v1/edges]], Sum[Ceil[v2/edges]]] Hög entropi -> Högre sannolikhet



current_configuration = 0 #Initial Condition
for t in range(9):
    current_configuration = choice(configuration_indicies, p=probabilityMatrix[current_configuration])
    
    for i in range(len(probabilityMatrix)):

        first_nonzero_probability = next((x for x in probabilityMatrix[i] if x != 0), None)
        #Normalizes the probabilities of going from one configuration to another since going back to itself is 0, we do not want loops.
        F = 1 / (1 - probabilityMatrix[i][current_configuration])
        
        for j in range(len(probabilityMatrix[i])):
            if (j == current_configuration):
                probabilityMatrix[i][j] = 0
            else:
                probabilityMatrix[i][j] *= F
        

    print(configurations[current_configuration])
    