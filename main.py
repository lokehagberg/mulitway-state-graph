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
config_length = 4
in_sum=4
weight = 0.9
timesteps = 25

configuration_path = Configuration_path(config_length=config_length, in_sum=in_sum, path_length=5, start_configuration=3)
configurations = [np.array([configuration_path.in_sum-sum(p)] + p) for p in configuration_path.partition(configuration_path.in_sum, configuration_path.config_length-1)]
configurations *= in_sum
configuration_indicies = np.arange(len(configurations))


#Initialization of the Matrix
probabilityMatrix = [[1/len(configuration_indicies) for i in range(len(configuration_indicies))] for j in range(len(configuration_indicies))]
highest_entropy_configuration = np.array([0.5 for i in range(config_length)])
for i in range(len(configurations)):

    pertubation_similarity = 0
    entropy_similarity = 0
    for z in range(len(configurations)):
        configuration_difference = configurations[i] - configurations[z]
        if (configuration_difference.all(0)):
            
            #Pertubation similaruty tells us how similar two configurations are.
            pertubation_similarity += 2 / (np.sum(np.abs(configuration_difference)))
            
            #Example: [3,0,0] is a low entropy configuration, [1,1,1] is a high entropy configuration.
            entropy_similarity += 2 / (np.sum(np.abs(highest_entropy_configuration - configurations[z])))

    for j in range(len(configurations)):

        configuration_difference = configurations[i] - configurations[j] 

        if (not configuration_difference.all(0)):
            probabilityMatrix[i][j] = 0

        else:
            similarity_component = (weight*(2 / np.sum(np.abs(configuration_difference))))/pertubation_similarity
            entropy_component = (1 - weight)*((2/np.sum(np.abs(highest_entropy_configuration - configurations[j])))/entropy_similarity)

            probabilityMatrix[i][j] = similarity_component + entropy_component


current_configuration = 0 #Initial Condition
for t in range(timesteps):
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
    
# We have assumed that we have the following linear principles: because the change frequency is 
# higher with some experiences, similarity is higher than not, entropy increases 
# over known universal time, experiences of a kind stays of that kind (there are multiple
# multiway state graphs that are coordinated). Looking at the configuration path, we will not get information about which
# ones change frequently and which ones that do not, it gives us a set of 
# possibilities as a function of |V|, |E|, the number of time steps and the trade-off 
# weight between similarity and entropy (that is weighted toward similarity). 
# What then is the linear possibility set like (check frequency and spatial graphs)? and what 
# happens in the dynamic case, where the dynamics is linear (entropy's speed actually
# decreases with an expanding universe but does so linearly as the universe expands 
# with a constant speed uniformly) (the universe is less light-like, so similarity 
# should matter more, see compositional change)? 
