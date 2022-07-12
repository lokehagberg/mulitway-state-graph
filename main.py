import math as ma
from this import d
import numpy as np
# import matplotlib.pyplot as plt 
from numpy.random import choice
import matplotlib.pyplot as plt
import networkx as nx

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
in_sum=3
weight = 0.9
timesteps = 5

configuration_path = Configuration_path(config_length=config_length, in_sum=in_sum, path_length=5, start_configuration=3)
configurations = [np.array([configuration_path.in_sum-sum(p)] + p) for p in configuration_path.partition(configuration_path.in_sum, configuration_path.config_length-1)]
configurations *= in_sum
configuration_indicies = np.arange(len(configurations))


#Initialization of the Matrix
probabilityMatrix = [[1/len(configuration_indicies) for i in range(len(configuration_indicies))] for j in range(len(configuration_indicies))]
highest_entropy_configuration = np.array([0.5 for i in range(config_length)])
for k in range(len(configurations)):

    pertubation_similarity = 0
    entropy_similarity = 0
    for z in range(len(configurations)):
        configuration_difference = configurations[k] - configurations[z]
        if (configuration_difference.all(0)):
            
            #Pertubation similaruty tells us how similar two configurations are.
            pertubation_similarity += 2 / (np.sum(np.abs(configuration_difference)))
            
            #Example: [3,0,0] is a low entropy configuration, [1,1,1] is a high entropy configuration.
            entropy_similarity += 2 / (np.sum(np.abs(highest_entropy_configuration - configurations[z])))

    for m in range(len(configurations)):

        configuration_difference = configurations[k] - configurations[m] 

        if (not configuration_difference.all(0)):
            probabilityMatrix[k][m] = 0

        else:
            similarity_component = (weight*(2 / np.sum(np.abs(configuration_difference))))/pertubation_similarity
            entropy_component = (1 - weight)*((2/np.sum(np.abs(highest_entropy_configuration - configurations[m])))/entropy_similarity)

            probabilityMatrix[k][m] = similarity_component + entropy_component


current_configuration = 0 #Initial Condition
actualized_configuration_path = []
for t in range(timesteps):
    current_configuration = choice(configuration_indicies, p=probabilityMatrix[current_configuration])
    actualized_configuration_path.append(configurations[current_configuration])

    for k in range(len(probabilityMatrix)):

        first_nonzero_probability = next((x for x in probabilityMatrix[k] if x != 0), None)
        #Normalizes the probabilities of going from one configuration to another since going back to itself is 0, we do not want loops.
        F = 1 / (1 - probabilityMatrix[k][current_configuration])
        
        for m in range(len(probabilityMatrix[k])):
            if (m == current_configuration):
                probabilityMatrix[k][m] = 0
            else:
                probabilityMatrix[k][m] *= F
        

    
configuration_history = np.asarray(actualized_configuration_path)  
print(configuration_history)
configuration_history_trimmed = []  
for configuration in configuration_history:
    configuration_history_trimmed.append(np.delete(configuration, np.where(configuration == 0)))


print(configuration_history_trimmed)

G = nx.Graph()
pos = {}
for i in range(len(configuration_history_trimmed)):
    for j in range(len(configuration_history_trimmed[i])):
        
        G.add_node(i*10 + j)
        pos.update([(i*10 + j,[i,j])])

        diff = len(configuration_history_trimmed[i]) - len(configuration_history_trimmed[i-1])
        config_diff = []
        if diff > 0:
            zer = np.array([0 for i in range(diff)])
            newcon = np.concatenate((configuration_history_trimmed[i-1], zer))
            config_diff = configuration_history_trimmed[i] - newcon
        
        elif diff < 0:
            zer = np.array([0 for i in range(diff)])
            newcon = np.concatenate((configuration_history_trimmed[i], zer))
            config_diff = newcon - configuration_history_trimmed[i]
        else: 
            config_diff = configuration_history_trimmed[i] - configuration_history_trimmed[i-1]

            # while config_diff.any(1):
            #     for i in range(len(config_length)):
            #         if (i != 0):
            #             G.add_edge((i-1)*10 + j, i*10 + i)
        if (i != 0):

            m = 0
            while m in range(len(config_diff)):
                if (config_diff[m] == 0):
                    m+=1
                else:
                    if (config_diff[m] < 0):
                        k = 0
                        while k in range(len(config_diff)):
                            if (config_diff[k] > 0):
                                G.add_edge((i-1)*10 + m, i*10 + k)
                                config_diff[m] = config_diff[m] + 1
                                config_diff[k] = config_diff[k] - 1
                                m+=1
                                k+=len(config_diff) + 1
                    else:
                        m+=1

    G.add_edge(20,30)

print(G)
# print(pos)
# G.add_node(12)
# G.add_node(21)
# G.add_node(22)
# G.add_edge(11,21)
# G.add_edge(12,22)
nx.draw(G, pos, with_labels=True)
plt.show()  

# We have assumed that we have the following linear principles: because the change frequency is 
# higher with some experiences, similarity is higher than not, entropy increases 
# over known universal time, experiences of a kind stays of that kind (there are multiple
# multiway state graphs that are coordinated). Looking at the configuration path, we will not get information about which
# ones change frequently and which ones that do not, it gives us a set of 
# possibilities as a function of |V|, |E|, the number of time steps and the trade-off 
# weight between similarity and entropy (that is weighted toward similarity). 
# What then is the linear possibility set like (check frequency and spatial graphs)?  
# By Life, gravity and the second law of thermodynamics - Charles H. Lineweaver, Chas A. Egan
# and Entropy in an Expanding Universe - Steven Frautschi, entropy increases in a linear way (except possibly after
# a large number of time-steps). Frequency statistically have a large distribution at most times, 
# making the similarity measurement motivated. 
