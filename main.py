from copy import deepcopy
import math as ma
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
config_length = 4
in_sum=4
weight = 0.9
timesteps = 10

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


for i in range(len(configuration_history_trimmed)):

    if (i != 0):
        
        con1 = deepcopy(configuration_history_trimmed[i-1])
        con2 = deepcopy(configuration_history_trimmed[i])

        double_edge = []
        #set adr = 0 and bdr = 0 if you want the convention of former nodes to diverge less.
        bdr = choice(range(len(con1)))

        for m in range(len(con1)):
            adr = choice(range(len(con2)))            
            mbdr = ((m+bdr) % len(con1))
            
            for n in range(len(con2)):
                
                madr = ((m+n+adr)%len(con2))
                
                if((con1[mbdr] > 0) and (con2[madr] >= con1[mbdr]) and ([mbdr,madr] not in double_edge)):
                    G.add_edge((i-1)*10 + mbdr, i*10 + madr)
                    con2[madr] = con2[madr] - con1[mbdr]
                    con1[mbdr] = 0
                    double_edge.append([mbdr,madr])
                    
                
                elif((con1[mbdr] > 0) and (con1[mbdr] > con2[madr]) and ([mbdr,madr] not in double_edge)):
                    G.add_edge((i-1)*10 + mbdr, i*10 + madr)
                    con1[mbdr] = con1[mbdr] - con2[madr]
                    con2[madr] = 0
                    double_edge.append([mbdr,madr])


print(G)
nx.draw(G, pos, with_labels=True)
plt.show()  

# We have assumed that we have the following linear principles: 
# 1. because the change frequency is higher with some experiences, similarity is higher than not.
# 2. experiences of a kind stays of that kind (there are multiple multiway state graphs that are coordinated). 
# 3. experiences are more scrambled in the future as entropy increases over known universal time, 
# and it adds linearly to similarity. 
# What are the possible spatial graphs likely to look like? 

# By Life, gravity and the second law of thermodynamics - Charles H. Lineweaver, Chas A. Egan
# and Entropy in an Expanding Universe - Steven Frautschi, entropy increases in a linear way (except possibly after
# a large number of time-steps). Frequency statistically have a large distribution at most times, 
# making the similarity measurement motivated. 
