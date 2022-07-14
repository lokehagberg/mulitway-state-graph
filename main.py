from copy import deepcopy
import math as ma
import numpy as np
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
        
        nodes_last_timestep = deepcopy(configuration_history_trimmed[i-1])
        nodes = deepcopy(configuration_history_trimmed[i])

        double_edge = []
        #set adr = 0 and bdr = 0 if you want the convention of former nodes to diverge less.
        random_node_last_timestep = choice(range(len(nodes_last_timestep)))

        for m in range(len(nodes_last_timestep)):
            random_node = choice(range(len(nodes)))            
            edge_randomness_last_timestep = ((m+random_node_last_timestep) % len(nodes_last_timestep))
            
            for n in range(len(nodes)):
                
                edge_randomness = ((m+n+random_node)%len(nodes))
                
                if((nodes_last_timestep[edge_randomness_last_timestep] > 0) and (nodes[edge_randomness] >= nodes_last_timestep[edge_randomness_last_timestep]) and ([edge_randomness_last_timestep,edge_randomness] not in double_edge)):
                    G.add_edge((i-1)*10 + edge_randomness_last_timestep, i*10 + edge_randomness)
                    nodes[edge_randomness] = nodes[edge_randomness] - nodes_last_timestep[edge_randomness_last_timestep]
                    nodes_last_timestep[edge_randomness_last_timestep] = 0
                    double_edge.append([edge_randomness_last_timestep,edge_randomness])
                    
                
                elif((nodes_last_timestep[edge_randomness_last_timestep] > 0) and (nodes_last_timestep[edge_randomness_last_timestep] > nodes[edge_randomness]) and ([edge_randomness_last_timestep,edge_randomness] not in double_edge)):
                    G.add_edge((i-1)*10 + edge_randomness_last_timestep, i*10 + edge_randomness)
                    nodes_last_timestep[edge_randomness_last_timestep] = nodes_last_timestep[edge_randomness_last_timestep] - nodes[edge_randomness]
                    nodes[edge_randomness] = 0
                    double_edge.append([edge_randomness_last_timestep,edge_randomness])


print(G)
nx.draw(G, pos, with_labels=True)
plt.show()  

# We have assumed that we have the following linear principles: 
# 1. because the change frequency is higher with some experiences, similarity is higher than not.
# 2. experiences of a kind stays of that kind (there are multiple multiway state graphs that are coordinated). 
# 3. experiences are more scrambled in the future over known universal time (entropy),
# and it adds linearly to similarity. [This addition might be wrong.]

# By Life, gravity and the second law of thermodynamics - Charles H. Lineweaver, Chas A. Egan
# and Entropy in an Expanding Universe - Steven Frautschi, entropy increases in a linear way (except possibly after
# a large number of time-steps). Frequency statistically have a large distribution at most times, 
# making the similarity measurement motivated. 
