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
configuration_path = Configuration_path(config_length=3, in_sum=3, path_length=5, start_configuration=3)
configurations = [[configuration_path.in_sum-sum(p)] + p for p in configuration_path.partition(configuration_path.in_sum, configuration_path.config_length-1)]
configuration_indicies = np.arange(len(configurations))

probabilityMatrix = [[1/len(configuration_indicies) for i in range(len(configuration_indicies))] for j in range(len(configuration_indicies))]
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
        

    print(probabilityMatrix)
#Likhetsmått: |v1 - v2|/2. Hög likhet -> Högre sannolikhet
#Entropimått: [3,0,0] = Hög entropi, [1,1,1] = Låg entropi. Min[Sum[Ceil[v1/edges]], Sum[Ceil[v2/edges]]] Hög entropi -> Högre sannolikhet




# runs = 100
# visited_data = []
# for z in range(runs):
#     similarity = [] 
#     for i in range(len(lst)):
#         similarity_local = []
#         for j in range(len(lst)):
#             if (i == j):
#                 similarity_local.append(0.0)
#             else:
#                 numerator = np.dot(lst[i], lst[j])
#                 proto_denom1 = 0
#                 proto_denom2 = 0
#                 for k in range(configuration_path.config_length):
#                     proto_denom1 = proto_denom1 + lst[i][k]**2
#                     proto_denom2 = proto_denom2 + lst[j][k]**2
#                 denominator = (ma.sqrt(proto_denom1) * ma.sqrt(proto_denom2))
#                 similarity_local.append(numerator/denominator)
#         similarity.append(similarity_local)


#     for i in range(len(lst)):
#         difference_to_one = 1
#         counter = 0
#         summation = 0
#         for j in range(len(lst)):
#             summation = summation + similarity[i][j]
#         for j in range(len(lst)):
#             similarity[i][j] = (similarity[i][j]/summation)
#             similarity[i][j] = round(similarity[i][j], 4)
#             difference_to_one = difference_to_one - similarity[i][j]
#             if (similarity[i][j] > 0):
#                 counter = counter + 1 
#         for j in range(len(lst)):
#             if (similarity[i][j] > 0):
#                 similarity[i][j] = similarity[i][j] + (difference_to_one/counter)

#     #Set starting value
#     for i in range(len(lst)):
#         similarity[i][configuration_path.start_configuration] = 0.0
#     probability_dist = similarity[configuration_path.start_configuration]

#     #Random starting value
#     #starting_con = choice(lst_indx, 1)
#     #for i in range(len(lst)):
#     #    similarity[i][starting_con[0]] = 0.0
#     #probability_dist = similarity[starting_con[0]]


#     path = []
#     path_indx = []
#     for i in range(0,configuration_path.path_length):
        
#         draw = choice(lst_indx, 1, p=probability_dist)
#         path.append(lst[draw[0]])
#         path_indx.append(draw[0])
        
#         for k in range(len(lst)):
#             similarity[k][draw[0]] = 0.0
#             temp_counter = 0
#             temp_difference_to_one = 1
#             for j in range(len(lst)):
#                 if (similarity[k][j] > 0):
#                     temp_counter = temp_counter + 1
#                     temp_difference_to_one = temp_difference_to_one - similarity[k][j]
#             for j in range(len(lst)):
#                 if (similarity[k][j] > 0):
#                     similarity[k][j] = similarity[k][j] + (temp_difference_to_one/temp_counter)
        
#         probability_dist = similarity[draw[0]]
    
#     visited_data.append(path_indx)
    

# flat_list = [x for xs in visited_data for x in xs]
# configuration_amount = []
# positionary_vector = []
# for i in range(len(lst)):
#     configuration_amount.append(flat_list.count(i))
#     positionary_vector.append(i)


# plt.plot(positionary_vector, configuration_amount)
# plt.show()