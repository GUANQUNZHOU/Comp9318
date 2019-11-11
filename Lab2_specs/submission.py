## import modules here 
import pandas as pd
import numpy as np



################# Question 1 #################

def v_opt_dp(x, num_bins):# do not change the heading of the function
    matrix = [[-1 for i in range(len(x))] for j in range(num_bins)]
    matrix_index = [[-1 for i in range(len(x))] for j in range(num_bins)]
    for i in range(num_bins):
        for j in range(len(x)):
            if num_bins-i-j <= 1  and len(x)-j>=i+1:
                if i == 0:
                    matrix[i][j] = np.var(x[j:])*len(x[j:])
                else:
                    miinl = []
                    for k in range(j+1,len(x)):
                        miinl.append(matrix[i-1][k]+np.var(x[j:k])*(k-j))#find all cases when current element in current bin 
                    matrix[i][j] = min(miinl)#decide on case
                    matrix_index[i][j] = miinl.index(matrix[i][j])+j+1#decide next element's index
    #print(matrix_index)
    start_index = 0
    current_index = 0
    current_bin = []
    for g in range(len(matrix_index) - 1, 0, -1):#group the given list
        start_index = matrix_index[g][start_index]
        current_bin.append(x[current_index:start_index])
        current_index = start_index
    current_bin.append(x[current_index:])
    return matrix,current_bin
