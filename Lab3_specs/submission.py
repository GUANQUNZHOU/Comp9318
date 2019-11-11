import numpy as np
def logistic_regression(data, labels, weights, num_epochs, learning_rate): # do not change the heading of the function
    result_weight = []
    n = data.shape
    numb = n[0]
    data_for = np.insert(data, 0, np.ones(numb), axis = -1)
    data_matrix = np.mat(data_for)
    label_matrix = np.mat(labels)
    label_matrix_transport = label_matrix.T
    weight_matrix = np.mat(weights)
    weight_matrix_trans = weight_matrix.T
    for i in range(0, num_epochs):
        dot = np.dot(data_matrix, weight_matrix_trans)
        #dot = dot.sum(axis = 1)
        h_theta = 1 / (1 + np.exp(-dot))
        h_arr = np.array(h_theta)
        cost = np.hstack((-label_matrix_transport,h_arr))
        #print(cost.shape)
        cost_function = cost.sum(axis = 1)
        #print(cost_s)
        cost_function_trans = cost_function.T
        ff = np.dot(cost_function_trans,data_matrix)
        weight_matrix = weight_matrix - learning_rate*ff
        weight_matrix_trans = weight_matrix.T
    weight_matrix_arr = np.array(weight_matrix)
    #print(weight_matrix)
    result = weight_matrix_arr[0]
    for i in result:
        i = float(format(i,'.8f'))
        result_weight.append(i)
    return result_weight
    
