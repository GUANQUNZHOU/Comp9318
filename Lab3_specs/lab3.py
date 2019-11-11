import pandas as pd
import numpy as np

data_file='./asset/a'
raw_data = pd.read_csv(data_file, sep=',')
#print(raw_data.head())

labels=raw_data['Label'].values
data=np.stack((raw_data['Col1'].values,raw_data['Col2'].values), axis=-1)
#print(data)
weights = np.zeros(3) # We initialize the coefficients with ZERO. We also compute the intercept term.
num_epochs = 20000
learning_rate = 50e-5
'''
def logistic_regression(data, labels, weights, num_epochs, learning_rate): # do not change the heading of the function
    #print(data, labels, weights)
    data = np.insert(data, 0, np.ones(data.shape[0]), axis=-1)
    print(data)
    for i in range(num_epochs):
    	h_t = 1/(1+np.exp(-1.0*np.dot(data, weights)))
    	grads = np.array([(labels-h_t) * data[:, j] for j in range(len(weights))]).transpose((1,0))
    	grads = np.sum(grads, axis=0)
    	w = weights - learning_rate*grads
    return w

co = logistic_regression(data, labels, weights, num_epochs, learning_rate)
print(co)
'''

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
    for i in range(num_epochs):
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
    
co = logistic_regression(data, labels, weights, num_epochs, learning_rate)
print(co)
