import numpy as np

# IMPORTOVANJE TEGOVA I BIASA

w_indx = np.genfromtxt('Pcnt_iter.csv', delimiter = ',')

indx = np.argmax(w_indx[:, 0])

#------------

w_h = np.genfromtxt('Hidden_weights.csv', delimiter = ',')
w_h = w_h[indx : indx +10]

#-----

b_h = np.genfromtxt('Hidden_bias.csv', delimiter = ',')
b_h = b_h[indx]

#-----

w_o = np.genfromtxt('Out_weights.csv', delimiter = ',')
w_o = w_o[indx : indx +10]

#-----

b_o = np.genfromtxt('Out_bias.csv', delimiter = ',')
b_o = b_o[indx]

#===========================


import mnist as mn

eig_vec = np.genfromtxt('eig_vec.csv', delimiter = ',')

#-------------

images = mn.test_images()
images = images.reshape(10000,784)
images = np.dot(eig_vec.T, images.T)
images = images[:, :50] # menjas 5 na sta hoces, ali moras da promenis i label na isti br
#images = images.reshape(5,10)
images = images.T

#-------------

int_labels = mn.test_labels()
int_labels = int_labels[:50] # promeniti br na isti broj koji pise i kod images
temp = [0,0,0,0,0,0,0,0,0]
labels = []


for i in range(len(int_labels)): # nisam u stanju da uradim ovo :)
    temp.insert(int_labels[i], 1)
    labels.append(temp)
    temp = [0,0,0,0,0,0,0,0,0]
    
#==========================

    
# AKTIVACIONA

def tanhf(x):
    return np.tanh(x)    
    
#=========================


# PREDICTION
    
hidden_input = np.dot(images, w_h) + b_h   

hidden_input = hidden_input/np.std(hidden_input) # normalizacija podataka br/sr.vr

hidden_activation = tanhf(hidden_input)

#---
    
out_input = np.dot(hidden_activation, w_o) + b_o

output = tanhf(out_input)

counter = np.count_nonzero(np.array([np.argmax(output[i]) == np.argmax(labels[i]) for i in range(len(labels))]))

print('{}%'.format(counter/(len(labels)*100)))


















    
    
    
    
