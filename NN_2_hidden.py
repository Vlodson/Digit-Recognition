import numpy as np
import mnist as mn
import matplotlib.pyplot as plt

# SREDJIVANJE PROMENLJIVIH

eig_vec = np.genfromtxt('eig_vec.csv', delimiter = ',')

#-------------

images = mn.train_images()
images = images.reshape(60000,784)
#images1 = images1.reshape(60000,784)
#images2 = mn.test_images()
#images2 = images2.reshape(10000,784)
#images = np.vstack((images1, images2))
images = np.dot(eig_vec.T, images.T)
images = images[:, :100] # menjas ovo na sta hoces, ali moras da promenis i label na isti br
#images = images.reshape(5,10)
images = images.T
#np.insert(images, images[:, -1], 1)
images = (images-np.mean(images))/np.std(images)

#print('images:  {}'.format(images.shape))

#for i in range(images.shape[0]):
#    j = 0
#    for j in range(images.shape[1]):
#        if images[i][j] > 0:
#            images[i][j] = 1
#        elif images[i][j] >= -0.6 and images[i][j] <= 0.6:
#            images[i][j] = 0
#        elif images[i][j] < 0:
#            images[i][j] = 0

# NIJE SPORO 

#-------------

int_labels = mn.train_labels()
int_labels = int_labels[:100] # promeniti br na isti broj koji pise i kod images
temp = [0,0,0,0,0,0,0,0,0]
labels = []


for i in range(len(int_labels)): # nisam u stanju da uradim ovo :)
    temp.insert(int_labels[i], 1)
    labels.append(temp)
    temp = [0,0,0,0,0,0,0,0,0]


#=============================


# POSTAVLJANJE VAZNIH VAR

in_neurons = images.shape[1] # ovo ce se menjati kako menjam kolicinu eig_vec

out_neurons = 10

hidden1_neurons = 20 # probati razlicite kolicine (in + out neurons / 2)
hidden2_neurons = 20

learn_rate = 1e-10 # poigravaj se sa ovime (ovo je na loss funkciji za koliko ces da se spustis kada radis slope)

#learn_iter_iter = 5

learn_iter = 5000 # PROMENI ME NA VELIKI BROJ

#============================


# AKTIVACIONA I NJEN IZVOD

def sqrtf(x):
    return np.sqrt(x)

def d_sqrtf(x):
    return 1/(2*np.sqrt(x))

def sqrf(x):
    return  pow(x, 2)

def d_sqrf(x):
    return 2*x

def sigmaf(x):
    return 1/(1+np.exp(-x))

def d_sigmaf(x):
    return x*(1-x)

#===========================


# RANDOM WIEGHTS I BIASES

np.random.seed(27836)

w_h1 = np.random.uniform(low = 0.9, high = 1.1, size = (in_neurons, hidden1_neurons)) # mozda je obrnuto (nije)

b_h1 = np.random.uniform(low = 0.9, high = 1.1, size = (1, hidden1_neurons)) # i ovo

w_h2 = np.random.uniform(low = 0.9, high = 1.1, size = (hidden1_neurons, hidden2_neurons))

b_h2 = np.random.uniform(low = 0.9, high = 1.1, size = (1, hidden2_neurons))

w_o = np.random.uniform(low = 0.9, high = 1.1, size = (hidden2_neurons, out_neurons))

b_o = np.random.uniform(low = 0.9, high = 1.1, size = (1, out_neurons))

#========================


# FUNKCIJA PISANJA FILEA

def write_data_append(file_name, data):
    
    with open(file_name, 'ab') as fn:
        
        np.savetxt(fn, data, delimiter = ',')
        
def write_data(txt_file, data):
    
    data = np.real(data)
    np.savetxt(txt_file, data, delimiter = ',')

#========================

# UCENJE

pct = []
loss = []
rez = []
i = 1

while i <= learn_iter:
    
    i += 1
    
    # forward feed
    
    hidden1_input = np.dot(images, w_h1) + b_h1
    hidden1_input = (np.abs(hidden1_input - np.mean(hidden1_input))) / np.std(hidden1_input) # ovo sam ubacio
    

    hidden1_activation = sqrf(hidden1_input)
    
    #---
    
    hidden2_input = np.dot(hidden1_activation, w_h2) + b_h2
    hidden2_input = (np.abs(hidden2_input - np.mean(hidden2_input))) / np.std(hidden2_input) # ovo sam ubacio
    
    hidden2_activation = sqrtf(hidden2_input)
    
    #---
    
    out_input = np.dot(hidden2_activation, w_o) + b_o
    out_input = (np.abs(out_input - np.mean(out_input))) / np.mean(out_input) # ovo sam ubacio

    output = sigmaf(out_input)
    
    #------------------------

    # backpropagation

    Loss_out = np.sum((labels - output).T.dot(labels - output))/(images.shape[0]-1)
    
    #Loss_out = labels - output
    
    
    #---

    out_slope = d_sigmaf(output)
    
    hidden1_slope = d_sqrf(hidden1_activation)
    
    hidden2_slope = d_sqrtf(hidden2_activation)

    #---

    out_delta = Loss_out * out_slope

    #---
    
    Loss_hidden2 = np.dot(out_delta, w_o.T)
    
    hidden2_delta = Loss_hidden2 * hidden2_slope # **
    
    Loss_hidden1 = np.dot(hidden2_delta, w_h2.T)

    #---

    hidden1_delta = Loss_hidden1 * hidden1_slope
    
    #--------
    
    w_o += np.dot(hidden2_activation.T, out_delta) * learn_rate # vrati na +=
    
    w_h1 += np.dot(images.T, hidden1_delta) * learn_rate
    
    w_h2 += np.dot(hidden1_activation.T, hidden2_delta) * learn_rate
    
    #---
    
    b_o += np.sum(out_delta, axis = 0) * learn_rate
    b_h1 += np.sum(hidden1_delta, axis = 0) * learn_rate
    b_h2 += np.sum(hidden2_delta, axis = 0) * learn_rate

    #=================
        
        
    #GUESS (za vezbanje)
    
    if ((i/(learn_iter/10)).is_integer() == True) and (i/(learn_iter/10) != 0.0): # promeniti oba broja na learn_iter/10

        temp = [np.argmax(output[j]) == np.argmax(labels[j]) for j in range(len(images))]
        # da li je one line for loop brzi od viselinijskog?
    
        rez.append((np.count_nonzero(np.array(temp) == True) / len(images) * 100, i))
        
        
        write_data('Out_weights-%d.csv' %i, w_o)
        write_data('Hidden1_weights-%d.csv' %i, w_h1)
        write_data('Hidden2_weights-%d.csv' %i, w_h2)
        write_data_append('Pcnt_iter.csv', np.array(rez))
        pct.append(rez)
        rez = []
        write_data('Out_bias-%d.csv' %i, b_o)
        write_data('Hidden1_bias-%d.csv' %i, b_h1)
        write_data('Hidden2_bias-%d.csv' %i, b_h2)
        
        
        print(i)
        print(Loss_out)
    loss.append(Loss_out)
#===================================

x = []
[x.append(i) for i in range(learn_iter)]
plt.plot(x, loss, 'b-')
#plt.xlim(6000)
#plt.ylim(100)

#print(np.delete(lst, (i,i+1)))




