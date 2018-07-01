# JEBEM TI MAMU BRE KOJI TI JE KURAC

import numpy as np
import mnist as mn

# SREDJIVANJE PROMENLJIVIH

eig_vec = np.genfromtxt('eig_vec.csv', delimiter = ',')

#-------------

images = mn.train_images()
images = images.reshape(60000,784)
images = images[:60000]
images = np.dot(eig_vec.T, images.T)
        
images = images.T
images = images/np.std(images)

#-------------

int_labels = mn.train_labels()
int_labels = int_labels[:60000] # promeniti br na isti broj koji pise i kod images
temp = [0,0,0,0,0,0,0,0,0]
labels = []


for i in range(len(int_labels)): # nisam u stanju da uradim ovo :)
    temp.insert(int_labels[i], 1)
    labels.append(temp)
    temp = [0,0,0,0,0,0,0,0,0]

for k in range(images.shape[0]):
        l = 0
        for l in range(images.shape[1]):
            if images[k][l] > 0.6:
                images[k][l] = 1
            elif images[k][l] >= -0.6 and images[k][l] <= 0.6:
                images[k][l] = 0
            elif images[k][l] < 0.6:
                images[k][l] = 0

#=============================

# FUNKCIJA PISANJA FILEA

def write_data_append(file_name, data):
    
    with open(file_name, 'ab') as fn:
        
        np.savetxt(fn, data, delimiter = ',')
        
def write_data(txt_file, data):
    
    data = np.real(data)
    np.savetxt(txt_file, data, delimiter = ',')

#========================

# AKTIVACIONA I NJEN IZVOD (NORMALIZUJ PODATKE PA URADI TANH ILI SIGMA)

def tanhf(x):
    return np.tanh(x)

def d_tanhf(x):
    return 1 - pow(x,2)

#===========================


# POSTAVLJANJE VAZNIH VAR

in_neurons = images.shape[1] # ovo ce se menjati kako menjam kolicinu eig_vec

out_neurons = 10

hidden_neurons = 10 # probati razlicite kolicine (in + out neurons / 2)

learn_rate = 0.01 # poigravaj se sa ovime (ovo je na loss funkciji za koliko ces da se spustis kada radis slope)

#learn_iter_iter = 5

learn_iter = 10000 # PROMENI ME NA VELIKI BROJ

#============================


# RANDOM WIEGHTS I BIASES

w_h = np.random.uniform(size = (in_neurons, hidden_neurons)) # mozda je obrnuto (nije)
b_h = np.random.uniform(size = (1, hidden_neurons)) # i ovo

w_o = np.random.uniform(size = (hidden_neurons, out_neurons))
b_o = np.random.uniform(size = (1, out_neurons))

#========================




# UCENJE I CROSS VALIDATION

j = 0

for j in range(int(images.shape[0]/10000)): # ovo na 10000 i prati ostale
    
    print(j)
    rez = []
    i = 0
#    print(images.shape)
#    
#    images = mn.train_images()
#    images = images.reshape(60000,784)
#    images = images[:600]
#    images = np.dot(eig_vec.T, images.T)
#    print(images.shape)    
#    
#    images = images.T
#    print(images.shape)
#    images = images/np.std(images)
    
    
    int_labels = mn.train_labels()
    int_labels = int_labels[:60000] # promeniti br na isti broj koji pise i kod images
    temp = [0,0,0,0,0,0,0,0,0]
    labels = []


    for o in range(len(int_labels)): # nisam u stanju da uradim ovo :)
        temp.insert(int_labels[o], 1)
        labels.append(temp)
        temp = [0,0,0,0,0,0,0,0,0]
    
    
    if j == 0:
        images = images[10000:]
        labels = labels[10000:]
        
    elif j == 5:
        images = images[:50000]
        labels = labels[:50000]
        
    else:
        images = np.concatenate((images[:j*10000], images[(j+1)*10000:]))
        labels = np.concatenate((labels[:j*10000], labels[(j+1)*10000:]))


    for k in range(images.shape[0]):
        l = 0
        for l in range(images.shape[1]):
            if images[k][l] > 0.6:
                images[k][l] = 1
            elif images[k][l] >= -0.6 and images[k][l] <= 0.6:
                images[k][l] = 0
            elif images[k][l] < 0.6:
                images[k][l] = 0
                
    print(images.shape)
    
    while i < learn_iter:
    
        i += 1
    
        # forward feed
    
        hidden_input = np.dot(images, w_h) + b_h
    
        #hidden_input = hidden_input/1000 # normalizacija podataka br/sr.vr (np.std(hidden_input))

        hidden_activation = tanhf(hidden_input)

        #---
    
        out_input = np.dot(hidden_activation, w_o) + b_o

        output = tanhf(out_input)

        #------------------------

        # backpropagation

        Loss_out = labels - output
    
        #---

        out_slope = d_tanhf(output)
    
        hidden_slope = d_tanhf(hidden_activation)

        #---

        out_delta = Loss_out * out_slope

        #---
    
        Loss_hidden = np.dot(out_delta, w_o.T)

        #---

        hidden_delta = Loss_hidden * hidden_slope
    
        #--------
    
        w_o += np.dot(hidden_activation.T, out_delta) * learn_rate
    
        w_h += np.dot(images.T, hidden_delta) * learn_rate

        #---
    
        b_o += np.sum(out_delta, axis = 0) * learn_rate
        b_h += np.sum(hidden_delta, axis = 0) * learn_rate
            
        #=================
        
            
        #GUESS (za vezbanje)
    
        if ((i/(learn_iter/10)).is_integer() == True) and (i/(learn_iter/10) != 0.0): # promeniti oba broja na learn_iter/10

            temp = [np.argmax(output[j]) == np.argmax(labels[j]) for j in range(len(images))]
            # da li je one line for loop brzi od viselinijskog?
        
            rez.append((np.count_nonzero(np.array(temp) == True) / len(images) * 100, i))
        
        
            write_data('Out_weights-{}-{}.csv'.format(j,i), w_o)
            write_data('Hidden_weights-{}-{}.csv'.format(j,i), w_h)
            write_data_append('Pcnt_iter.csv', np.array(rez))
            rez = []
            write_data('Out_bias-{}-{}.csv'.format(j,i), b_o)
            write_data('Hidden_bias-{}-{}.csv'.format(j,i), b_h)


        print('\r{} / {}'.format(i,learn_iter), end = '\r')
        
    print(images.shape)
    
    images = mn.train_images()
    images = images.reshape(60000,784)
    images = images[:60000]
    images = np.dot(eig_vec.T, images.T)
    print(images.shape)    
    
    images = images.T
    print(images.shape)
    images = images/np.std(images)
#===================================



