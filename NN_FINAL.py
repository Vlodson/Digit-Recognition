import numpy as np
import mnist as mn

# SREDJIVANJE PROMENLJIVIH

eig_vec = np.genfromtxt('eig_vec.csv', delimiter = ',')

#-------------

images1 = mn.train_images()
images1 = images1.reshape(60000,784)
images2 = mn.test_images()
images2 = images2.reshape(10000,784)
images = np.vstack((images1, images2))
images = np.dot(eig_vec.T, images.T)
images = images[:, :60000] # menjas ovo na sta hoces, ali moras da promenis i label na isti br
#images = images.reshape(5,10)
images = images.T
images = images/np.std(images)

for i in range(images.shape[0]):
    j = 0
    for j in range(images.shape[1]):
        if images[i][j] > 0.5:
            images[i][j] = 1
        elif images[i][j] >= -0.5 and images[i][j] <= 0.5:
            images[i][j] = 0
        elif images[i][j] < -0.5:
            images[i][j] = -1

# NIJE SPORO 

#-------------

int_labels = mn.train_labels()
int_labels = int_labels[:15] # promeniti br na isti broj koji pise i kod images
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

hidden_neurons = 10 # probati razlicite kolicine (in + out neurons / 2)

learn_rate = 0.01 # poigravaj se sa ovime (ovo je na loss funkciji za koliko ces da se spustis kada radis slope)

#learn_iter_iter = 5

learn_iter = 5000 # PROMENI ME NA VELIKI BROJ

#============================


# AKTIVACIONA I NJEN IZVOD (NORMALIZUJ PODATKE PA URADI TANH ILI SIGMA)

def tanhf(x):
    return np.tanh(x)

def d_tanhf(x):
    return 1 - pow(x,2)

#===========================


# RANDOM WIEGHTS I BIASES

w_h = np.random.uniform(size = (in_neurons, hidden_neurons)) # mozda je obrnuto (nije)
b_h = np.random.uniform(size = (1, hidden_neurons)) # i ovo

w_o = np.random.uniform(size = (hidden_neurons, out_neurons))
b_o = np.random.uniform(size = (1, out_neurons))

#========================


# FUNKCIJA PISANJA FILEA

def write_data(file_name, data):
    
    with open(file_name, 'ab') as fn:
        
        np.savetxt(fn, data, delimiter = ',')

#========================

# UCENJE

rez = []
i = 0

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
    
    if ((i/500).is_integer() == True) and (i/500 != 0.0): # promeniti oba broja na learn_iter/10

        temp = [np.argmax(output[j]) == np.argmax(labels[j]) for j in range(len(images))]
        # da li je one line for loop brzi od viselinijskog?
    
        rez.append((np.count_nonzero(np.array(temp) == True) / len(images) * 100, i))
        
        
        write_data('Out_weights.csv', w_o)
        write_data('Hidden_weights.csv', w_h)
        write_data('Pcnt_iter.csv', np.array(rez))
        rez = []
        write_data('Out_bias.csv', b_o)
        write_data('Hidden_bias.csv', b_h)


        print(i)
#===================================








