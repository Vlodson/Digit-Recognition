# MOZDA (ALI MOOOOOOOOOOOOOOZDA) BI TREBALO DA TRANSPONUJEM ODMA IMAGE_ARR
# NE ZNAM DA LI TO MENJA REZULTATE
# NE BI TREBALO JER SAM POSTAVIO SVE ZA NE TRANSPONOVANO A SHAPE NA KRAJU SE DOBIJE ISTI


# IMPORTOVANJE MNIST DATABASEA

import mnist as mn

#mndata = MNIST('C:/Users/Vlada/Desktop/programs')

# u image su slike... u label su 'odgovori'
image_lst = mn.train_images()
label = mn.train_labels()

image_lst = image_lst.reshape(60000,784)

image_lst = image_lst[:50]

#===================================


# SREDJIVANJE DATASETA

import numpy as np
#np.set_printoptions(threshold = np.nan)


image_arr = image_lst 
#===================================


# MEAN VECTOR

mean_vector = np.array([])

for i in range(784):
    mean_vector = np.append(mean_vector, [np.mean(image_arr[:,i])])

#==================================


# SCATTER MATRIX

scatter_matrix = np.zeros([784,784])

for i in range(image_arr.shape[0]):
        scatter_matrix += (image_arr[i,:].reshape(784,1) - mean_vector).dot((image_arr[i,:].reshape(784,1) - mean_vector).T)                

#==================================


# EIGENVECTORS AND EIGENVALUES

eig_val, eig_vec = np.linalg.eig(scatter_matrix)

# tjt lmao, hvala bogu za numpy

#==================================


# SORTIRANJE VAZNIH EIGVAL I EIGVEC

eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]

eig_pairs.sort(key=lambda x: x[0], reverse=True)

#==================================


# PROBA U 2D

#matrix_w = np.hstack((eig_pairs[0][1].reshape(784,1), eig_pairs[1][1].reshape(784,1)))
#
#transform = matrix_w.T.dot(image_arr.T)
#
#import matplotlib.pyplot as plt
#
#plt.plot(transform[0,:], transform[1, :], 'ro', markersize = 0.3)
#plt.show()

# radi dobro, vide se grupacije?

#==================================


# MATRICA EIGVEC (probati za 3-10 eigvec)

# hstack - po collumns spaja vektore pa samo uzimam vektore najvecih eigval pa ih spajam kolko je sirok
# tolka je dimenzijalnost, pa je meni 3 u 3d logicno
# vrv moze u neki for loop da se spakuje ali nemam zivaca trebam samo jednom da pokrenem prog
matrix_w = np.hstack((eig_pairs[0][1].reshape(784,1), 
                      eig_pairs[1][1].reshape(784,1), 
                      eig_pairs[2][1].reshape(784,1),
                      eig_pairs[3][1].reshape(784,1),
                      eig_pairs[4][1].reshape(784,1),
                      eig_pairs[5][1].reshape(784,1),
                      eig_pairs[6][1].reshape(784,1),
                      eig_pairs[7][1].reshape(784,1),
                      eig_pairs[8][1].reshape(784,1),
                      eig_pairs[9][1].reshape(784,1)))


#==================================

# TRANSFORMACIJA DATASETA

# ovde su krajnji rezultati y = W^t * dataset^t (prava formula je bez dataset transponovan)
# ali ja nisam na pocetku transponovao tkd bi ovo bilo u redu
# .dot je isto sto i matmul ali brze jer dve iste stvari u numpy rade razlicitim brzinama :)))))))))))
transform = matrix_w.T.dot(image_arr.T)
transform = np.real(transform)

#===================================


# RAZVRSTAVANJE CIFARA

dig0 = []
dig1 = []
dig2 = []
dig3 = []
dig4 = []
dig5 = []
dig6 = []
dig7 = []
dig8 = []
dig9 = []

for i in range(len(transform[0])):
    
    if label[i] == 0:
        
        dig0.append(transform[:,i])
    
    elif label[i] == 1:
        
        dig1.append(transform[:,i])
     
    elif label[i] == 2:
        
        dig2.append(transform[:,i])    

    elif label[i] == 3:
        
        dig3.append(transform[:,i])

    elif label[i] == 4:
        
        dig4.append(transform[:,i])
        
    elif label[i] == 5:
        
        dig5.append(transform[:,i])    
    
    elif label[i] == 6:
        
        dig6.append(transform[:,i])
        
    elif label[i] == 7:
        
        dig7.append(transform[:,i])
    
    elif label[i] == 8:
        
        dig8.append(transform[:,i])
    
    elif label[i] == 9:
        
        dig9.append(transform[:,i])
        
#====================================


# ZAPIS EIG_VEC I CIFARA

def write_data(txt_file, data):
    
    data = np.real(data)
    np.savetxt(txt_file, data, delimiter = ',')

#-----------------------------------

write_data('0.csv', dig0)
write_data('1.csv', dig1)
write_data('2.csv', dig2)
write_data('3.csv', dig3)
write_data('4.csv', dig4)
write_data('5.csv', dig5)
write_data('6.csv', dig6)
write_data('7.csv', dig7)
write_data('8.csv', dig8)
write_data('9.csv', dig9)
write_data('eig_vec.csv', matrix_w)


#===================================








#  PREDSTAVLJANJE EIGENVECTORA (3-10), pokazuje samo poslednji posto sto da ne lmao

#import matplotlib.pyplot as plt
#
#for i in range (len(matrix_w[0])):
#    plt.imshow(np.real(eig_pairs[i][1].reshape(28,28)))

#====================================


# 3D ODAVDE (useless)

# pravim x,y,z za svaku cifru

#def xyz(arr):
#    
#    arr = np.array(arr)
#    
#    x = arr[:, 0]
#    y = arr[:, 1]
#    z = arr[:, 2]
#
#    return x,y,z
#
#x0,y0,z0 = xyz(dig0)
#x1,y1,z1 = xyz(dig1)
#x2,y2,z2 = xyz(dig2)
#x3,y3,z3 = xyz(dig3)
#x4,y4,z4 = xyz(dig4)
#x5,y5,z5 = xyz(dig5)
#x6,y6,z6 = xyz(dig6)
#x7,y7,z7 = xyz(dig7)
#x8,y8,z8 = xyz(dig8)
#x9,y9,z9 = xyz(dig9)

##-----------------------------------
#
## mozda ne treba idfk nikad nisam radio u 3d, ostavicu ga
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection = '3d') # wat dis do?, sta je prvi arg
#
##x = transform[0, :]
##y = transform[1, :]
##z = transform[2, :]
#
#ax.scatter(x0,y0,z0, c = 'r', marker = 'o')
#ax.scatter(x1,y1,z1, c = 'b', marker = 'o')
#ax.scatter(x2,y2,z2, c = 'g', marker = 'o')
#ax.scatter(x3,y3,z3, c = 'y', marker = 'o')
#ax.scatter(x4,y4,z4, c = 'c', marker = 'o')
#ax.scatter(x5,y5,z5, c = 'm', marker = 'o')
#ax.scatter(x6,y6,z6, c = 'k', marker = 'o')
#ax.scatter(x7,y7,z7, c = 'w', marker = 'o')
#ax.scatter(x8,y8,z8, c = (0.1, 0.2, 0.5), marker = 'o')
#ax.scatter(x9,y9,z9, c = (0.5, 0.1, 0.2), marker = 'o')
#
#plt.show()
#
## radi dobro, vide se grupacije, bolje nego 2D
#    
#plt.imshow(eig_pairs[0][1].reshape(28,28))














