# prog ne radi ako nema makar po jedna cifra u .csv fileovima
# dodati sta da se radi u tom slucaju ali s obzirom da se ucitava 60k slika nema sanse da ce se to
# dogoditi

#===============================


# UNOS TEST SETOVA IZ MNISTA

import mnist as mn

image = mn.test_images()
image = image.reshape(10000, 784)
image = image[:10]

label = mn.test_labels()

#===============================


# UZIMANJE DATASETA ZA UPOREDJIVANJE

import numpy as np

matrix_w = np.genfromtxt('eig_vec.csv', delimiter = ',')
#
#digits = np.genfromtxt('0.csv', delimiter = ',')
#digits = np.vstack((digits, [0,0,0,0,0,0,0,0,0,0])) # umesto [0,0,0...] stavi numpy 0s, 10 ih je

digits = np.array([0,0,0,0,0,0,0,0,0,0])

for i in range(10): # br cifara
    digits = np.vstack((digits, np.genfromtxt('{}.csv'.format(i), delimiter = ',')))
    digits = np.vstack((digits, [0,0,0,0,0,0,0,0,0,0])) # umesto [0,0,0...] stavi numpy 0s 
    
# sum ce joj biti nula i tako ces razaznati cifre jednu od drugih, nadjes kada je sum = 0 i znas
# odakle dokle su cifre

# ovo bi mogao da uradis i na PCA_FINAL

#================================


# MNOZENJE TEST SETA SA MATRIX_W (y = W^t * testset^t)

image_transf = matrix_w.T.dot(image.T)

#================================


# CUVAM MESTA IZMEDJU KOJIH SE NALAZE CIFRE

zeros = []

for i in range (len(digits)):
    if sum(digits[i]) == 0:
        zeros.append(i)
        
#================================        

# ESTIMATE

def est(arr1, arr2): # mozda kao arg dodati koliko komsija hoce pa su komsije input
    
    e_dist = []
    min_dist = []
    min_idx = []
    est_val = []
    
    # euklidska distanca (2d = x1 - x2 na kv + y1 - y2 na kv pa koren iz toga)
    
    for i in range(len(digits)):
        if sum(arr2[i]) != 0:
            temp = pow(sum(pow((arr1 - arr2[i]), 2)), 0.5)
            e_dist.append(temp)
    
    #---------------------------
    
    # uzimanje 10 (promeniti koliko hoces) najmanjih
    
    for i in range(10): # promeniti u koliko najmanjih hoces
        min_dist.append(min(e_dist)) # mozda ne treba lmao
        min_idx.append(e_dist.index(min(e_dist))) 
        e_dist.remove(min(e_dist))
    
    min_dist.sort() # a ni ovo ali cu ga nositi
    min_idx.sort()
    
    #---------------------------
    
    # stavljanje predpostavki koji je broj u listu iz koje cu uzimati broj sa najvecim % u njoj
    
    for i in range(10): # menjas za vise neighboura
        j = 0
        for j in range(10):
            if (min_idx[j] >= zeros[i]) and (min_idx[j] < zeros[i+1]): # dodato >=
            # nije dovoljno samo da budu > i < , treba da bude >= od manjeg a < od veceg
            # tako pocinju granice
                est_val.append(i)
                
    #---------------------------            
    
    # odredjivanje % cifre u est_val
    
    est_prcnt = []
    
    for i in range(10): # menjas ako ces na vise nearest neigboura
        if est_val.count(i) > 0:    
            est_prcnt.append([est_val.count(i), i])
    
    #---------------------------
    
    # pravljenje guessa
    
    guess = est_prcnt[est_prcnt.index(max(est_prcnt))][1] # prvi [] mi vrati listu u kojoj se nalazi
    # najveci broj komsija neke cifre, a drugi [] vraca koja je to cifra koja ima najvie komsija
    
    #---------------------------
    
    return guess

#===============================


# tacnost programa

guesses = []
counter = 0

for i in range(len(image_transf[0])):
    if est(image_transf[:, i], digits) == label[i]:
        counter += 1

print('{0:.2f}%'.format(counter/len(image_transf[0])*100))



















