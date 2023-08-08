# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:09:54 2021

@author: mdabrowski
"""

import numpy as np
import matplotlib.pyplot as plt # ipython -pylab
import scipy as sp
from sys import argv
import time
import scipy.signal as ss
import random as rnd
import simpy
import pyclustering as pc
from random import gauss

# ***** Chapter 2 *****

#%% Five different ways to create a vector

vec1 = np.array([0., 1., 2., 3., 4.]) # from a Python list

vec2 = np.arange(0, 5, 1, dtype = float) # arange(start inclusive, stop exclusive, step size) (default: double)

vec3 = np.linspace(0, 4, 5) # linspace(start inclusice, stop inclusive, number of elements)

vec4 = np.zeros(5) # zeron(n) returns a vector filled with n zeros
for i in range(5):
    vec4[i] = i

vec5 = np.loadtxt("E:\ML\ML_courses\Janert_DataAnalysis\data.txt") # read from a text file, one number per row

#%% ... continuation from the previous listing

v1 = vec1 + vec2 # add a vector to another (true even for multiplication v1 * v2)

# Unnecessary: adding two vectors using an explicit loop
v2 = np.zeros(5) 
for i in range(5):
    v2[i] = vec1[i] + vec2[i]
    
vec1 += vec2 # adding a vector to another in place

# Broadcasting: combining scalars and vectors
v3 = 2*vec3 
v4 = vec4 + 3

v5 = np.sin(vec5) # ufuncs: applying a function to a vector, element by element

lst = v5.tolist # converting to Python list object again

#%% Calculating kernel density estimates

def kde(z, w, xv):
    '''z: position; w: bandwidth; xv: vector of points'''
    
    return np.sum(np.exp(-0.5 * ((z - xv) / w)**2) / np.sqrt(2 * np.pi * w**2))

#%%

# Generate two vectors with 12 elements each
d1 = np.linspace(0, 11, 12)
d2 = np.linspace(0, 11, 12)

# Reshape the first vector to a 3x4 (row x col) matrix
d1.shape = (3, 4)
print(d1)

# Generate a matrix view to the second vector
view = d2.reshape((3, 4)) # also possible: view = np.reshape(d2, (3, 4))

total = d1 + view # now: possible to combine the matrix and the view

print(d1[0, 1]) # element access: [row, column] for matrix
print(view[0, 1])
print(d2[1]) # ... and [pos] for vector

# Shape or layout information
print(d1.shape)
print(d2.shape)
print(view.shape)

# Number of elements (both commands equivalent)
print(d1.size)
print(len(d2))

# Number of dimensions (both commands equivalent)
print(d1.ndim) # d.ndim == len(d.shape)
print(np.rank(d2))

#%%

# Create a 12-element vector and reshape into 3x4 matrix
d = np.linspace(0, 11, 12)
d.shape = (3, 4)
print(d)

# Slicing ... (slicing return views, not copies) (syntax: start:stop:step)
print(d[0, :]) # first row
print(d[:, 1]) # second column
print(d[0, 1]) # individual element: scalar
print(d[0:1, 1]) # subvector of shape 1
print(d[0:1, 1:2]) # subarray of shape 1x1

# Indexing ... (advances indexing returns copies, not views)
print(d[:, [2, 0]]) # integer indexing: third and first columns

k = np.array([False, True, True])
print(d[k, :]) # Boolean indexing: second and third rows

# ***** Chapter 3 *****

#%% Commands show(), hold(), ishold() and clf() control display of the figure

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))

plt.plot(x, 0.5 * np.cos(2 * x))
plt.title("A matplotlib plot")
plt.text(1, -0.8, "A text label")
plt.ylim(-1.1, 1.1) # try also: asix() command 
plt.clf()

#%%

x1 = np.linspace(0, 10, 40)
plt.plot(x1, np.sqrt(x1), 'k-')

plt.figure(2)
x2 = np.linspace(0, 10, 100)
plt.plot(x1, np.sin(x1), 'k--', x2, 0.2 * np.cos(3 * x2), 'b:')

plt.figure(1)
plt.plot(x1, 3 * np.exp(-x1/2), linestyle = 'None', color = 'red', marker = 'o', markersize = 7)
plt.savefig('graph1.png') # also possible: PostScript, PDF and SVG
plt.clf()

#%%

def loess(x, h, xp, yp):
    '''x: location; h: bandwidth; xp, yp: data points (vectors)'''
    
    w = np.exp(-0.5 * (((x - xp) / h)**2) / np.sqrt(2 * np.pi * h**2))
    
    b = np.sum(w * xp) * np.sum(w * yp) - np.sum(w) * np.sum(w * xp * yp)
    b /= np.sum(w * xp)**2 - np.sum(w) * np.sum(w * xp**2)
    
    a = (np.sum(w * yp) - b * np.sum(w * xp)) / np.sum(w)
    
    return a + b * x

d = np.loadtxt("E:\ML\ML_courses\Janert_DataAnalysis\draftlottery.txt")
s1, s2 = [], []

for k in d[:, 0]:
    s1.append(loess(k, 5, d[:, 0], d[:, 1]))
    s2.append(loess(k, 100, d[:, 0], d[:, 1]))
    
plt.xlabel("Day in Year")
plt.ylabel("Draft Number")

plt.gca().set_aspect('equal') # gca() for current plot, gcf() for entire 'Figure' object

plt.plot(d[:, 0], d[:, 1], 'o', color = 'red', markersize = 7, linewidth = 3)
plt.plot(d[:, 0], np.array(s1), 'k-', d[:, 0], np.array(s2), 'k--')

q = 4
plt.axis([1 - q, 366 + q, 1 - q, 366 + q]) # don't confuse it with axes() command

plt.savefig("draftlottery.eps")
plt.clf()

#%%

x = np.linspace(0, 10, 100)
ps = plt.plot(x, np.sin(x), x, np.cos(x))

t1 = plt.text(1, -0.5, "Hello")
t2 = plt.text(3, 0.5, "Hello again")

t1.set_position([7, -0.5])
t2.set(position = [5, 0], text = "Goodbye")

plt.draw()

plt.setp([t1, t2], fontsize = 14) # try: setp(r) and getp(r) with object r

t2.remove() # remove() function is derived from 'Artist' class
plt.Artist.remove(ps[1]) # plot() returns a list of objects

plt.clf() # clear the figure

# ***** Chapter 4 *****

#%%

filename = 'E:\ML\ML_courses\Janert_DataAnalysis\callcenter'

# Read data from a text file, retaining only the second column (column indexes start at 0). The default delimiter
# is any whitespace.
data = np.loadtxt(filename + ".txt", comments = '#', delimiter = None, usecols = (1, ))
n = data.shape[0] # the number of points in the time series (we will need it later)

#%% Finding a smoothed version of the time series

# 1) Construct a 31-point Gaussian filter with standard deviation = 4
filt = ss.gaussian(31, 4)

# 2) Normalize the filter through dividing by the sum of its elements
filt /= np.sum(filt)

# 3) Pad data on both sides with half the filter length of the last value (the function np.ones(k) returns
# a vector of length k, with all elements equal 1).
padded = np.concatenate((data[0] * np.ones(31 // 2), data, data[n-1] * np.ones(31 // 2)))

# 4) Convolve the data with the filter. See text for the meaning of "mode".
smooth = ss.convolve(padded, filt, mode = 'valid') # also possible: mode = 'same'

#%% Plot the raw data together with the smoothed data

# 1) Create a figure, sized to 7x5 inches
plt.figure(1, figsize = (7, 5))

# 2) Plot the raw data in red
plt.plot(data, 'r')

# 3) Plot the smoothed data in blue
plt.plot(smooth, 'b')

# 4) Save the figure to file
plt.savefig(filename + "_smooth.png")

#%% Calculate the autocorrelation function

# 1) Subtract the mean
tmp = data - np.mean(data)

# 2) Pad one copy of data on the right with zeros, then form correlation fct
# (the function np.zeros_like() creates a vector with the same dimensions
# as the input vector v but with all elements equal zero).

corr = sp.correlate(tmp, sp.concatenate((np.zeros_like(tmp), tmp)), mode = 'valid')

# 3) Ratain only some of the elements
corr = corr[:200]

# 4) Normalize by dividing by the first element
corr /= corr[0]

#%% Plot the correlation function

plt.figure(2, figsize = (7, 5))
plt.plot(corr)
plt.savefig(filename + "_corr.png")

# ***** Chapter 9 ***** (previous chapters don't contain Python's code)
#%%

def pareto(alpha):
    y = rnd.random()
    return 1.0/pow(1 - y, 1.0 / alpha)

alpha = 0.8 # alpha < 1 (divergence), alpha > 1 (convergence)
n, ttl, mx = 0, 0, 0
l_ttl, l_mx = [], []

while n < 1e7:
    
    n += 1
    v = pareto(alpha)
    ttl += v
    mx = max(mx, v)
    
    if(n % 50000 == 0):
        print(n, ttl / n, mx)
        l_ttl.append(ttl / n)
        l_mx.append(mx)
        
plt.subplot(121)
plt.plot(l_ttl)
plt.subplot(122)
plt.plot(l_mx)
        
# ***** Chapter 12 ***** (previous chapters don't contain Python's code)
#%%

repeats, tosses = 60, 8

def heads(tosses, p):
    
    h = 0
    for x in range(0, tosses):
        if rnd.random() < p:
            h += 1
            
    return h

p = 0
l_p, l_head = [], []

while p < 1.01:
    for t in range(0, repeats):
        head = heads(tosses, p)
        print(p, "\t", head)
        l_p.append(p)
        l_head.append(head)
    p += 0.05
    
plt.scatter(l_p, l_head)

#%%

strategy = 'stick' # must be 'stick', 'choose', or 'switch'

wins = 0
for trial in range(1000):
    
    envelopes = [0, 1, 2] # the prize is always in envelope 0 ... but we don't know that!
    first_choice = rnd.choice(envelopes)
    
    if first_choice == 0:
        envelopes = [0, rnd.choice([1 , 2])] # randomly retain 1 or 2
    else:
        envelopes = [0, first_choice] # retain winner and first choice

    if strategy == 'stick':
        second_choice = first_choice
    elif strategy == 'chooce':
        second_choice = rnd.choice(envelopes)
    elif strategy == 'switch':
        envelopes.remove(first_choice)
        second_choice = envelopes[0]
        
    if second_choice == 0: # remember that the prize is in envelope 0
        wins += 1
        
print(wins/1000)

#%%

n = 1000    # total visitors
k = 100     # avg visitors per day
s = 50      # daily variation

def trial():
    
    visitors_for_day = [0] # no visitors on day 0
    
    has_visited = [0] * n # a flag for each visitor
    for day in range(31):
        
        visitors_today = max(0, int(rnd.gauss(k, s)))
        
        # Pick the individuals who visited today and mark them
        for i in rnd.sample(range(n), visitors_today):
            has_visited[i] = 1
            
        # Find the total number of unique visitors so far
        visitors_for_day.append(sum(has_visited))
        
    return visitors_for_day

l_day, l_r = [], []
for t in range(25):
    
    r = trial()
    for i in range(len(r)):
        print(i, r[i])
        l_day.append(i)
        l_r.append(r[i])
    print('\n')
    
plt.scatter(l_day, l_r)

#%%

class Customer(Process):
    
    def doit(self):
        
        print("Arriving")
        yield request, self, bank
        
        print("Being served")
        yield hold, self, 100.0
        
        print("Arriving")
        yield release, self, bank
        
# Begining of main simulation program
initialize()

bank = Resource()

cust = Customer()
cust.start(cust.doit())

simulate(until = 1000)

#%%

interarrival_time = 10.0
service_time = 8.0

class CustomerGenerator(Process):
    
    def produce(self, b):
        
        while True:
            c = Customer(b)
            c.start(c.doit())
            yield hold, self, rnd.expovariate(1.0 / interarrival_time)
            
class Customer(Process):
    
    def __init__(self, resource):
        
        Process.__init__(self)
        self.bank = resource
    
    def doit(self):
        
        yield request, self, self.bank
        yield hold, self, self.bank.servicetime()
        yield release, self, self.bank
        
class Bank(Resource):
    
    def servicetime(self):
        return rnd.expovariate(1.0 / service_time)
    
initialize()

bank = Bamk(capacity = 1, monitored = True, monitorType = Monitor)

src = CustomerGenerator()
activate(src, src.produce(bank))

simulate(until = 1000)

print(bank.waitMon.mean())

for evt in bank.waitMon:
    print(evt[0], evt[1])
    
#%%
    
interarrival_time = 10.0

class CustomerGenerator(Process):
    
    def produce(self, bank):
        
        while True:
            c = Customer(bank, sim = self.sim)
            c.start(c.doit())
            yield hold, self, rnd.expovariate(1.0 / interarrival_time)
            
class Customer(Process):
    
    def __init__(self, resource, sim = None):
        
        Process.__init__(self, sim = sim)
        self.bank = resource
    
    def doit(self):
        
        yield request, self, self.bank
        yield hold, self, self.bank.servicetime()
        yield release, self, self.bank
        
class Bank(Resource):
    
    def setServicetime(self, s):
        self.service_time = s
        
    def servicetime(self):
        return rnd.expovariate(1.0 / service_time)
    
def run_simulation(t, steps, runs):
    
    for r in range(runs):
        
        sim = Simulation()
        sim.initialize()
        
        bank = Bank(monitored = True, monitorType = Tally, sim = sim)
        bank.setServicetime(t)
        
        src = CustomerGenerator(sim = sim)
        sim.activate(src, src.produce(bank))

    sim.startCollection(when = steps//2)
    sim.simulate(until = steps)
    
    print(t, bank.waitMon.mean())
    
t = 0
while t <= 11.0:
    t += 0.5
    run_simulation(t, 100000, 10)
    #run_simulation(t, 1000, 10)
    
# ***** Chapter 13 *****
#%%
    
# Read data filename and desired number of clusters
filename, n = 'E:\ML\ML_courses\Janert_DataAnalysis\clusters_data.txt', 10

# x and y coordinates, whitespace-separated
data = np.loadtxt(filename, usecols = (0, 1))
k = len(data)

# Perform clustering and find centroids
clustermap = pc.kcluster(data, nclusters = n, npass = 50)[0]
centroids = pc.clustercentroids(data, clusterid = clustermap)[0]

# Obtain distance matrix
m = pc.distancematrix(data)

# Find the masses of all clusters
mass = np.zeros(n)
for c in clustermap:
    mass[c] += 1
    
# Create a matrix for individual silhouette coefficients
sil = np.zeros(n * k)
sil.shape = (k, n)

# Evaluate the distance for all pairs of points
for i in range(0, k):
    for j in range(i+1, k):
        
        d = m[j][i]
        sil[i, clustermap[j]] += d
        sil[j, clustermap[i]] += d
        
# Normalize by cluster size (that is: form average over cluster)
for i in range(0, k):
    sil[i, :] /= mass
    
# Evaluate the silhouette coefficient
s = 0
for i in range(0, k):
    
    c = clustermap[i]
    a = sil[i, c]
    b = min(sil[i, range(0, c) + range(c+1, n)])
    si = (b - a)/max(b, a) # this is the silhouette coeff of point i
    s += si
    
# Print overall silhouette coefficient
print(n, s/k)

#%%
    
# Our own distance function: maximum norm
def dist(a, b):
    return max(abs(a - b))

# Read data filename and desired number of clusters
filename, n = 'E:\ML\ML_courses\Janert_DataAnalysis\clusters_data.txt', 10

# x and y coordinates, whitespace-separated
data = np.loadtxt(filename, usecols = (0, 1))
k = len(data)

# Calculate the distance matrix
m = np.zeros(k*k)
m.shape = (k, k)

for i in range(0, k):
    for j in range(i, k):
        
        d = dist(data[i], data[j])
        m[i][j] = d
        m[j][i] = d
        
# Perform the actual clustering
clustermap = pc.kmedoids(m, nclusters = n, npass = 20)[0]

# Find the indices of the points used as medoids, and the cluster masses
medoids = {}
for i in clustermap:
    medoids[i] = medoids.get(i, 0) + 1
    
# Print points, grouped by cluster
for i in medoids.keys():
    print("Cluster =", i, ", Mass =", medoids[i], ", Centroid: ", data[i])
    
    for j in range(0, k):
        if clustermap[j] == i:
            print("\t", data[j])
            
    
# ***** Chapter 15 (previous chapter doesn't contain Python's code) *****
#%%

def permutations(v):
    
    if len(v) == 1: return [[v[0]]]

    res = []
    for i in range(0, len(v)):
        
        w = permutations(list(v[:i]) + list(v[i+1:]))
        for k in w:
            k.append(v[i])
            
        res += w
        
    return res

n = 5 # try values n = [5, ... , 11]
v = range(n)
t0 = time.process_time()
z = permutations(v)
t1 = time.process_time()

print(n, t1 - t0)

# ***** Chapter 17 (previous chapter contains spcific DB-related Python's code) *****
#%%

c0 = 1.0
c1 = np.arange(1.5, 6.0, 0.5)
mu, sigma = 100, 10
maxtrials = 1000
n_list = np.arange(mu - 5 * sigma, mu + 5 * sigma + 1)

for k in range(len(c1)):
    
    l_avg = np.zeros(len(n_list))
    for n in n_list:
    
        avg = 0
        for trial in range(maxtrials):
        
            m = int(0.5 + gauss(mu, sigma))
            r = c1[k] * min(n, m) - c0 * n
            avg += r
    
        l_avg[n-n_list[0]] = avg/maxtrials    
        print(c1[k], n, np.round(avg/maxtrials, 2))
        
    plt.plot(n_list, l_avg, '.')

# ***** Chapter 18 *****
#%% A Nearest-Neighbor Classifier
    
filepath = 'E:\ML\ML_courses\Janert_DataAnalysis'

train = np.loadtxt(filepath + "\iris_trn.txt", delimiter = ' ', usecols = (0, 1, 2, 3))
trainlabel = np.loadtxt(filepath + "\iris_trn.txt", delimiter = ' ', usecols = (4,), dtype = str)

test = np.loadtxt(filepath + "\iris_tst.txt", delimiter = ' ', usecols = (0, 1, 2, 3))
testlabel = np.loadtxt(filepath + "\iris_tst.txt", delimiter = ' ', usecols = (4,), dtype = str)

hit, miss = 0, 0
for i in range(test.shape[0]):
    
    dist = np.sqrt(np.sum((test[i] - train)**2, axis = 1))
    k = np.argmin(dist)
    
    if trainlabel[k] == testlabel[i]:
        
        flag = '(+)'
        hit += 1
        
    else:
        
        flag = '(-)'
        miss += 1
        
    print(flag, "\t Predicted: ", trainlabel[k], "\t True: ", testlabel[i])
    
print("\n", hit, " out of ", hit + miss, " correct - Accuracy: ", hit / (hit + miss + 0.0))

#%% A Naive Bayesian Classifier

total = {}  # training instances per class label
histo = {}  # histogram

# Read the training set and build up a histogram
filepath = 'E:\ML\ML_courses\Janert_DataAnalysis'

train = open(filepath + "\iris_trn.txt")

for line in train:
    
    f = line.rstrip().split(' ')    # sep_len, sep_wid, pet_len, pet_wid, label
    label = f.pop()
    
    if not label in total:
        
        total[label] = 0
        histo[label] = [{}, {}, {}, {}]
        
    # Counting training instances for the current label
    total[label] += 1
    
    # Iterate over features
    for i in range(4):
        histo[label][i][int(float(f[i]))] = 1 + histo[label][i].get(int(float(f[i])), 0.0)
        
train.close()

# Read the test set and evaluate the probilities
test = open(filepath + "\iris_tst.txt")

hit, miss = 0, 0
for line in test:
    
    f = line.rstrip().split(' ')    # sep_len, sep_wid, pet_len, pet_wid, label
    true = f.pop()
    
    p = {}  # probability for class label, given the test features
    for label in total.keys():
        
        p[label] = 1
        for i in range(4):
            p[label] *= histo[label][i].get(int(float(f[i])), 0.0) / total[label]
            
    # Find the label with the largest probability
    mx, predicted = 0, -1
    for k in p.keys():
        if p[k] >= mx:
            mx, predicted = p[k], k
    
    if true == predicted:
        
        flag = '(+)'
        hit += 1
        
    else:
        
        flag = '(-)'
        miss += 1
        
    print(flag, "\t True: ", true, "\t Predicted: ", predicted, "\t")
    
    for label in p.keys():
        print(label, ":", round(p[label], 2), "\t")
    
    print("\n")
    
    
print(hit, " out of ", hit + miss, " correct - Accuracy: ", hit / (hit + miss + 0.0))
test.close()