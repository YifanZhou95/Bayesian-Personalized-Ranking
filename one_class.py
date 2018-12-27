# -*- coding: utf-8 -*-
"""
Assignment 1 (visit_predict) -- UCSD CSE 258

Created on Sun Nov 12 09:45:30 2017

@author: zyf
"""
import numpy as np
from math import exp
from math import log
import scipy.optimize
import random
import gzip
from collections import defaultdict
import matplotlib.pylab as plt

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

rawData = []
for l in readGz("train.json.gz"):
    rawData.append(l)

size = len(rawData)
trainSize = int(0.99*size)
validSize = size - trainSize


allPair = []
userBusi_train = defaultdict(list)
busiUser_train = defaultdict(list)

count = 0
for l in rawData[:size]:
    count += 1
    user,business = l['userID'],l['businessID']
    allPair.append((user,business))    
    if count < trainSize:
        userBusi_train[user].append(business)
        busiUser_train[business].append(user)

trainPair = allPair[:trainSize]
validPair = allPair[trainSize:]
uniUser = np.unique([p[0] for p in trainPair])
uniBusi = np.unique([p[1] for p in trainPair])
Lu = len(uniUser)
Lb = len(uniBusi)
userDict = {uniUser[i]:i for i in range(len(uniUser))}
busiDict = {uniBusi[i]:i for i in range(len(uniBusi))}

# add negative samples
validPair_plus = allPair[trainSize:]
samplingTimes = 0
negativeTimes = 0
while negativeTimes < size-trainSize:
    samplingTimes += 1
    u = random.choice(uniUser)
    b = random.choice(uniBusi)
    if b in userBusi_train[u]:
        continue
    else:
        validPair_plus.append((u,b))
        negativeTimes += 1


### similarity ###
#userDict = {uniUser[i]:i for i in range(len(uniUser))}
#
#busiUser_train = defaultdict(list)
#for l in rawData[:trainSize]:    
#    user,business = l['userID'],l['businessID']
#    busiUser_train[business].append(user)
#
#def Jaccard(listA, listB):
#    deno = len(np.unique(listA+listB))
#    nume = len(listA) + len(listB) - deno
#    return 1.0*nume/deno
#
#def Cosine(listA, listB):   # computing is too slow
#    vectA = np.zeros(len(uniUser))
#    vectB = np.zeros(len(uniUser))
#    for u in listA:
#        vectA[userDict[u]] = 1
#    for u in listB:
#        vectB[userDict[u]] = 1
#    nume = np.dot(vectA,vectB)
#    deno = (sum(vectA**2)*sum(vectB**2))**0.5
#    return nume/deno
#
#busiSimi_train = defaultdict(list)
#count = 0
#for it in uniBusi[:]:
#    simList = [(Jaccard(busiUser_train[it],busiUser_train[itprime]),itprime) for itprime in uniBusi[:]]
#    busiSimi_train[it] = sorted(simList, reverse = True)[1:501]
#    count += 1
#    if count%10 == 0:
#        print count
#
#userPot_train = defaultdict(list)
#userPot_train2 = defaultdict(list)
#for u in uniUser:
#    for it in userBusi_train[u]:
#        for score, itprime in busiSimi_train[it]:
#            if score > 0.04:
#                userPot_train[u].append(itprime)
#            if score > 0.08:
#                userPot_train2[u].append(itprime)
#
#def JaccardPredict((u,b)):
#    return 1 if b in userPot_train[u] else 0
#
#predictions = [JaccardPredict(p) for p in validPair_plus]
#accuracy = sum(predictions[:validSize])*1.0/(validSize*2) + 0.5 \
#              - sum(predictions[validSize:])*1.0/(validSize*2)
#print sum(predictions[:validSize])
#print sum(predictions[validSize:])
#print accuracy
# Jaccard similarity: accuracy is 0.83!



def sigmoid(x):
    return 1.0 / (1 + exp(-x))

def findNegBusi(user):
    while 1:
        negBusi = random.choice(uniBusi)
        if negBusi not in userBusi_train[user]:
            return negBusi

def findPosBusi(user):
    return random.choice(userBusi_train[user])

# evaluate
def gammaPredict((user,busi), gammaU, gammaB, threshold): # 0.2*thresholdDict[user]
    if user in userDict and busi in busiDict:
        u = userDict[user]
        b = busiDict[busi]
        return 1 if np.dot(gammaU[u,:],gammaB[:,b]) > threshold else 0
    else:
        return 0

def Accuracy(gammaU, gammaB, threshold):
    valid_predictions = [gammaPredict(p,gammaU,gammaB,threshold) for p in validPair_plus]
    valid_accuracy = sum(valid_predictions[:validSize])*1.0/(validSize*2) + \
                        0.5 - sum(valid_predictions[validSize:])*1.0/(validSize*2)
    return valid_accuracy

def gammaJaccardPredict((user,busi), gammaU, gammaB, threshold, offset):
    if user in userDict and busi in busiDict:
        bonus = offset if busi in userPot_train[user] else 0
        #bonus += offset if busi in userPot_train2[user] else 0
        u = userDict[user]
        b = busiDict[busi]
        return 1 if np.dot(gammaU[u,:],gammaB[:,b])+bonus > threshold else 0
    else:
        return 0   

def OneClass(lam, K, learnRate, max_iter):
    gammaU = np.random.rand(Lu, K)/1 - 0.5
    gammaB = np.random.rand(K, Lb)/1 - 0.5
    accRec = []
    for it in range(max_iter):
        objective = 0
        regularization = 0
        gu = np.zeros((Lu, K))
        gb = np.zeros((K, Lb))
        
        for (user, busi) in trainPair:
            u = userDict[user]
            i = busiDict[busi]
            j = busiDict[findNegBusi(user)]
            z = sigmoid(np.dot(gammaU[u,:],gammaB[:,i]) - \
                        np.dot(gammaU[u,:],gammaB[:,j]))
            gu[u,:] += (1-z)*(gammaB[:,i]-gammaB[:,j])
            gb[:,i] += (1-z)*(gammaU[u,:])
            gb[:,j] += (1-z)*(-gammaU[u,:])
            
            objective += log(z)
        
        gu -= lam*gammaU
        gb -= lam*gammaB
        gammaU += learnRate*gu
        gammaB += learnRate*gb 
        
        regularization = objective - lam*np.sum(np.square(gammaU)) - \
                                           lam*np.sum(np.square(gammaB))        
        if (it+1)%2==0:
            print 'iteration: ' + str(it+1) + '\t' + str(regularization) \
                                     + '\t' + str(objective)
        if (it+1)%10==0:
            accCur = max([Accuracy(gammaU,gammaB,t) for t in \
                          [0.3,0.4,0.5,0.6,0.7]])
            accRec.append(accCur)
            print accCur
            
        if it+1 == max_iter - 100:
            gammaU1 = gammaU
            gammaB1 = gammaB
        if it+1 == max_iter - 50:
            gammaU2 = gammaU
            gammaB2 = gammaB
    
    return gammaU, gammaB, accRec, gammaU1, gammaB1, gammaU2, gammaB2

gammaU,gammaB,accRec,gammaU1,gammaB1,gammaU2,gammaB2 = OneClass(lam = 0.6,K = 400,learnRate = 0.1,max_iter = 360)
plt.figure(figsize=(8,6))
plt.plot(accRec, linewidth = '2')
plt.xlabel("iteration (x10)")  
plt.ylabel("accuracy")  
plt.title("Performance on validation set")

thList = [k/100.0+0.3 for k in range(50)]
accList = [Accuracy(gammaU, gammaB, th) for th in thList]
plt.figure(figsize=(8,6))
plt.plot(thList, accList, linewidth = '2')
plt.xlabel("threshold")  
plt.ylabel("accuracy")  
plt.title("Performance on validation set")

print max(accList)
print thList[accList.index(max(accList))]

# incorrect SGD
# K = 40, lambda = 0.2: threshold = 0.21
# K = 200, lambda = 0.1: threshold = 0.45
# K = 500, lambda = 0.1: threshold = 0.43
 
# correct SGD (batch)
# K = 100, lambda = 0.4: threshold = 0.53
# K = 400, lambda = 0.4: threshold = 0.48
# K = 200, lambda = 0.5: threshold = 0.48
# K = 400, lambda = 0.6: threshold = 0.50 - c/250
# K = 800, lambda = 0.54 (good)

# dynamic threshold based on user activity
def activitySet(trainPair):    
    userCount = defaultdict(int)
    totalPurchases = 0    
    for p in trainPair:
        user = p[0]
        userCount[user] += 1
        totalPurchases += 1
        
    mostActive = [(userCount[x], x) for x in userCount]
    mostActive.sort()
    mostActive.reverse()
    return mostActive

actUser = activitySet(trainPair)

for upbound in [x/50.0+0.48 for x in range(6)]:
    print upbound
    for scale in [1.0,1.2,1.5,1.8,2.0,2.5]:
        thresholdDict = defaultdict(int)
        for (c,user) in actUser:
            thresholdDict[user] = upbound - scale*(1-exp(-c/100.0))
        print scale,
    
        valid_predictions = [gammaPredict(p,gammaU,gammaB,\
                                                 thresholdDict[p[0]]) for p in validPair_plus]
        valid_accuracy = sum(valid_predictions[:validSize])*1.0/(validSize*2) + \
                            0.5 - sum(valid_predictions[validSize:])*1.0/(validSize*2)
    #        print sum(valid_predictions[:validSize])
    #        print sum(valid_predictions[validSize:])
        print valid_accuracy
 
thresholdDict = defaultdict(int)
for (c,user) in actUser:
    thresholdDict[user] = 0.52 - 0.7*(c**0.9/120.0)
valid_predictions = [gammaPredict(p,gammaU,gammaB,\
                                         thresholdDict[p[0]]) for p in validPair_plus]
valid_accuracy = sum(valid_predictions[:validSize])*1.0/(validSize*2) + \
                    0.5 - sum(valid_predictions[validSize:])*1.0/(validSize*2)
print sum(valid_predictions[:validSize])
print sum(valid_predictions[validSize:])
print valid_accuracy    

# predict on test set and write to a new file
def testPredict():
    predictions = open("predictions_Visit.txt", 'w')
    for l in open("pairs_Visit.txt"):
        if l.startswith("userID"):
            #header
            predictions.write(l)
            continue
        u,i = l.strip().split('-')
        #prediction = 1 if (u in actSet or i in popSet) else 0  # popularity + activity
        prediction = gammaPredict((u,i),gammaU,gammaB,thresholdDict[u])
        predictions.write(u + '-' + i + ',' + str(prediction) + '\n')
    predictions.close()
    
#testPredict()

