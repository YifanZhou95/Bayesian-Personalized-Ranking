# -*- coding: utf-8 -*-
"""
Assignment 1 (rating_predict) -- UCSD CSE 258

Created on Sat Nov 18 09:49:44 2017

@author: zyf
"""

import numpy as np
import scipy.optimize
import random
import gzip
from collections import defaultdict
import matplotlib.pylab as plt

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

rawData = []
for l in readGz("train/train.json.gz"):
    rawData.append(l)

size = len(rawData)
allPair = []
for l in rawData[:size]:
    user,business,rating = l['userID'],l['businessID'],l['rating']
    allPair.append((user,business,rating))  

trainSize = int(0.99*size)
trainPair = allPair[:trainSize]
validPair = allPair[trainSize:]
uniUser = np.unique([p[0] for p in allPair[:trainSize]])
uniBusi = np.unique([p[1] for p in allPair[:trainSize]])

userBusi_train = defaultdict(list)
busiUser_train = defaultdict(list)
for l in rawData[:trainSize]:
    user,busi,rating = l['userID'],l['businessID'],l['rating']
    userBusi_train[user].append((busi,rating))
    busiUser_train[busi].append((user,rating))

# evaluate performance
def predict(user, busi, theta):
    length = len(theta)
    if length == 3:
        alpha,betaU,betaB = theta
    elif length == 5:
        alpha,betaU,betaB,gammaU,gammaB = theta
    else:
        print 'theta do not match!'
        return 0    
    beta_u = betaU[user] if user in betaU else 0
    beta_b = betaB[busi] if busi in betaB else 0
    gamma_u = gammaU[user] if length==5 and user in gammaU else np.zeros(1)
    gamma_b = gammaB[busi] if length==5 and busi in gammaB else np.zeros(1)
#    if user in uniUser:
#        beta_u = betaU[user]
#        if length == 5:
#            gamma_u = gammaU[user]
#    if busi in uniBusi:
#        beta_b = betaB[busi]
#        if length == 5:
#            gamma_b = gammaB[busi]      
    return alpha + beta_u + beta_b + sum(gamma_u*gamma_b)

def meanSquaredError(dataset,theta):
    size = len(dataset)
    error = 0
    for (user,busi,rating) in dataset:      
        prediction = predict(user,busi,theta)
        error = error + (prediction-rating)**2/size
#        count += 1
#        if count%5000==0:
#            print count
    print 'MSE: ' + str(error)
    return error


# latent factor model (alpha, beta)
def LFM1(lam, max_iter):
    alpha = 0.1
    betaU = {i:0.0 for i in uniUser}
    betaB = {i:0.0 for i in uniBusi}
    for it in range(max_iter):
        alpha = 0
        loss = 0
        for (user, busi, rating) in trainPair:
            alpha += (rating - betaU[user] - betaB[busi])/len(trainPair)
        for (user, busi, rating) in trainPair:
            diff = alpha + betaU[user] + betaB[busi] - rating
            loss += diff**2
        squareError = loss
        for (user, busi_list) in userBusi_train.items():
            betaU[user] = 0
            for (busi,rating) in busi_list:
                betaU[user] += (rating - alpha - betaB[busi])/(lam + \
                     len(busi_list))
            loss += lam*(betaU[user]**2)
        for (busi, user_list) in busiUser_train.items():
            betaB[busi] = 0
            for (user,rating) in user_list:
                betaB[busi] += (rating - alpha - betaU[user])/(lam + \
                     len(user_list))
            loss += lam*(betaB[busi]**2)
        if (it+1)%5==0:
            print 'iteration: ' + str(it+1)
            print loss, squareError
            meanSquaredError(validPair,[alpha,betaU,betaB])
    return alpha, betaU, betaB

alpha, betaU, betaB = LFM1(lam = 4.0, max_iter = 30)


# latent factor model (alpha, beta, gamma)
def LFM2(lam, K, learnRate, max_iter):
    errList = []
    alpha = 0
    gammaU = {i:np.random.rand(K)/10-0.05 for i in uniUser}
    gammaB = {i:np.random.rand(K)/10-0.05 for i in uniBusi}
    betaU = {i:0.2 for i in uniUser}
    betaB = {i:0.2 for i in uniBusi}
    for it in range(max_iter):
        alpha = 0
        loss = 0

        gra_gammaU = {user:lam*gammaU[user] for user in uniUser}
        gra_gammaB = {busi:lam*gammaB[busi] for busi in uniBusi}
        for (user, busi, rating) in trainPair:
            alpha += (rating - betaU[user] - betaB[busi] - np.inner(gammaU[user],gammaB[busi]))/len(trainPair)
        
        for (user, busi, rating) in trainPair:
            diff = alpha + betaU[user] + betaB[busi] + np.inner(gammaU[user],gammaB[busi]) - rating
            gra_gammaU[user] += gammaB[busi]*diff
            gra_gammaB[busi] += gammaU[user]*diff
            loss += diff**2
        squareError = loss
        for (user, busi_list) in userBusi_train.items():
            gammaU[user] -= learnRate*gra_gammaU[user]
            betaU[user] = 0
            for (busi,rating) in busi_list:
                betaU[user] += (rating - alpha - betaB[busi] - np.inner(gammaU[user],gammaB[busi]))/(lam + len(busi_list))
            loss += lam*(betaU[user]**2+sum(gammaU[user]**2))
        for (busi, user_list) in busiUser_train.items():
            gammaB[busi] -= learnRate*gra_gammaB[busi]
            betaB[busi] = 0
            for (user,rating) in user_list:
                betaB[busi] += (rating - alpha - betaU[user] - np.inner(gammaU[user],gammaB[busi]))/(lam + len(user_list))
            loss += lam*(betaB[busi]**2+sum(gammaB[busi]**2))
        if (it+1)%5==0:
            print 'iteration: ' + str(it+1)
            print loss, squareError
            err = meanSquaredError(validPair,[alpha,betaU,betaB,gammaU,gammaB])
            errList.append(err)
    return alpha,betaU,betaB,gammaU,gammaB

#alpha,betaU,betaB,gammaU,gammaB = LFM2(lam = 10.0, K = 1, learnRate = 0.05, max_iter = 500)
# LFM2 likelihood: 76668 (lam = 4.0, K = 1, learnRate = 0.04, max_iter = 500)

# latent factor model (alpha, beta, gamma)
def LFM3(lam, K, learnRate, max_iter):
    alpha, betaU, betaB = LFM1(lam = 4.0, max_iter = 30)
    errList = []
    gammaU = {i:np.random.rand(K)/10-0.05 for i in uniUser}
    gammaB = {i:np.random.rand(K)/10-0.05 for i in uniBusi}
    for it in range(max_iter):
        loss = 0

        gra_gammaU = {user:lam*gammaU[user] for user in uniUser}
        gra_gammaB = {busi:lam*gammaB[busi] for busi in uniBusi}

        for (user, busi, rating) in trainPair:
            diff = alpha + betaU[user] + betaB[busi] + \
                                np.inner(gammaU[user],gammaB[busi]) - rating
            gra_gammaU[user] += gammaB[busi]*diff
            gra_gammaB[busi] += gammaU[user]*diff
            loss += diff**2
        squareError = loss
        for (user, busi_list) in userBusi_train.items():
            gammaU[user] -= learnRate*gra_gammaU[user]
            loss += lam*(betaU[user]**2+sum(gammaU[user]**2))
        for (busi, user_list) in busiUser_train.items():
            gammaB[busi] -= learnRate*gra_gammaB[busi]
            loss += lam*(betaB[busi]**2+sum(gammaB[busi]**2))
        if (it+1)%10==0:
            print 'iteration: ' + str(it+1)
            print loss, squareError
            err = meanSquaredError(validPair,[alpha,betaU,betaB,gammaU,gammaB])
            errList.append(err)
    return alpha,betaU,betaB,gammaU,gammaB,errList

alpha,betaU,betaB,gammaU,gammaB,errList = LFM3(lam = 10.0, K = 2, \
                                               learnRate = 0.05, \
                                               max_iter = 500)
plt.figure()
plt.plot(errList)
# lam=12 K=4 rate=0.06
# lam=10 K=4 rate=0.06 (good but slow)
# lam=10 K=2 rate=0.05 (fast)

MSE_train1 = meanSquaredError(trainPair,[alpha,betaU,betaB])
MSE_train2 = meanSquaredError(trainPair,[alpha,betaU,betaB,gammaU,gammaB])

MSE_valid1 = meanSquaredError(validPair,[alpha,betaU,betaB])
MSE_valid2 = meanSquaredError(validPair,[alpha,betaU,betaB,gammaU,gammaB])

#lamList = [x/10.0+3.5 for x in range(20)]
#mseList = []
#for lam in lamList:
#    #alpha,betaU,betaB = LFM1(lam)
#    mseList.append(meanSquaredError(validPair,LFM1(lam, max_iter = 20)))
#plt.plot(lamList,mseList)
#print lamList[mseList.index(min(mseList))]

# predict on test set and write to a new file
def testPredict():
    predictions = open("test/predictions_Rating.txt", 'w')
    for l in open("test/pairs_Rating.txt"):
        if l.startswith("userID"):
            #header
            predictions.write(l)
            continue
        u,i = l.strip().split('-')
        prediction = predict(u,i,(alpha,betaU,betaB,gammaU,gammaB))
        predictions.write(u + '-' + i + ',' + str(prediction) + '\n')
    predictions.close()
    
testPredict()