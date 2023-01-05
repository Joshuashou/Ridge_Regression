import numpy as np
M = 1000 #number of samples constant
def ridgeregressioncalculator(p, rho, n, lam):

    covariance = [[0 for _ in range(p)] for _ in range(p)]

    for r in range(len(covariance)):
        for c in range(len(covariance)):
            if r == c:
                covariance[r][c] = 1
            else:
                covariance[r][c] = rho

    meanarray = [0] * p
    Amatrix = np.identity(p + 1)
    Amatrix[0][0] = 0

    X = np.random.multivariate_normal(meanarray, covariance, (M, n))  # 1000 samples * 100 values * 20 p's
    Ones = np.ones((M, n, 1))
    newX = np.append(Ones, X, axis=2)

    U = np.random.normal(0, 1, M*n)
    firsthalf = np.array(p // 2 * [1])
    secondhalf = np.array(p // 2 * [-1])
    initial = np.array([3])
    betamult = np.append(initial, firsthalf)
    betamult = np.append(betamult, secondhalf)
    # print(betamult)
    U = np.reshape(U, (M, n))
    Y = np.sum([np.dot(newX, betamult), U], axis=0)

    betaarray = []  # Append betaarray values to this array

    ridgearray = np.dot(lam, Amatrix)



    for m in range(M): #Use a for loop to calculate beta values
        Xm = np.array(newX[m, ::]) #100 * 21
        Ym = np.array(Y[m, :]) #100 * 1
        # print("hi")
        # print(np.dot(1 / n, np.dot(np.transpose(Xm), Xm)))
        # print(np.sum([np.dot(1 / n, np.dot(np.transpose(Xm), Xm)), ridgearray], axis=0))


        betaridge = np.dot(np.linalg.inv(np.sum([np.dot(1/n, np.dot(np.transpose(Xm),Xm)), ridgearray], axis=0)),
                           np.dot(1/n, np.dot(np.transpose(Xm), Ym)))

        betaarray.append(betaridge)

    betaarray = np.array(betaarray)

    # print(betaarray)  # are we supposed to have 1000 * 21 beta values ?
    print(betaarray)

    # Now, we calculate average deviation of our estimator betas
    deviationmatrix = []
    # deviation = np.dot(np.transpose(np.subtract(Y, np.dot(newX,np.transpose(betaarray)))),np.subtract(Y, np.dot(newX,np.transpose(betaarray))))


    for m in range(M): #Only this and beta are the parts where I use a for loop, matrix multiplication dimensions don't work.
        Xm = np.array(newX[m, :])  # this has dimensions 100 * 21
        Ym = np.array(Y[m, :]) #100 * 1
        betas = betaarray[m]
        deviation = np.subtract(Ym, np.dot(Xm, betas))
        deviationsquared = np.dot(deviation, np.transpose(deviation))
        deviationmatrix.append(1/n * deviationsquared) #Need to multiply by 1/n for average

    deviationmatrix = np.array(deviationmatrix)

    # Now, we want to calculate the average deviation by taking average over all 1000 M samples.
    deviationaverage = np.average(deviationmatrix)
    return deviationaverage

p = [2,20,70]
rho = [0,0.5,0.9]
n = [100,500,1000]
lam = [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]

#

# print(ridgeregressioncalculator(20,0.99,100, 0.005))
# for p in p:
#     print(ridgeregressioncalculator(p,0.5,500,1))
#
# for lamb in lam:
#    print(str(lamb) +" "+ str(ridgeregressioncalculator(2, 0.1, 500, lamb)))

for i in p:
    for j in n:
        for k in rho:
            print("Deviation for p: " + str(i) + " and rho: " + str(j) + " and n:" + str(k) +" is " +
                  str(ridgeregressioncalculator(i, k, j, lam))