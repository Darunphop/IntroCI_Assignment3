import preprocess as pp
import GA as ga
import MLP
import numpy as np
import gc

def test(x):
    return x + 1000

if __name__ == '__main__':
    model = '30x-5t-3t-2s'
    k = 10
    data = pp.preprocess(pp.input('wdbc.data'))
    trainSet, testSet = pp.kFolds(data, k)
    print(trainSet.shape)
    print(testSet.shape)

    for i in range(k):
        if i == 0:
            gai = ga.GeneticAlgorithm((MLP.getError, MLP.feedForward), MLP.modelInit, (trainSet[i][:,1:],trainSet[i][:,0]), model, 50, 0.2, 0.1)
            lBests = gai.run(100)
            print(gai.gBest[0])
            # print(gai.fitness)
            testData = (testSet[i][:,1:],testSet[i][:,0])
            oG = gai.fitnessFunc[1](testData[0],gai.gBest[1],0,gai.act)
            gError = gai.fitnessFunc[0](oG[-1],testData[1])
            print('gE', gError)
            print(lBests)