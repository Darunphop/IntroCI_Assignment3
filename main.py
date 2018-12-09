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
            gai = ga.GeneticAlgorithm((MLP.getError, MLP.feedForward), MLP.modelInit, (trainSet[i][:,1:],trainSet[i][:,0]), model, 1, 0.2, 0.1)
            lBest = gai.run(100)
            print(gai.gBest[0])
            print(lBest[0])