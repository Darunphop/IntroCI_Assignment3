import preprocess as pp
import GA as ga
import MLP
import numpy as np
import gc

def test(x):
    return x + 1000

if __name__ == '__main__':
    model = '30x-5t-3t-2s'
    data = pp.preprocess(pp.input('wdbc.data'))
    trainSet, testSet = pp.kFolds(data, 10)
    print(trainSet.shape)
    print(testSet.shape)
    # print(trainSet[0])

    # w,b,act = MLP.modelInit('30x-5t-3t-2s')
    # print(np.concatenate((w[0],w[1],w[2]), axis=None) == np.concatenate((w[0].flatten(),w[1].flatten(),w[2].flatten())))
    # print(np.concatenate((w[0],w[1],w[2]), axis=None))
    # o = MLP.feedForward(trainSet[0][:,1:],w,b,act)
    # print(testSet[0][:,0])
    # print(trainSet[0][:,0])
    # MLP.getError(o[-1],trainSet[0][:,0])
    gai = ga.GeneticAlgorithm((MLP.getError, MLP.feedForward), MLP.modelInit, model, 100, 1, 0.2, 0.1)
    # print(gai.fitness)
    gai.updateFitness((trainSet[0][:,1:],trainSet[0][:,0]))
    # print(gai.fitness)
    print(len(gai.population))
    gai.selection()
    print(len(gai.population))
    # print(gai.population[0][0])
    # print(gai.population[1][0])
    # del gai.population
    # print(gai.population)
    # print(gai.getFitness(test))
    gc.collect()