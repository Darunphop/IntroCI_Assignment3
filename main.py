import preprocess as pp
import GA as ga
import MLP
import matplotlib.pyplot as plt
import numpy as np
import gc

def writeFile(fname, content):
    sContent = ''
    for s in content:
        for i in s:
            sContent += str(i) + ' '
        sContent += '\n'
    f = open(fname, 'w')
    f.write(sContent)
    f.close()

if __name__ == '__main__':
    model = '30x-5t-3t-2s'
    k = 10
    epoch = 100
    data = pp.preprocess(pp.input('wdbc.data'))
    trainSet, testSet = pp.kFolds(data, k)
    print(trainSet.shape)
    print(testSet.shape)

    res = []
    for i in range(k):
        if i > -1:
            gai = ga.GeneticAlgorithm((MLP.getError, MLP.feedForward), MLP.modelInit, (trainSet[i][:,1:],trainSet[i][:,0]), model, 5, 0.2, 0.1)
            lBests = gai.run(epoch)
            testData = (testSet[i][:,1:],testSet[i][:,0])
            oG = gai.fitnessFunc[1](testData[0],gai.gBest[1],0,gai.act)
            gError = gai.fitnessFunc[0](oG[-1],testData[1])
            lAcc = []
            for j in range(epoch):
                oTmp = gai.fitnessFunc[1](testData[0],lBests[j][1],0,gai.act)
                lAcc.append(gai.fitnessFunc[0](oTmp[-1],testData[1]))

            fig = plt.figure(i+1)
            plt.title('Fold '+str(i+1))
            plt.plot(np.arange(epoch), lAcc, 'bo-', label='Test by each iteration BEST', ms=5)
            plt.plot(np.arange(epoch), np.asarray([gError for i in range(epoch)]), 'r-', label='Test by Global BEST', ms=5)
            plt.legend(loc='best')
            fig.savefig('exp1,'+str((i+1))+'.png')

            res.append([gError, np.max(lAcc), lAcc[-1]])
    
    writeFile('out.txt', res)