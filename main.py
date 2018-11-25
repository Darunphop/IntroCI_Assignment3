import preprocess as pp
import MLP

if __name__ == '__main__':
    data = pp.preprocess(pp.input('wdbc.data'))
    trainSet, testSet = pp.kFolds(data, 10)
    print(trainSet.shape)
    print(testSet.shape)
    # print(trainSet[0])

    w,b,act = MLP.modelInit('30x-5t-3t-2s')
    # print(trainSet[0][:,1:])
    o = MLP.feedForward(trainSet[0][:,1:],w,b,act)
    print(o[2])