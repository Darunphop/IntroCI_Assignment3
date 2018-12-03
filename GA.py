import numpy as np
import copy

class GeneticAlgorithm:
    def __init__(self, fitnessFunc, initFunc, seed, nPopulation, it, sRate, mRate):
        self.population = []
        self.fitnessFunc = fitnessFunc
        self.initFunc = initFunc
        self.model = seed
        self.nPopulation = nPopulation
        self.it = it
        self.sRate = sRate
        self.mRate = mRate

        self.fitness = np.zeros(nPopulation)
        self.act = self.initFunc(self.model)[2]

        for i in range(nPopulation):
            self.population.append(self.initFunc(self.model)[0])

    def updateFitness(self, data):
        for i in range(self.nPopulation):
            o = self.fitnessFunc[1](data[0],self.population[i],0,self.act)
            self.fitness[i] = self.fitnessFunc[0](o[-1],data[1])

    def selection(self):
        csum = np.cumsum(self.fitness)
        spareSize = (int(self.nPopulation * self.sRate) >> 1) << 1

        id = []
        for i in range(spareSize):
            x = csum[-1] * np.random.ranf()
            id.append(np.argmax(csum>x))

        # sum1 = 0
        # for i in self.population:
        #     x = np.concatenate((i[0],i[1],i[2]), axis=None)
        #     sum1 += x.sum()

        tmp = np.arange(100)
        itmPopuplation = np.asarray(self.population)[id]
        itmTmp = tmp[id]
        tmp = np.delete(tmp, id)
        self.population = np.delete(self.population, id, axis=0).tolist()
        self.fitness = np.delete(self.fitness, id)

        # sum1 = 0
        # for i in self.population:
        #     x = np.concatenate((i[0],i[1],i[2]), axis=None)
        #     sum1 += x.sum()
        # print('after sum1', sum1)
        drop = spareSize - len(list(set(id)))
        if drop > 0:
            rIdx = np.argpartition(self.fitness, drop)
            print(rIdx[:drop])
            self.population = np.delete(self.population, rIdx)
            self.fitness = np.delete(self.fitness, rIdx)
        
        matedPop = self.mating(itmPopuplation)
        mutatedPop = self.mutate(matedPop)
        return 0

    def mating(self, population):
        res = []
        np.random.shuffle(population)
        for i in range(int(population.shape[0]/2)):
            if i != -1:
                res.extend(self.crossover(population[i], population[i+1]))
        return np.asarray(res)

    def crossover(self, i1, i2):
        shape = [(i.shape[0],i.shape[1]) for i in i1]
        size = np.cumsum([i.shape[0]*i.shape[1] for i in i1])[:-1]
        chromosome1 = np.concatenate((i1[0],i1[1],i1[2]), axis=None)
        chromosome2 = np.concatenate((i2[0],i2[1],i2[2]), axis=None)

        cp = [0,0]

        while abs(cp[0] - cp[1]) < 0.10*chromosome1.shape[0]: 
            cp = np.random.randint(0,chromosome1.shape[0],2)
        cp.sort()

        tmp = chromosome1[cp[0]:cp[1]].copy()
        chromosome1[cp[0]:cp[1]] = chromosome2[cp[0]:cp[1]].copy()
        chromosome2[cp[0]:cp[1]] = tmp

        w1 = [i.reshape(shape[it]) for it,i in enumerate(np.split(chromosome1, size))]
        w2 = [i.reshape(shape[it]) for it,i in enumerate(np.split(chromosome2, size))]

        return [w1,w2]

    def mutate(self, population):
        return 0