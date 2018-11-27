import numpy as np

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

        itmPopuplation = np.asarray(self.population)[id]
        self.population = np.delete(self.population, id)
        self.fitness = np.delete(self.fitness, id)

        drop = spareSize - len(list(set(id)))
        if drop > 0:
            rIdx = np.argpartition(self.fitness, drop)
            print(rIdx[:drop])
            self.population = np.delete(self.population, rIdx)
            self.fitness = np.delete(self.fitness, rIdx)
        
        self.mating()
        # print(len(itmPopuplation))
        # print(list(set(id)))
        # print(csum[id])
        # self.population = np.delete(self.population, id)
        # print(len(self.population))
        # for i, it in enumerate(csum):
            # print(i,it)
        return 0

    def mating(self):
        return 0