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
        x = csum[-1] * np.random.ranf()
        id = np.argmax(csum>x)
        # print(x, id)
        # print(zip(csum))
        # for i, it in enumerate(csum):
            # print(i,it)
        return 0 