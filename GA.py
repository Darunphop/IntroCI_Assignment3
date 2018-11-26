
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

        for i in range(nPopulation):
            self.population.append(self.initFunc(self.model)[0])

    def getFitness(self):
        return 0