import  numpy as np
from text_recog import *


class Particle:
    """A single particle in the swarm"""

    def __init__(self, dimension, X_max=-5, X_min=5):
        self.position = ((X_max - X_min) *
                         np.random.randn(dimension[0], dimension[1]) + X_min)
        self.velocity = (0.1 * (X_max - X_min) * np.random.randn(
            dimension[0], dimension[1])) + 0.1 * X_min
        self.pbest = self.position


    def update_pbest(self, error_function)
        if error_function(self.pbest) < error_function(self.position):
            self.pbest = self.position

class Swarm(Particle):

    def __init__(self, numParticles, error_function, pdeath=0.005,
                 dimension=[784 * 15, 15 * 10], w=0.729, c1=1.49445, c2=1.49445):
        self.swarm = np.asarray([Particle(dimension) for i in range(numParticles)])
        gbest_index = np.argmin([error_function(swarm[i].position)])
        self.gbest = swarm[gbest_index].position
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.r1, self.r2 = np.random.randn(2)
    
    def update_gbest(self, error_function):
        gbest_index = np.argmin([error_function(swarm[i].position)])
        self.gbest = swarm[gbest_index].position

    def upate_step(self):
        for i in range(numParticles):
            self.r1, self.r2 = np.random.randn(2)
            swarm[i].velocity = (self.w * swarm[i].velocity) +
                                 self.c1 * self.r1 * (swarm[i].pbest - swarm[i].position +
                                 self.c2 * self.r2 * (self.gbest - swarm[i].position)
            self.r1, self.r2 = np.random.randn(2)
            swarm[i].position = swarm[i].position + swarm[i].velocity
            swarm[i].update_pbest(error_function)
        self.update_gbest(error_function)





