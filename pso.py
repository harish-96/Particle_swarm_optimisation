import numpy as np
import random
import matplotlib.pyplot as plt


class Particle:
    """A single particle in the swarm"""

    def __init__(self, dimension, X_max=5, X_min=-5):
        self.position = []
        self.velocity = []
        for d in dimension:
            self.position.append(((X_max - X_min) *
                np.random.rand(d) + X_min))
            self.velocity.append((0.1 * (X_max - X_min) * np.random.rand(d) +
                                 0.1 * X_min))
        self.position = np.array(self.position)
        self.velocity = np.array(self.velocity)
        self.pbest = self.position

    def update_pbest(self, error_function):
        if error_function(self.position) < error_function(self.pbest):
            self.pbest = self.position


class Swarm(Particle):

    def __init__(self, numParticles, error_function, pdeath=0.005,
                 dimension=[784 * 15, 15 * 10, 15, 10], w=0.729, c1=1.49445,
                 c2=1.49445, exit_error=0.1):
        self.dimension = dimension
        self.exit_error = exit_error
        self.error_function = error_function
        self.swarm = np.asarray([Particle(self.dimension)
                                 for i in range(numParticles)])
        gbest_index = np.argmin([self.error_function(self.swarm[i].position)
                                 for i in range(numParticles)])
        self.gbest = self.swarm[gbest_index].position
        self.numParticles = numParticles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.pdeath = pdeath

    def update_gbest(self):
        gbest_index = np.argmin([self.error_function(self.swarm[i].position)
                                 for i in range(self.numParticles)])
        self.gbest = self.swarm[gbest_index].position

    def update_step(self):
        for i in range(self.numParticles):
            self.r1, self.r2 = np.random.rand(2)
            self.swarm[i].velocity = (self.w * self.swarm[i].velocity +
                                self.c1 * self.r1 * (self.swarm[i].pbest - self.swarm[i].position) +
                                self.c2 * self.r2 * (self.gbest - self.swarm[i].position))
            self.swarm[i].position = (self.swarm[i].position +
                                      self.swarm[i].velocity)
            self.swarm[i].update_pbest(self.error_function)
            
            rnd = random.random()
            if rnd < self.pdeath:
                self.swarm[i].__init__(self.dimension)
        self.update_gbest()

    def optimise(self, iterations=100):
        it = 0
        J = []
        while self.error_function(self.gbest) > self.exit_error:
            J.append(self.error_function(self.gbest))
#            if it % 100 == 0:
#                self.display_positions()
#                plt.show()
            self.update_step()
            it += 1
            print("iteration: ",it, "cost: ", J[-1])
            if it > iterations: break
        plt.plot(J); plt.show()
        return J
    
    def display_positions(self):
        x_pos = [self.swarm[i].position[0] for i in range(self.numParticles)]
        y_pos = [self.swarm[i].position[1] for i in range(self.numParticles)]
        plt.plot(x_pos, y_pos, 'r*')
        # plt.plot(x_pos, 'r*')
            

