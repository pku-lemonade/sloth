import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import nnls
import os
inertia_weight=0.5
cognitive_constant=1
social_constant=2

class particle:
    def __init__(self, value):
        self.dimensions = len(value)
        self.position=[]
        self.velocity=[]
        self.best_position= []
        self.best_fitness = 0
        for i in range(len(value)):
            self.position.append(value[i])
            self.velocity.append(np.random.uniform(-1, 1))

    def update_velocity(self, global_best_position):
        for i in range(self.dimensions):
            r1 = np.random.rand()
            r2 = np.random.rand()
            cognitive = cognitive_constant * r1 * (self.best_position[i] - self.position[i])
            social = social_constant * r2 * (global_best_position[i] - self.position[i])
            self.velocity[i] = inertia_weight * self.velocity[i] + cognitive + social

    def update_position(self,bounds):
        for i in range(self.dimensions):
            self.position[i] += self.velocity[i]
            if self.position[i] < bounds[0]:
                self.position[i] = bounds[0]
                self.velocity[i] = -self.velocity[i] 
            elif self.position[i] > bounds[1]:
                self.position[i] = bounds[1]
                self.velocity[i] = -self.velocity[i]
    
    def fitness_function(self):
        fitness= fitness_calculator.abs_calculate(self.position)
        if fitness > self.best_fitness or len(self.best_position) == 0:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        return fitness
    
    def mutation(self, bounds):
        self.position=[]
        self.velocity=[]
        self.best_position= []
        self.best_fitness = 0
        for _ in range(self.dimensions):
            self.position.append(np.random.uniform(bounds[0], bounds[1]))
            self.velocity.append(np.random.uniform(-1, 1))
        
class PSO:
    def __init__(self, population_size, dimensions, bounds, lstsq_sol):
        self.population = self.initial_population(population_size, dimensions, bounds, lstsq_sol)
        self.population_size = population_size
        self.bounds=bounds  
        self.global_best_fitness = 0
        self.global_best_position = []
        
    def initial_population(self, population_size, dimensions, bounds, lstsq_sol):
        population = []
        for _ in range(population_size-1):
            position = np.random.uniform(bounds[0], bounds[1], dimensions)
            population.append(particle(position))
        population.append(particle(lstsq_sol))
        return population
        
    def optimize(self, iterations,mutation_generation,num_to_mutate=2):
        best_fitness_history = [0 for _ in range(mutation_generation)]
        best_fitness_cnt=0
        for _ in range(iterations):
            # 计算每个粒子的适应度，并更新全局最优解
            for p in self.population:
                fitness = p.fitness_function()
                if fitness > self.global_best_fitness or len(self.global_best_position) == 0:
                    self.global_best_fitness = fitness
                    self.global_best_position = p.position.copy()
                    
            # 更新每个粒子的速度和位置
            for p in self.population:
                p.update_velocity(self.global_best_position)
                p.update_position(self.bounds)
            if int(self.global_best_fitness) == 0:
                return self.global_best_position
            
            # 记录最优适应度，适时进行变异
            if best_fitness_cnt != 0 and best_fitness_history[best_fitness_cnt-1] != int(self.global_best_fitness) :
                best_fitness_cnt=0
            best_fitness_history[best_fitness_cnt]= int(self.global_best_fitness)
            best_fitness_cnt+=1
            if best_fitness_cnt >= mutation_generation:
                best_fitness_cnt = 0
                indices_to_mutate = np.random.choice(self.population_size, num_to_mutate, replace=False)
                for idx in indices_to_mutate:
                    self.population[idx].mutation(self.bounds)
        return self.global_best_position
                
class fitness_function:
    def __init__(self, adjacency_matrix, selected_route, end_to_end_delays, targeted_links):
        self.adjacency_matrix = adjacency_matrix
        self.selected_route = selected_route
        self.end_to_end_delays = end_to_end_delays
        self.targeted_links = targeted_links
        
    def abs_calculate(self,x):
        sum=0
        for i in range(len(self.selected_route)):
            tmp=0
            for j in range(len(self.selected_route[i])):
                tmp+=self.selected_route[i][j]*x[j]
            tmp-= self.end_to_end_delays[i]
            sum+=abs(tmp)
        return -sum
        
def delay_vector_calculator(adjacency_matrix,selected_route,end_to_end_delays,targeted_links,min=1,max=10000,population_size=50,iterations=90,mutation_generation=50):
    global fitness_calculator
    fitness_calculator = fitness_function(adjacency_matrix, selected_route, end_to_end_delays, targeted_links)
    np.random.seed(400)
    lstsq_sol = nnls(selected_route, end_to_end_delays)[0]
    pso= PSO(population_size, len(targeted_links), (min, max), lstsq_sol)
    return pso.optimize(iterations,mutation_generation)
    
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file1= os.path.join(script_dir, "adj_mat.csv") 
    
    adjacency_matrix=[]
    with open(csv_file1, 'r') as file1:
        for line in file1:
            adjacency_matrix.append([float(x) for x in line.strip().split(',')])
    csv_file2= os.path.join(script_dir, "inference.csv") 
    data_list=[]
    with open(csv_file2, 'r') as file2:
        for line in file2:
            data_list.append([float(x) for x in line.strip().split(',')])
    selected_route = [row[1:-1] for row in data_list] 
    end_to_end_delays = [row[-1] for row in data_list]                                                                                                                                                                                                                                    
    delay_vector = delay_vector_calculator(adjacency_matrix, selected_route[:100], end_to_end_delays[:100], range(len(selected_route[0])))
    
    print("Calculated Delay Vector:", delay_vector)
