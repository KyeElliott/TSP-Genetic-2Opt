#!/usr/bin/python3

from functools import reduce
from pdb import post_mortem
from queue import PriorityQueue
from typing import MutableSet

from pkg_resources import evaluate_marker
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import copy
import numpy as np
from TSPClasses import *
import heapq
import itertools

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour(self, time_allowance = 60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time() - start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			#print(perm)
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	#Time Complexity: O(n^3)
	#Space Complexity: O(n) - since we have max n cities on the stack at any given point
	#If we account for the adjacency matrix then the space complexity is actually O(n^2)
	def greedy(self,time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		stack_index = 0
		foundTour = False
		count = 0
		bssf = None
		self.greedy_route = None
		start_time = time.time()

		#Algorithm tries to find a greedy path starting from city A; if a path cannot be found, the algorithm instead starts from city B, then city C, etc.
		while not foundTour and stack_index != ncities and time.time() - start_time < time_allowance:
			city_stack = []
			
			for city in cities:
				city_stack.append(city)

			route = []

			current_city = city_stack.pop(stack_index)
			first_city = current_city
			stack_index += 1

			route.append(first_city)

			while len(city_stack) > 0 and time.time() - start_time < time_allowance:

				smallest_edge = np.inf
				dest_city = None

				index = -1
				
				#Choose the shortest edge coming from the current city; once we have the dest city, pop it off the queue
				for i in range(len(city_stack)):
					if current_city.costTo(city_stack[i]) < smallest_edge:
						smallest_edge = current_city.costTo(city_stack[i])
						dest_city = city_stack[i]
						index = i
				
				city_stack.pop(index)

				if dest_city is not None:
					current_city = dest_city

					if len(city_stack) == 0:
						#Once the stack is empty, make sure we can still get back to the starting city
						if dest_city.costTo(first_city) < float('inf'):
							route.append(current_city)
							foundTour = True

							#Used later for fancy algorithm
							self.greedy_route = copy.deepcopy(route)
							self.greedy_route.append(first_city)

							count += 1
							break

						else:
							break

					route.append(current_city)

				else:
					break

		bssf = TSPSolution(route)
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	class Subproblem:
		def __init__(self):
			self.cost_matrix = [[None], [None]]
			self.tour_so_far = []
			self.lower_bound = 0
			self.cities_remaining = 0

		def __lt__(self, rhs):
			if (self.lower_bound < rhs.lower_bound and self.cities_remaining <= rhs.cities_remaining):
				return True
			else:
				return False


	#Total time complexity for this function is going to be O(n^2 * max_queue_size)
	#Since we go by the worst case scenario (every subproblem gets expanded), the time complexity is O(n^2(n + 1)!)
	#Total space complexity: O(n^2 * max_queue_size)
	def branchAndBound(self, time_allowance = 60.0):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		bssf = self.greedy(time_allowance)['soln']
		num_solutions = 0
		max_queue_size = 0
		total_states_created = 0
		num_pruned_states = 0
		start_time = time.time()

		priorityQueue = []
		heapq.heapify(priorityQueue)

		cost_matrix = np.zeros((ncities, ncities))

		#Creating the initial reduced matrix
		for i in range(ncities):
			for j in range(ncities):
				cost_matrix[i][j] = cities[i].costTo(cities[j])

		reduced_cost_matrix, initial_lower_bound = self.reduceMatrix(cost_matrix, ncities)

		city_index = 0

		first_city = self.Subproblem()
		first_city.tour_so_far.append(city_index)
		first_city.cost_matrix = reduced_cost_matrix
		first_city.lower_bound = initial_lower_bound
		first_city.cities_remaining = ncities - 1

		heapq.heappush(priorityQueue, first_city)

		city_index = 0

		while len(priorityQueue) != 0 and time.time() - start_time < time_allowance:
			
			if len(priorityQueue) > max_queue_size:
				max_queue_size = len(priorityQueue)

			reduced_cost_matrix = priorityQueue[0].cost_matrix

			city_index = priorityQueue[0].tour_so_far[-1]	

			if priorityQueue[0].lower_bound > bssf.cost:
				heapq.heappop(priorityQueue)
				num_pruned_states += 1
				continue

			for j in range(ncities):
				if reduced_cost_matrix[city_index][j] != np.inf:
					new_subproblem = self.Subproblem()
					total_states_created += 1

					#Create a copy of the parent problem's tour and cost matrix for each new subproblem
					new_subproblem.tour_so_far = copy.deepcopy(priorityQueue[0].tour_so_far)
					new_subproblem.cost_matrix = copy.deepcopy(priorityQueue[0].cost_matrix)
					new_subproblem.lower_bound = priorityQueue[0].lower_bound

					#When we choose a path, inf out the row and column and add the next city to the tour
					reduced_cost_matrix_copy, cost_of_path = self.infRowAndCol(new_subproblem.cost_matrix, ncities, city_index, j)
					new_subproblem.tour_so_far.append(j)
					new_subproblem.cities_remaining = ncities - len(new_subproblem.tour_so_far)

					#After inf out the row and column, reduced the matrix and add the additional reduction cost to the lower bound
					reduced_cost_matrix_copy, additional_cost = self.reduceMatrix(new_subproblem.cost_matrix, ncities)
					new_lower_bound = new_subproblem.lower_bound + cost_of_path + additional_cost

					new_subproblem.cost_matrix = reduced_cost_matrix_copy
					new_subproblem.lower_bound = new_lower_bound

					#If we have found a solution, see if it's better than the BSSF
					if len(new_subproblem.tour_so_far) == ncities:
						if new_subproblem.lower_bound < bssf.cost:
							route = []

							for i in range(ncities):
								route.append(cities[new_subproblem.tour_so_far[i]])
							bssf = TSPSolution(route)
						num_solutions += 1

					#Push the new subproblem into the queue if its lower bound is less than the BSSF cost
					if bssf.cost > new_subproblem.lower_bound:
						heapq.heappush(priorityQueue, new_subproblem)

					else:
						num_pruned_states += 1


			heapq.heappop(priorityQueue)

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = num_solutions
		results['soln'] = bssf
		results['max'] = max_queue_size
		results['total'] = total_states_created
		results['pruned'] = num_pruned_states
		return results

	#Time Complexity: O(n^2)
	#We have to check every row and column - since we have n rows and n columns, total time complexity is O(n^2)
	def reduceMatrix(self, original_cost_matrix, ncities):

		new_cost_matrix = original_cost_matrix
		cost_sum = 0

		#Subtract from rows
		for i in range(ncities):
			min_value = np.inf

			for j in range(ncities):
				if original_cost_matrix[i][j] < min_value:
					min_value = original_cost_matrix[i][j]

			if min_value != np.inf and min_value != 0:
				cost_sum += min_value

				for j in range(ncities):
					new_cost_matrix[i][j] = original_cost_matrix[i][j] - min_value

		#Subtract from columns
		for j in range(ncities):
			min_value = np.inf

			for i in range(ncities):
				if original_cost_matrix[i][j] < min_value:
					min_value = original_cost_matrix[i][j]
				
			if min_value != np.inf and min_value != 0:
				cost_sum += min_value

				for i in range(ncities):
					new_cost_matrix[i][j] = original_cost_matrix[i][j] - min_value

		return new_cost_matrix, cost_sum

	#Time Complexity: O(2n) => O(n)
	#Since all we are doing is checking every element in the row and column of the path we are taking
	def infRowAndCol(self, cost_matrix, ncities, current_city_index, parent_index):

		row = current_city_index
		col = parent_index

		cost_of_path = cost_matrix[current_city_index][parent_index]
		
		#Change row to be infinity
		for i in range(ncities):
			cost_matrix[i][col] = np.inf

		#Change column to be infinity
		for j in range (ncities):
			cost_matrix[row][j] = np.inf

		return cost_matrix, cost_of_path

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found during search, the
		best solution found.  You may use the other three field however you like.
		algorithm</returns>
	'''

	class Individual:
		def __init__(self):
			self.genome = []
			self.fitness = 0
		

		def __lt__(self, rhs):
			if (self.fitness < rhs.fitness):
				return True
			else:
				return False

	def create_genome(self, cities, ncities):
		genome = []

		current_city_index = 0
		foundTour = False

		while not foundTour and current_city_index != ncities:
			cities_not_visited = []

			for i in range(ncities):
				cities_not_visited.append(cities[i])

			route = []

			current_city = cities_not_visited.pop(current_city_index)
			first_city = current_city
			current_city_index += 1

			route.append(first_city)

			while len(cities_not_visited) > 0:
				potential_edges = []
				edge_indexes = []

				dest_city = None
				index = -1

				for i in range(len(cities_not_visited)):
					if current_city.costTo(cities_not_visited[i]) < np.inf:
						potential_edges.append(cities_not_visited[i])
						edge_indexes.append(i)

				if (len(potential_edges) == 0):
					break
				
				rand = random.randint(0, len(potential_edges) - 1)
				dest_city = potential_edges[rand]
				cities_not_visited.pop(edge_indexes[rand])

				if dest_city is not None:
					current_city = dest_city

					if len(cities_not_visited) == 0:
						if dest_city.costTo(first_city) < np.inf:
							route.append(current_city)
							foundTour = True
							break

						else: 
							break

					route.append(current_city)

				else:
					break

		route.append(first_city)

		if (len(route) != ncities + 1):
			genome = []
			genome.append(0)
			city_permutation = np.random.permutation(np.arange(1, ncities))
			
			for i in range(len(city_permutation)):
				genome.append(city_permutation[i])

			genome.append(0)

		else:
			for i in range(len(route)):
				genome.append(route[i]._index)

		return genome

	#Time Complexity: O(n)
	#Since we have to loop through all of the cities in order to find the total cost
	def get_fitness_level(self, cities, ncities, genome):
		cost = 0

		for i in range(1, ncities + 1):
			cost += cities[genome[i - 1]].costTo(cities[genome[i]])

		return cost

	#O(1) - dominating term are min() and max() functions
	#Since we know that only two challengers will get picked each time, function is in constant time
	def tournament_selection(self, population):
		#Probability of choosing the parent with the higher fitness value - 75%
		p = .5

		challengers = np.random.choice(population, size=2)

		if (random.random() < p):
			return min(challengers, key=lambda x: x.fitness)
		
		else:
			return max(challengers, key=lambda x: x.fitness)

	#O(n) - dominated by get_fitness_level function and looping through subgenomeB to find the missing cities
	def crossover(self, parent1, parent2, cities, ncities):
		#genes = []
		#map = {}

		#solution = np.inf

		#while solution == np.inf:
			#rand_index1 = random.randint(1, len(parent1.genome) - 2)
			#rand_index2 = random.randint(1, len(parent1.genome) - 2)

			#crossover_point1 = min(rand_index1, rand_index2)
			#crossover_point2 = max(rand_index1, rand_index2)

			#genes = copy.deepcopy(parent1.genome)

			#for i in range(len(genes)):
				#map[genes[i]] = i

			#for i in range(crossover_point1, crossover_point2 + 1):
				#value = parent2.genome[i]
				#t = genes[map[value]]
				#genes[map[value]] = genes[i]
				#genes[i] = t
				#t = map[genes[map[value]]]
				#map[genes[map[value]]] = map[genes[i]]
				#map[genes[i]] = t

			#solution = self.get_fitness_level(cities, ncities, genes)

		#child = self.Individual()
		#child.genome = genes
		#child.fitness = self.get_fitness_level(cities, ncities, child.genome)
		#return child

		
		#DO BETTER CROSSOVERS - DON'T WANT INF ROUTES
		new_genome = []
		subgenomeA = []
		subgenomeB = []

		geneAIndex = np.random.randint(1, len(parent1.genome) - 1)
		geneBIndex = np.random.randint(1, len(parent1.genome) - 1)

		start_index = min(geneAIndex, geneBIndex)
		end_index = max(geneAIndex, geneBIndex)

		subgenomeA.append(cities[0]._index)

		for i in range(start_index, end_index):
			subgenomeA.append(parent1.genome[i])

		subgenomeB = [item for item in parent1.genome if item not in subgenomeA]

		subgenomeB.append(cities[0]._index)

		new_genome = subgenomeA + subgenomeB

		child = self.Individual()
		child.genome = new_genome
		child.fitness = self.get_fitness_level(cities, ncities, new_genome)
		return child

	#O(1) - since all we end up doing is swapping the indexes of two cities in a genome
	def mutate(self, genome, mutation_rate, ncities):
		if (random.random() < mutation_rate):
			index1 = random.randint(0, len(genome) - 2)
			index2 = random.randint(index1, len(genome) - 1)

			chromosome_sequence = genome[index1:index2]
			chromosome_sequence.reverse()

			genome = genome[0:index1] + chromosome_sequence + genome[index2:]

			return genome
		else:
			return genome
	
	def converged(self, population):
		return all(genome.fitness == population[0].fitness for genome in population)

	#Time Complexity: 
	def fancy(self,time_allowance=60.0):
		results = {}
		start_time = time.time()
		cities = self._scenario.getCities()
		ncities = len(cities)

		population = []
		heapq.heapify(population)

		population_size = 100
		num_generations = 5
		mutation_rate = .2
		elitism_rate = .05

		elitism_offset = math.ceil(population_size * elitism_rate)

		generations_so_far = 0

		start_index = 0
		best_fitness_so_far = np.inf

		self.greedy(time_allowance=60)

		if (self.greedy_route != None):
			start_index = 1
			greedy_individual = self.Individual()

			for i in range(len(self.greedy_route)):		
				greedy_individual.genome.append(self.greedy_route[i]._index)

			greedy_individual.fitness = self.get_fitness_level(cities, ncities, greedy_individual.genome)
			heapq.heappush(population, greedy_individual)

		for i in range(start_index, population_size):
			new_individual = self.Individual()
			new_individual.genome = self.create_genome(cities, ncities)
			print(new_individual.genome)
			print(" | ")
			new_individual.fitness = self.get_fitness_level(cities, ncities, new_individual.genome)
			heapq.heappush(population, new_individual)
				
		while generations_so_far < num_generations:#time.time() - start_time < time_allowance and generations_so_far < num_generations:
			for generation in range(num_generations):
				if population[0].fitness < best_fitness_so_far:
					solution = population[0].genome
					best_route_so_far = solution
					best_fitness_so_far = self.get_fitness_level(cities, ncities, solution)

				new_population = []
				heapq.heapify(new_population)

				elites = heapq.nsmallest(elitism_offset, population)

				for i in range(0, elitism_offset):
					heapq.heappush(new_population, elites[i])

				for i in range(elitism_offset, len(population)):
					#if (time.time() - start_time >= time_allowance):
							#route = []

							#for i in range(len(solution) - 1):
							#	route.append(cities[solution[i]])
							#	bssf = TSPSolution(route)

							#end_time = time.time()
							#results['cost'] = bssf.cost
							#results['time'] = end_time - start_time
							#results['soln'] = bssf
							#results['count'] = 1
							#results['max'] = None
							#results['total'] = None
							#results['pruned'] = None
							#return results

					#CROSSOVER
					parent1 = self.tournament_selection(population)
					parent2 = self.tournament_selection(population)
					child = self.crossover(parent1, parent2, cities, ncities)
					heapq.heappush(new_population, child)

					#MUTATE
					population[i].genome = self.mutate(population[i].genome, mutation_rate, ncities)
					heapq.heappush(new_population, population[i])
		
				population = heapq.nsmallest(population_size, new_population)
				generations_so_far += 1

				if self.converged(population):
					generations_so_far = num_generations
					break

		route = []
		
		for i in range(len(solution) - 1):
			route.append(cities[solution[i]])

		bssf = TSPSolution(route)
		end_time = time.time()
		results['cost'] = best_fitness_so_far
		results['time'] = end_time - start_time
		results['soln'] = bssf
		results['count'] = 1
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


