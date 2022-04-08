#!/usr/bin/python3

from asyncio.windows_utils import Popen
from functools import reduce
from multiprocessing.pool import INIT
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

	#O(n)
	def crossover(self, parent1, parent2, cities, ncities):
		genes = []
		map = {}

		rand_index1 = random.randint(1, len(parent1) - 2)
		rand_index2 = random.randint(1, len(parent1) - 2)

		crossover_point1 = min(rand_index1, rand_index2)
		crossover_point2 = max(rand_index1, rand_index2)

		genes = copy.deepcopy(parent1)

		for i in range(len(genes)):
			map[genes[i]] = i

		for i in range(crossover_point1, crossover_point2 + 1):
			value = parent2[i]
			t = genes[map[value]]
			genes[map[value]] = genes[i]
			genes[i] = t
			t = map[genes[map[value]]]
			map[genes[map[value]]] = map[genes[i]]
			map[genes[i]] = t

		return genes

	#O(n) - since we may potentially have to reverse all of the cities in a given tour
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

	#O(1) - we just compare two routes in the population each time
	def tournament_selection(self, population, distance_matrix):
		p = .5

		challenger1 = population[random.randint(0, len(population) - 1)]
		challenger2 = population[random.randint(0, len(population) - 1)]

		if (random.random() < p):
			if (self.cost(distance_matrix, challenger1) < self.cost(distance_matrix, challenger2)):
				return challenger1
			
			else:
				return challenger2
		
		else:
			if (self.cost(distance_matrix, challenger1) > self.cost(distance_matrix, challenger2)):
				return challenger1
			
			else:
				return challenger2
	
	def cost(self, distance_matrix, route):
		return distance_matrix[np.roll(route, 1), route].sum()

	#O(n^2) for each iteration
	def two_opt(self, distance_matrix, route):
		best = route
		improved = True

		while improved:
			improved = False

			for i in range(1, len(route) - 2):
				for j in range(i + 1, len(route)):
					if j - 1 == 1: continue

					new_route = route[:]
					new_route[i:j] = route[j - 1:i - 1:-1]

					if self.cost(distance_matrix, new_route) < self.cost(distance_matrix, best):
						best = new_route
						improved = True
						route = best

		return best

	#Total time complexity is O(population_size * num_generations * n^2) -> O(n^2) although depending on the parameters we could have large constant factors
	def fancy(self, time_allowance=60):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		greedy = self.greedy(time_allowance)['soln'].route
		start_time = time.time()

		population = []
		heapq.heapify(population)

		population_size = 10
		num_generations = 10
		generations_so_far = 0
	
		best_cost_so_far = np.inf
		best_solution_so_far = []

		distance_matrix = np.zeros((ncities, ncities))

		#Creating the initial reduced matrix
		for i in range(ncities):
			for j in range(ncities):
				distance_matrix[i][j] = cities[i].costTo(cities[j])

		#Putting the greedy route in the population first
		greedy_route = []

		for i in range(len(greedy)):
			greedy_route.append(greedy[i]._index)

		heapq.heappush(population, greedy_route)

		#if ncities > 50:
		#	final_route = []

		#	best = self.two_opt(distance_matrix, greedy_route)	

		#	for i in range(len(best)):
		#		final_route.append(cities[best[i]])

		#	bssf = TSPSolution(final_route)

		#	end_time = time.time()
		#	results['cost'] = bssf.cost
		#	results['time'] = end_time - start_time
		#	results['count'] = 1
		#	results['soln'] = bssf
		#	return results

		#Mutating the greedy route to get part of our initial population
		for i in range(1, population_size):
			new_route = self.mutate(greedy_route, 1, ncities)
			heapq.heappush(population, new_route)

		#Doing crossovers to get the other part of our initial population
		for i in range(population_size):
			parent1 = self.tournament_selection(population, distance_matrix)
			parent2 = self.tournament_selection(population, distance_matrix)
			child = self.crossover(parent1, parent2, cities, ncities)
			heapq.heappush(population, new_route)

		#While we haven't hit the max number of generations
		while generations_so_far != num_generations and time.time() - start_time < time_allowance:
			new_population = []
			heapq.heapify(new_population)

			#First run 2-opt on every route in the population to see if we can find an improvement
			for i in range(population_size):
				solution = []

				best = self.two_opt(distance_matrix, population[i])	

				if (self.cost(distance_matrix, best) < best_cost_so_far):
					best_solution_so_far = best
					best_cost_so_far = self.cost(distance_matrix, best)

				if time.time() - start_time < time_allowance:
					num_generations = generations_so_far
					break

			#Do mutations and add them to the new population
			for j in range(1, population_size):
				new_route = self.mutate(population[i], 1, ncities)
				heapq.heappush(new_population, new_route)

			#Do crossovers and add them to the new population
			for j in range(population_size):
				parent1 = self.tournament_selection(population, distance_matrix)
				parent2 = self.tournament_selection(population, distance_matrix)
				child = self.crossover(parent1, parent2, cities, ncities)
				heapq.heappush(new_population, new_route)

			#The population becomes the best results from the new population that we just generated
			population = heapq.nsmallest(population_size, new_population)
			generations_so_far += 1
		
		final_route = []

		for i in range(len(best_solution_so_far)):
			final_route.append(cities[best_solution_so_far[i]])

		bssf = TSPSolution(final_route)

		end_time = time.time()
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = 1
		results['soln'] = bssf
		return results
