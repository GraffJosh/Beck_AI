#CS4341 Assignment 2
from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager
from sys import argv
import matplotlib.pyplot as plt
from matplotlib import gridspec
import sys
import time
import itertools
import random
import numpy
import math
import os

#Here is our function switch case
math_func = {'+' : numpy.add,
		   '-' : numpy.subtract,
		   '*' : numpy.multiply,
		   '/' : numpy.divide,
		   '^' : numpy.power,
}

#Algorithm Search Class for each type
class SearchAlgorithm:
	def __init__(self, start, goal, operations_list):
		self.start = start
		self.goal = goal
		self.operations_list = operations_list
		self.generation = 0
		self.best_node = Node(self.start, self.goal, self.operations_list)
		self.h_list_graph = []
		self.f_list_graph = []

		# Const Variables

		self.max_population_size = 100		# number of nodes in a zoo
		self.max_num_generations = 1000      # how many generations to try
		self.max_num_operations = 30		# max number of operators
		self.percent_cull = 0.5				# percentage to kill off / generation
		self.percent_mutation = 0.8			# percentage to mutate randomly / generation

		# A zoo is an array of "nodes" where each node contains 
		# a list of operators.
		self.zoo = []

	#Reorders the open list according to our heuristic function
	def reorder_nodes(self):
		self.zoo.sort(key=lambda x: abs(self.goal - x.value), reverse=True)

	#Initialize the first generation
	def init_population(self):
		self.generation += 1
		for node_num in range(self.max_population_size):
			op_array = [] # a list of nodes
			# appends to the zoo an initialized node given a maximum number of operations
			for op_num in range(random.randint(1,self.max_num_operations)):
				# appends to op_array a random operator from our pool
				random.shuffle(self.operations_list) # why have this? lol
				op_array.append(random.choice(self.operations_list))
			# add the node into the zoo (population)
			new_node = Node(self.start, self.goal, op_array)
			self.zoo.append(new_node)
			# establish a best node at the start, doesn't matter which is it in the first population
			# as long as it's valid

		# evaluate the fitness and values
		self.evaluateOrganisms()

	def genetic_search(self):
		#create initial population
		self.init_population()
		#for each generation
		for generation in range(self.max_num_generations):
			# Reorder nodes in terms of best heuristic
			self.reorder_nodes()
			# Cull the weaker nodes
			self.cull()
			# birds & bees baby
			self.breed_population()
			#mutate randomly
			self.mutate()
			# evaluate the fitness and values
			self.evaluateOrganisms()
			# compute mean fitness
			self.h_list_graph.append(self.compute_mean_fitness())
		return self.best_node

	def evaluateOrganisms(self):
		#for every node in the zoo
		for node in self.zoo:
			node.value = node.eval_node_val()				#eval the node value
			node.heuristic = node.eval_node_fitness()		#eval the node heuristic
			if (node.value == node.goal):
				self.best_node = node
			if node.heuristic > self.best_node.heuristic:
				self.best_node = node
			if numpy.isnan(node.value) or numpy.isnan(node.heuristic): # make sure that the value or heuristic is valid
				self.zoo.remove(node)

		#kill the weak
	def cull(self):
		# find the point where to cut off the rest of the organisms
		cutoff_index = math.floor(len(self.zoo) * (1 - self.percent_cull))
		self.zoo = self.zoo[cutoff_index:]

	def mutate(self):
		# randomly mutate a percentage of the population
		for index in range(0, math.floor(len(self.zoo) * self.percent_mutation)):
			random.choice(self.zoo).irradiate(self.operations_list, self.max_num_operations)

		#breed the population with itself
	def breed_population(self):
		self.generation += 1
		fitnesses = []
		#make a copy of the current zoo to use as parents of the next generatin
		parent_list = list(self.zoo)
		for organism in self.zoo:
			fitness = organism.eval_node_fitness()
			if numpy.isnan(fitness) or numpy.isinf(fitness):
				# means that a solution has been found
				fitnesses.append(0)
			else:
				fitnesses.append(fitness)

		fitnesses = numpy.array(fitnesses)
		normalizer = fitnesses.sum()
		# normalize to make sum of proabilities = 1 ; get probability distribution
		prob_dist = fitnesses / normalizer
				
		# create children until zoo is full
		while (len(self.zoo) < self.max_population_size):
			# these two operations should happen based on the fitness function
			parentA = self.choose_weighted_parent(parent_list, prob_dist) 
			parentB = self.choose_weighted_parent(parent_list, prob_dist)
			child = self.reproduce(parentA, parentB)
			# append to zoo
			self.zoo.append(child)

	def reproduce(self, orgA, orgB):
		# pick the organism with the shorter length
		lengthA = len(orgA.operations)
		lengthB = len(orgB.operations)
		#initialize length to organism with least number of operators
		length = min(lengthA,lengthB)
		#evaluate cutoff point to crossover
		max_cut_off = random.randint(0, math.floor(self.max_num_operations/2))
		#finds the minimum of both to ensure no index of range
		cut_off = min(length, max_cut_off)
		#create child operations list
		child_operations = orgA.operations[:cut_off] + orgB.operations[cut_off:]
		#create child
		child = Node(orgA.start, orgB.goal, child_operations)
		return child

	def choose_weighted_parent(self, parents, probabilities):
		# return a random parent based on weighting
		return numpy.random.choice(parents, p = probabilities)

	def compute_mean_fitness(self):
		return numpy.sum(node.heuristic for node in self.zoo)/len(self.zoo)

	def collectFitnesses(self, generation_list):
		fitness_list = []
		for node in generation_list:
			data_point = []
			data_point.append(node.heuristic)
			data_point.append(self.generation)
			fitness_list.append(data_point)
		return fitness_list



#Operation class holding an operator and integer from file input
class Operation:
	def __init__(self, operator, integer):
		self.operator = operator
		self.integer = integer

# For the genetic algorithm, a node contains information about
# the operators that are being evaluated.  
class Node:
	def __init__(self, start_value, target_value, operations):
		self.value = 0
		self.operations = operations
		self.start = start_value
		self.goal = target_value
		self.heuristic = 0
		
	def eval_node_val(self):
		num = self.start
		for operation in self.operations:
			num = math_func[operation.operator](num, operation.integer)
		return num

	def eval_node_fitness(self):
		num = self.start
		for operation in self.operations:
			num = math_func[operation.operator](num, operation.integer)
		return 1/abs(self.goal - num)

	#inject nuclear material into portion of the population
	def irradiate(self, operations_list, max_operations):
		radiation = random.randint(0,2)
		if(len(self.operations)>0):
			#print(radiation)
			if(radiation == 0):
				# SUBSTITUTE
				self.operations[random.randint(0, len(self.operations)) - 1] = random.choice(operations_list)
				#print ("Sub")
			elif(radiation == 1):
				# REMOVE
				if(len(self.operations) > 1):
					self.operations.pop(random.randint(0, len(self.operations)) - 1)
				#print ("Rem")
			if(radiation == 2):
				# ADD (need to add a maximum operations)
				self.operations.append(random.choice(operations_list))
				#print ("Add")
		else:
			self.operations.append(random.choice(operations_list))

	def printSolution(self):
		num = self.start

		for operation in self.operations:
			next_num = math_func[operation.operator](num, operation.integer)
			print (str(num) + ' ' + operation.operator + ' ' + str(operation.integer) + ' = ' + str(next_num))
			num = next_num

#Breaks up a string into parameters to create an Operations object assuming first value is the operation
#and the rest is the integer
def parseOperations(strlist):
	string_arr = strlist
	operations_list = []
	for string in string_arr:
		letter_arr = list(string)
		operations_list.append(Operation(letter_arr[0],int(''.join(letter_arr[1:]))))
	return operations_list

#Prints the stats
def printStats(search_type, error, steps, time, population_size, max_generation):
	print ('\n\n' + search_type)
	print ('Error: '+ error)
	print ('Size of organism: ' + steps)
	print ('Search time required: ' + time + ' seconds')
	print ('Population size: ' + population_size)
	print ('Number of generations: ' + max_generation)

#creates a matplot of list of data
def generateFitnessGraph(h_list, indvar, xlabel, ylabel,population_size, num_generations, cull_percent, mutation_percent, max_num_operations):
	txt = ('Population size: ' + str(population_size) + ' | ' +
		'Generations: ' + str(num_generations) + ' | ' +
		'Max operations: ' + str(max_num_operations) + '\n' +
		'Mutation percentage: ' + str(mutation_percent*100) + '%' + ' | ' +
		'Cull percentage: ' + str(cull_percent*100) + '%')
	fig = plt.figure()
	gs = gridspec.GridSpec(2, 1, height_ratios=[7, 1]) 
	m = fig.add_subplot(gs[0])
	m.set_title('Genetic Algorithm: '+ ylabel + ' vs. ' + xlabel)
	m.grid(True)
	for data_line in h_list:
		m.plot(data_line)
	m.set_ylabel(xlabel)
	m.set_xlabel(ylabel+'\n\n' + txt)
	sv_txt = ('P' + str(population_size) +
		'G' + str(num_generations) +
		'O' + str(max_num_operations) +
		'M' + str(int(mutation_percent*100)) +
		'C' + str(int(cull_percent*100)))
	cwd = os.getcwd()
	graph_folder = '/saved_graphs/'
	fig.savefig(cwd + graph_folder + sv_txt + '.png')
	plt.show()


class TimeoutException(Exception): pass

# Handles our time out, raise the exception and does something about it
@contextmanager
def time_limit_manager(seconds, search):
	def signal_handler(signum, frame):
		search.f_list_graph.append(abs(search.best_node.value - search.best_node.goal))
		raise TimeoutException
	def signal_best_node(signum,frame):
		if len(seconds) == 1:
			signal.signal(signal.SIGALRM, signal_handler)
		tmint = seconds.pop()
		search.f_list_graph.append(abs(search.best_node.value - search.best_node.goal))
		signal.alarm(tmint)
		print (abs(search.best_node.value - search.best_node.goal))
	tmint0 = seconds.pop()
	if (len(seconds) == 0):
		signal.signal(signal.SIGALRM, signal_handler)
	else:
		signal.signal(signal.SIGALRM, signal_best_node)
	signal.alarm(tmint0)
	try:
		yield
	finally:
		signal.alarm(0)

_iterArg =iter(argv)
if len(argv) > 1:
	_iterArg =iter(argv)
	next(_iterArg)
w, h = 1,4
genetic_results = [[0 for x in range(w)] for y in range(h)] 
#the zero these place in the first index must be accounted for in the avg

#Limit the recursion limit
sys.setrecursionlimit(10000)
error_sum = 0.0
generation_sum = 0.0
graph_data_lines = []
graph_error_lines = []
for filename in _iterArg:
	if len(argv) > 1 :
		args = []
		with open(filename) as f:
			for line in f:
				args.append(line.strip())
		if len(args) > 4:
			search_type = args[0]
			starting_value = float(args[1])
			target_value = float(args[2])
			time_limit = float(args[3])
			n = 4
			operations_parsed = []
			while n < len(args):
				operations_parsed.append(Operation(args[n][:1],float(args[n][1:])))
				n = n+1
		else:
			print ("not enough arguments in file")
			exit(0)
	
	try:
		# Get time data
		# time_intervals0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 60]
		# time_intervals1 = numpy.diff(time_intervals0)
		# time_intervals2 = list(time_intervals1[::-1])
		tmlimit = []
		tmlimit.append(int(time_limit))
		id = SearchAlgorithm(float(starting_value), float(target_value), operations_parsed)
		with time_limit_manager(tmlimit, id):
			start_time = time.time()
			# ERIK IS OUR SOLUTION NODE
			if (search_type == 'genetic'):
				erik = id.genetic_search()
			end_time = time.time()
			print ('\nCLOSEST SOLUTION PATH')
			execution_time = str(end_time - start_time)
			erik.printSolution()
			printStats(search_type, str(abs(id.best_node.eval_node_val() - target_value)),str(len(id.best_node.operations)),
				execution_time, str(id.max_population_size), str(id.generation))

			error_sum = error_sum+abs(id.best_node.eval_node_val() - target_value)
			generation_sum = generation_sum+id.generation

			#add fitness data to graph
			graph_data_lines.append(id.h_list_graph)

			if (search_type == 'genetic'):
				genetic_results[0].append(float(execution_time)) #store execution time
				#genetic_results[1].append(id.num_nodesexpanded)#store num expanded
				#genetic_results[2].append(id.max_generation)#store maximum generation

	except (TimeoutException, RuntimeError) as error:
		end_time = time.time()
		print ('Could not find solution\nTimed Out: ' + str(error.args))
		if (search_type == 'genetic'):
				genetic_results[3][0] = genetic_results[3][0]+1
				id.best_node.printSolution()
		execution_time = str(end_time - start_time)
		error_sum = error_sum+abs(id.best_node.eval_node_val() - target_value)
		generation_sum = generation_sum+id.generation
		
		#add fitness data to graph
		graph_data_lines.append(id.h_list_graph)
		
		printStats(search_type, str(abs(id.best_node.eval_node_val() - target_value)),str(len(id.best_node.operations)),
				execution_time,str(id.max_population_size), str(id.generation))

print("Average Error: "+str(error_sum/(len(argv)-1)))
print("Average Generations: "+str(generation_sum/(len(argv)-1)))

# create graph for each run and savefig
# generateFitnessGraph(graph_data_lines, range(len(graph_data_lines)), 'Generation', 'Fitness', id.max_population_size, id.max_num_generations, 
# 	id.percent_cull, id.percent_mutation, id.max_num_operations)
