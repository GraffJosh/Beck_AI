#CS4341 Assignment 2
from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager
from sys import argv
import matplotlib.pyplot as plt
import sys
import time
import itertools
import random
import numpy
import math

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
		self.start_node = Node(start,goal,operations_list)
		self.current_node = self.start_node
		self.start = start
		self.goal = goal
		self.operations_list = operations_list
		self.generation = 0
		self.best_node = self.current_node
		self.h_list_graph = []

		# Const Variables
		self.init_num_nodes = 4 			# number of nodes in a zoo
		self.num_generations = 10000
		self.max_num_operations = 30
		self.cull_percent = 0.5
		self.mutation_percent = 0

		# A zoo is an array of "nodes" where each node contains 
		# a list of operators.
		self.zoo = []

	#Reorders the open list according to our heuristic function
	def reorder_nodes(self):
		self.zoo.sort(key=lambda x: abs(self.goal - x.value))

	#Initialize the first generation
	def init_operations(self):

		self.generation += 1
		for node_num in range(self.init_num_nodes):

			op_array = [] # a list of nodes
			
			# appends to the zoo an initialized node given a maximum number of operations
			for op_num in range(random.randint(0,self.max_num_operations)):
				# appends to op_array a random operator from our pool
				random.shuffle(self.operations_list) # why have this? lol
				op_array.append(random.choice(self.operations_list))

			self.zoo.append(Node(self.start, self.goal, op_array))

	def genetic_search(self):
		#create initial population
		self.init_operations()
		#for each generation
		for num in range(self.num_generations):
			#for every node in the zoo
			for node in self.zoo:
				node.value = node.eval_node_val()		#eval the node
				node.heuristic = node.eval_node_fitness()	#eval the heuristic
				if node.value == self.goal:
					self.best_node = node
					return self.best_node
				#print (len(node.operations))
			
			# Reorder nodes in terms of best heuristic
			self.reorder_nodes()
			# Cull the weaker nodes
			self.cull()
			# Breed the fittest of the generation to create more children nodes
			self.zoo.extend(self.breed_population())

			#mutate randomly
			self.mutate()

			# Maybe a child node can have the best heuristic?
			self.best_node = self.zoo[0]
			self.h_list_graph.append(self.computeMeanHeuristic(self.zoo))
			#print (len(self.zoo))

		for organism in self.zoo:
			if organism.eval_node_fitness() == float("inf"):
				self.best_node = organism

			if organism.eval_node_fitness() > self.best_node.eval_node_fitness():
				self.best_node = organism

		return self.best_node

		#kill the weak
	def cull(self):
		for index in range(math.floor(len(self.zoo)*self.cull_percent), len(self.zoo)):
			del(self.zoo[len(self.zoo)-1])

	def mutate(self):
		for index in range(0, math.floor(len(self.zoo) * self.mutation_percent)):
			random.choice(self.zoo).irradiate(self.operations_list)

		#breed the population with itself
	def breed_population(self):
		self.generation += 1
		new_zoo = [] # this is the new population
		for organism in self.zoo:

			# these two operations should happen based on the fitness function
			parentA = self.choose_weighted_parent(self.zoo) 
			parentB = self.choose_weighted_parent(self.zoo)

			child = self.reproduce(parentA, parentB)
			new_zoo.append(child)
		#don't really need new zoo
		return new_zoo

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

	def choose_weighted_parent(self, zoo):
		fitnesses = []
		weights = []
		sum = 0

		for organism in zoo:
			sum = sum + organism.eval_node_fitness()
			fitnesses.append(organism.eval_node_fitness())

		for fitness in fitnesses:
			weights.append(fitness/sum)

		return numpy.random.choice(zoo, p = weights)

	def computeMeanHeuristic(self, generation_list):
		return	sum(node.heuristic for node in generation_list)/(len(generation_list))



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
		
		if abs(self.goal - num) == 0:
			return float("inf")

		return 1 / abs(self.goal - num)

	def irradiate(self, operations_list):
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
def printStats(search_type, error, steps, time, max_generation):
	print ('\n\n' + search_type)
	print ('error: '+ error)
	print ('Number of steps required: ' + steps)
	print ('Search time required: ' + time + ' seconds')
	print ('Maximum generation: ' + max_generation)


class TimeoutException(Exception): pass

# Handles our time out, raise the exception and does something about it
@contextmanager
def time_limit_manager(seconds):
	def signal_handler(signum, frame):
		raise TimeoutException
	signal.signal(signal.SIGALRM, signal_handler)
	signal.alarm(seconds)
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

	# else:
	# 	# Manual Input
	# 	search_type = input('Search type? (iterative, genetic): ')
	# 	starting_value = float(input('Starting value?: '))
	# 	target_value = float(input('Target_Value?: '))
	# 	time_limit = float(input('Time limit? (seconds): '))
	# 	operations = input('Operations? (separate by spaces): ')
	# 	operations_parsed = parseOperations(operations)

	
	try:
		with time_limit_manager(int(time_limit)):

			start_time = time.time()
			id = SearchAlgorithm(float(starting_value), float(target_value), operations_parsed)

			# ERIK IS OUR SOLUTION NODE
			if (search_type == 'genetic'):
				erik = id.genetic_search()
			end_time = time.time()
			print ('\nCLOSEST SOLUTION PATH')
			execution_time = str(end_time - start_time)
			erik.printSolution()
			printStats(search_type, str(abs(id.best_node.eval_node_val() - target_value)),str(len(id.best_node.operations)),
				execution_time, str(id.generation))

			# plt.show()
			error_sum = error_sum+abs(id.best_node.eval_node_val() - target_value)
			generation_sum = generation_sum+id.generation
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
		printStats(search_type, str(abs(id.best_node.eval_node_val() - target_value)),str(len(id.best_node.operations)),
				execution_time, str(id.generation))
		# plt.plot(id.h_list_graph)
		# plt.ylabel('Heuristic')
		# plt.show()
print("average error: "+str(error_sum/(len(argv)-1)))
print("average generations: "+str(generation_sum/(len(argv)-1)))