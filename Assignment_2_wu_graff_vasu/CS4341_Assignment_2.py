#CS4341 Assignment 2
from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager
from sys import argv
import sys
import time
import itertools
import random
import numpy
import math

#Instead of using numpy arithmetic functions, just made some functions to use

def summation(a,b):
	return a+b

def difference(a,b):
	return a-b

def multiply(a,b):
	return a*b

def quotient(a,b):
	return a/b

def power(a,b):
	return a**b

#Here is our function switch case

math_func = {'+' : summation,
		   '-' : difference,
		   '*' : multiply,
		   '/' : quotient,
		   '^' : power,
}

#Algorithm Search Class for each type
class SearchAlgorithm:
	def __init__(self, start, goal, operations_list):
		#self.start_node = Node(None, None, int(start))
		#self.current_node = self.start_node
		self.start = start
		self.goal = goal
		self.operations_list = operations_list
		self.num_nodesexpanded = 1
		self.depth = 0
		#self.best_node = self.current_node

		# A zoo is an array of "nodes" where each node contains 
		# a list of operators.
		self.zoo = []

	#Reorders the open list according to our heuristic function
	def reorderOpen(self):
		self.OPEN.sort(key=lambda x: abs(self.goal - x.value))

	def init_operations(self):
		num_nodes = 50 			# number of nodes in a zoo
		num_operations = 30		# number of operators per node
		

		for node_num in range(num_nodes):

			op_array = [] # a list of nodes
			
			# appends to the zoo an initialized node.
			for op_num in range(num_operations):
				# appends to op_array a random operator from our pool
				op_array.append(random.choice(self.operations_list))

			self.zoo.append(Node(self.start, self.goal, op_array))

	def genetic_search(self):
		self.init_operations()
		num_reproductions = 10

		for num in range(num_reproductions):
			self.zoo = self.breed_population(self.zoo)

		best_organism = self.zoo[0]

		for organism in self.zoo:
			if organism.eval_node_fitness() == float("inf"):
				best_organism = organism

			if organism.eval_node_fitness() > best_organism.eval_node_fitness():
				best_organism = organism

		#best_organism.printSolution()
		#print(best_organism.eval_node_fitness())

		return best_organism

	def breed_population(self, zoo):

		new_zoo = [] # this is the new population
		for organism in self.zoo:

			# these two operations should happen based on the fitness function
			parentA = self.choose_weighted_parent(self.zoo) 
			parentB = self.choose_weighted_parent(self.zoo)

			child = self.reproduce(parentA, parentB)
			new_zoo.append(child)

		return new_zoo

	def reproduce(self, orgA, orgB):
		length = len(orgA.operations)
		cut_off = random.randint(0, length)

		child_operations = orgA.operations[:cut_off] + orgB.operations[cut_off:]
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

#Operation class holding an operator and integer from file input
class Operation:
	def __init__(self, operator, integer):
		self.operator = operator
		self.integer = integer

# For the genetic algorithm, a node contains information about
# the operators that are being evaluated.  
class Node:
		
	def eval_node_val(self):
		return 1

	def eval_node_fitness(self):
		num = self.start

		for operation in self.operations:
			num = math_func[operation.operator](num, operation.integer)
		
		if abs(self.goal - num) == 0:
			return float("inf")

		return 1 / abs(self.goal - num)

	def __init__(self, start_value, target_value, operations):
		#self.value = self.eval_node_val()
		self.operations = operations
		self.start = start_value
		self.goal = target_value
		#self.heuristic = self.eval_node_h()

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
def printStats(search_type, error, steps, time, nodes_expanded, max_depth):
	print ('\n\n' + search_type)
	print ('error: '+ error)
	print ('Number of steps required: ' + steps)
	print ('Search time required: ' + time + ' seconds')
	print ('Nodes expanded: ' + nodes_expanded)
	print ('Maximum depth: ' + max_depth)


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
				operations_parsed.append(Operation(args[n][:1],int(args[n][1:])))
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
			id = SearchAlgorithm(int(starting_value), int(target_value), operations_parsed)

			# ERIK IS OUR SOLUTION NODE
			if (search_type == 'genetic'):
				erik = id.genetic_search()
			end_time = time.time()
			print ('\nCLOSEST SOLUTION PATH')

			erik.printSolution()
			execution_time = str(end_time - start_time)
			#printStats(search_type, str(abs(id.best_node.value - target_value)),str(len(id.solution_path)),
			#execution_time, str(id.num_nodesexpanded), str(id.max_depth))

			if (search_type == 'genetic'):
				genetic_results[0].append(float(execution_time)) #store execution time
				#genetic_results[1].append(id.num_nodesexpanded)#store num expanded
				#genetic_results[2].append(id.max_depth)#store maximum depth

	except (TimeoutException, RuntimeError) as error:
		end_time = time.time()
		print ('Could not find solution\nTimed Out: ' + str(error.args))
		if (search_type == 'genetic'):
				genetic_results[3][0] = genetic_results[3][0]+1
				id.best_node.printSolution(id.solution_path)
		execution_time = str(end_time - start_time)
		printStats(search_type, str(abs(id.best_node.value - target_value)),str(len(id.solution_path)),
				execution_time, str(id.num_nodesexpanded), str(id.max_depth))