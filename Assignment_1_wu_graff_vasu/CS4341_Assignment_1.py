#CS4341 Assignment 1
from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager
from sys import argv
import sys
import time
import itertools

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
		self.start_node = Node(None, None, int(start))
		self.current_node = self.start_node
		self.goal = goal
		self.operations_list = operations_list
		self.num_nodesexpanded = 1
		self.OPEN = []
		self.CLOSED = []
		self.depth = 0
		self.max_depth = 0
		self.best_node = self.current_node
		self.solution_path = []

	#Normal depth limited; returns True is found
	def dl_search(self, node, depth):
		if (depth == 0) and (node.value == self.goal):
			return True
		elif (node.value == self.goal):
			return True
		elif (depth > 0):
			node.createChildren(self.operations_list)
			self.num_nodesexpanded += len(self.operations_list)
			for child in node.children:
				self.current_node = child
				if (abs(self.current_node.value - self.goal) < abs(self.best_node.value - self.goal)):
					self.best_node = self.current_node
				if (child.depth > self.max_depth):
					self.max_depth = child.depth
				#print (child.value)
				found = self.dl_search(child, depth-1)
				if found:
					return True
		return False
	#Iterative Deepening using depth first increasing depth each time
	def id_search(self):
		depth = 0
		#print ('GOT HERE')
		while self.current_node.value != self.goal:
			#print (depth)
			if self.dl_search(self.start_node, depth):
				return self.current_node
			else:
				depth += 1

	#Greedy Best First Search using OPEN and CLOSED node lists
	def gbf_search(self):
		while(self.current_node.value != self.goal):
			self.current_node.createChildren(self.operations_list)
			self.num_nodesexpanded += len(self.operations_list)
			if (abs(self.current_node.value - self.goal) < abs(self.best_node.value - self.goal)):
				self.best_node = self.current_node
			self.CLOSED.append(self.current_node)
			for child in self.current_node.children:
				if (child.depth > self.max_depth):
					self.max_depth = child.depth
				if(child not in self.OPEN) and (child not in self.CLOSED):
					self.OPEN.append(child)
					self.reorderOpen()
			self.current_node = self.OPEN.pop(0)
		self.best_node = self.current_node
		return self.best_node

	#Reorders the open list according to our heuristic function
	def reorderOpen(self):
		self.OPEN.sort(key=lambda x: abs(self.goal - x.value))

#Operation class holding an operator and integer from file input
class Operation:
	def __init__(self, operator, integer):
		self.operator = operator
		self.integer = integer

#Basic Node class that contains info about its value, the operation it took to get there, and its parents/children
class Node:
	def __init__(self, parent, operation, value):
		self.parent = parent
		self.operation = operation
		self.value = value
		self.children = []
		#keep track of depth
		if(parent is None):
			self.depth = 0
		else:
			self.depth = parent.depth + 1

	#Create the children nodes
	def createChildren(self, operations_list):
		if not self.children:
			for operation in operations_list:
				self.children.append(Node(self,operation, self.evalChildValue(operation)))

	#Evaluates the value of the child
	def evalChildValue(self, operation):
		return math_func[operation.operator](self.value, operation.integer)

	#Traverses the ancestry of a particular node and returns a list of nodes (ancestry)
	def backtrackNode(self, solution_path):
		if (self.parent is None):
			return solution_path.reverse()
		if not (self.parent is None):
			#print (str(self.parent.value) + ' ' + self.operation.operator + ' ' + str(self.operation.integer) + ' = ' + str(self.value))
			solution_path.append(self.parent)
			self.parent.backtrackNode(solution_path)

	#Prints out the operations that it took to reach the solution
	def printSolution(self, solution_path):
		if (self.parent is None):
			return
		if not (self.parent is None):
			print (str(self.parent.value) + ' ' + self.operation.operator + ' ' + str(self.operation.integer) + ' = ' + str(self.value))
			solution_path.append(self.parent)
			self.parent.printSolution(solution_path)

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
iterative_results = [[0 for x in range(w)] for y in range(h)] 
greedy_results = [[0 for x in range(w)] for y in range(h)] 
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
	# 	search_type = input('Search type? (iterative, greedy): ')
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
			if (search_type == 'iterative'):
				erik = id.id_search()
			elif (search_type == 'greedy'):
				erik = id.gbf_search()
			end_time = time.time()
			print ('\nFOUND SOLUTION')

			erik.printSolution(id.solution_path)
			execution_time = str(end_time - start_time)
			printStats(search_type, str(abs(id.best_node.value - target_value)),str(len(id.solution_path)),
				execution_time, str(id.num_nodesexpanded), str(id.max_depth))

			if (search_type == 'iterative'):
				iterative_results[0].append(float(execution_time)) #store execution time
				iterative_results[1].append(id.num_nodesexpanded)#store num expanded
				iterative_results[2].append(len(id.solution_path))#store maximum depth
			elif (search_type == 'greedy'):
				greedy_results[0].append(float(execution_time)) #store execution time
				greedy_results[1].append(id.num_nodesexpanded)#store num expanded
				greedy_results[2].append(id.max_depth)#store maximum depth
	except (TimeoutException, RuntimeError) as error:
		end_time = time.time()
		print ('Could not find solution\nTimed Out: ' + str(error.args))
		if (search_type == 'iterative'):
				iterative_results[3][0] = iterative_results[3][0]+1
				id.best_node.printSolution(id.solution_path)
		elif (search_type == 'greedy'):
				greedy_results[3][0] = greedy_results[3][0]+1
				id.best_node.printSolution(id.solution_path)
		execution_time = str(end_time - start_time)
		printStats(search_type, str(abs(id.best_node.value - target_value)),str(len(id.solution_path)),
				execution_time, str(id.num_nodesexpanded), str(id.max_depth))