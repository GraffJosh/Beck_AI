#CS4341 Assignment 1
from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager
from sys import argv
import sys
import time
import itertools

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

math_func = {'+' : summation,
		   '-' : difference,
		   '*' : multiply,
		   '/' : quotient,
		   '^' : power,
}

solution_path = []
bestNode = None

class SearchAlgorithm:
	def __init__(self, start, goal, operations_list):
		self.start_node = Node(None, None, int(start))
		self.current_node = self.start_node
		self.goal = goal
		self.operations_list = operations_list
		self.num_nodesexpanded = 0
		self.OPEN = []
		self.CLOSED = []
		self.depth = 0
		self.max_depth = 0
		self.best_node = self.current_node

	def dl_search(self, node, depth):
		if (depth == 0) and (node.value == self.goal):
			return True
		elif (node.value == self.goal):
			return True
		elif (depth > 0):
			node.createChildren(self.operations_list)
			for child in node.children:
				self.num_nodesexpanded += 1
				self.current_node = child
				if (abs(self.current_node.value - self.goal) < abs(self.best_node.value - self.goal)):
					self.best_node = self.current_node
				#print (child.value)
				found = self.dl_search(child, depth-1)
				if found:
					return True
		return False

	def id_search(self):
		depth = 0
		#print ('GOT HERE')
		global bestNode
		bestNode = self.best_node
		while self.current_node.value != self.goal:
			#print (depth)
			if self.dl_search(self.start_node, depth):
				self.best_node = self.current_node
				return self.current_node
			else:
				depth += 1
				if depth > self.max_depth:
					self.max_depth = depth

	def gbf_search(self):
		while(self.current_node.value != self.goal):
			self.current_node.createChildren(self.operations_list)
			if (abs(self.current_node.value - self.goal) < abs(self.best_node.value - self.goal)):
				self.best_node = self.current_node
			self.num_nodesexpanded += 1
			self.CLOSED.append(self.current_node)
			for child in self.current_node.children:
				if(child not in self.OPEN) and (child not in self.CLOSED):
					self.OPEN.append(child)
					self.reorderOpen()
			self.current_node = self.OPEN.pop(0)
		return self.current_node

	def reorderOpen(self):
		new_list = []
		for node in self.OPEN:
			if not new_list:
				new_list.append(node)
			else:	
				for item in new_list:
					h1 = abs(self.goal - node.value)
					h2 = abs(self.goal - item.value)
					if(h1 < h2):
						new_list.append(node)
						break
		self.OPEN = new_list

	def findMaxDepth(self):
		global solution_path
		for node in self.CLOSED:
				solution_path = []
				node.backtrackNode()
				if solution_path:
					current_depth = len(solution_path)+1 #total number of parents for selected node
					if(self.max_depth < current_depth):
						self.max_depth = current_depth
		print (self.max_depth)
		return self.max_depth

class Operation:
	def __init__(self, operator, integer):
		self.operator = operator
		self.integer = integer

class Node:
	def __init__(self, parent, operation, value):
		self.parent = parent
		self.operation = operation
		self.value = value
		self.children = []

	def createChildren(self, operations_list):
		if not self.children:
			for operation in operations_list:
				self.children.append(Node(self,operation, self.evalChildValue(operation)))

	def evalChildValue(self, operation):
		return math_func[operation.operator](self.value, operation.integer)

	def backtrackNode(self):
		global solution_path
		if (self.parent is None):
			return solution_path.reverse()
		if not (self.parent is None):
			#print (str(self.parent.value) + ' ' + self.operation.operator + ' ' + str(self.operation.integer) + ' = ' + str(self.value))
			solution_path.append(self.parent)
			self.parent.backtrackNode()

	def backtrackNode2(self, solution_nodes):
		if (self.parent is None):
			return solution_path.reverse()
		if not (self.parent is None):
			print (str(self.parent.value) + ' ' + self.operation.operator + ' ' + str(self.operation.integer) + ' = ' + str(self.value))
			solution_path.append(self.parent)
			self.parent.backtrackNode2(solution_path)



def parse_operations(strlist):
	string_arr = strlist.split()
	operations_list = []
	for string in string_arr:
		letter_arr = list(string)
		operations_list.append(Operation(letter_arr[0],int(''.join(letter_arr[1:]))))
	return operations_list


class TimeoutException(Exception): pass

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

sys.setrecursionlimit(10000)
for filename in _iterArg:
	global bestNode
	if len(argv) > 1 :
		args = []
		with open(filename) as f:
			for line in f:
				args.append(line.strip())
		if len(args) == 5:
			search_type = args[0]
			starting_value = float(args[1])
			target_value = float(args[2])
			time_limit = float(args[3])
			operations = args[4]
			operations_parsed = parse_operations(operations)
		else:
			print ("not enough arguments in file")
			exit(0)

	else:
		search_type = input('Search type? (iterative, greedy): ')
		starting_value = float(input('Starting value?: '))
		target_value = float(input('Target_Value?: '))
		time_limit = float(input('Time limit? (seconds): '))
		operations = input('Operations? (separate by spaces): ')
		operations_parsed = parse_operations(operations)


	# print (search_type +'\n'+starting_value+'\n'+target_value+'\n'+time_limit+'\n'+ operations_parsed[0]+'\n'+operations_parsed[1]+'\n'+operations_parsed[2])
	try:
		with time_limit_manager(int(time_limit)):
			start_time = time.time()
			id = SearchAlgorithm(int(starting_value), int(target_value), operations_parsed)
			id.num_nodesexpanded = 0
			id.depth = 0
			solution_path = []


			if (search_type == 'iterative'):
				erik = id.id_search()
			elif (search_type == 'greedy'):
				erik = id.gbf_search()
			end_time = time.time()
			print ('DONE')
			erik.backtrackNode2(solution_path)
			execution_time = str(end_time - start_time)
			curr_max_depth =id.findMaxDepth()
			print ('Number of steps required: ' + str(len(solution_path)))
			print ('Search required: ' + execution_time + ' seconds')
			print ('Nodes expanded: ' + str(id.num_nodesexpanded))
			print ('Maximum depth: ' + str(curr_max_depth))

			if (search_type == 'iterative'):
				iterative_results[0].append(float(execution_time)) #store execution time
				iterative_results[1].append(id.num_nodesexpanded)#store num expanded
				iterative_results[2].append(len(solution_path))#store maximum depth
			elif (search_type == 'greedy'):
				greedy_results[0].append(float(execution_time)) #store execution time
				greedy_results[1].append(id.num_nodesexpanded)#store num expanded
				greedy_results[2].append(curr_max_depth)#store maximum depth
	except (TimeoutException, RuntimeError) as error:
		print ('Timed Out: ' + str(error.args))
		if (search_type == 'iterative'):
				iterative_results[3][0] = iterative_results[3][0]+1
				bestNode = id.best_node
				bestNode.backtrackNode2(solution_path)
		elif (search_type == 'greedy'):
				greedy_results[3][0] = greedy_results[3][0]+1
				bestNode = id.best_node
				bestNode.backtrackNode2(solution_path)
	


if len(argv) > 1:
	#Average and print cumulative results
	for result in iterative_results:
		sigma = float(0)
		for value in result:
			sigma = sigma + value
		if len(result) > 1:
			result[0] = sigma / (len(result)-1) #average the results and store in index 0 (-1 accounts for preceeding 0)
			#iterative_results[n][x] retains the initial result as well. There happens to be a blank start.
	for result in greedy_results:
		sigma = float(0)
		for value in result:
			sigma = sigma + value
		if len(result) > 1:
			result[0] = sigma / (len(result)-1) #average the results and store in index 0 (-1 accounts for preceeding 0)

	print ('\nAverage execution time '+ 'Iterative: '+str(iterative_results[0][0])+', Greedy: '+str(greedy_results[0][0]))
	print ('Average nodes expanded '+ 'Iterative: '+str(iterative_results[1][0])+', Greedy: '+str(greedy_results[1][0]))
	print ('Average solution length '+ 'Iterative: '+str(iterative_results[2][0])+', Greedy: '+str(greedy_results[2][0]))
	print ('Number timed out  '+ 'Iterative: '+str(iterative_results[3][0])+', Greedy: '+str(greedy_results[3][0]))