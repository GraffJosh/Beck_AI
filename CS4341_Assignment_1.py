#CS4341 Assignment 1
from __future__ import with_statement # Required in 2.5
import signal
from contextlib import contextmanager
from sys import argv
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
				print (child.value)
				found = self.dl_search(child, depth-1)
				if found:
					return True
		return False

	def id_search(self):
		depth = 0
		print ('GOT HERE')
		while self.current_node.value != self.goal:
			print (depth)
			if self.dl_search(self.start_node, depth):
				return self.current_node
			else:
				depth += 1

	def gbf_search(self):
		while(self.current_node.value != self.goal):
			self.current_node.createChildren(self.operations_list)
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
			print (str(self.parent.value) + ' ' + self.operation.operator + ' ' + str(self.operation.integer) + ' = ' + str(self.value))
			solution_path.append(self.parent)
			self.parent.backtrackNode()


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

for filename in argv[1:]:
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
		starting_value = input('Starting value?: ')
		target_value = input('Target_Value?: ')
		time_limit = input('Time limit? (seconds): ')
		operations = input('Operations? (separate by spaces): ')
		operations_parsed = parse_operations(operations)


	# print (search_type +'\n'+starting_value+'\n'+target_value+'\n'+time_limit+'\n'+ operations_parsed[0]+'\n'+operations_parsed[1]+'\n'+operations_parsed[2])
	try:
		with time_limit_manager(int(time_limit)):
			start_time = time.time()
			id = SearchAlgorithm(int(starting_value), int(target_value), operations_parsed)
			if (search_type == 'iterative'):
				erik = id.id_search()
			elif (search_type == 'greedy'):
				erik = id.gbf_search()
			end_time = time.time()
			print ('DONE')
			erik.backtrackNode()
			execution_time = str(end_time - start_time)
			print ('Number of steps required: ' + str(len(solution_path)))
			print ('Search required: ' + execution_time + ' seconds')
			print ('Nodes expanded: ' + str(id.num_nodesexpanded))
			print ('Maximum depth: ' + str(len(solution_path)))
	except TimeoutException:
		print ('Timed out!')