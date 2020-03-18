import pygame
from pygame.locals import *
import random, math
from pprint import pprint
from functools import reduce
from collections import UserDict

class UndirectedGraph(UserDict):
	def connect(self,node1,node2):
		# only connect when not already connected
		if not node2 in self[node1].keys():
			self[node1][node2] = {}
			self[node2][node1] = {}

	def disconnect(self,node1,node2):
		if node2 not in self[node1]:
			raise ValueError('Cannot disconnect non existant edge')
		del self[node1][node2]	
		del self[node2][node1]	
	def set_attr(self, node1, node2, attr, value):
		# sets the attribute of an edge to attr
		if node2 not in self[node1]:
			raise ValueError('Cannot set attribute of non existant edge') 
		self[node1][node2][attr] = value
		self[node2][node1][attr] = value
	def __missing__(self, key):
		self.__setitem__(key,{})
		return self.__getitem__(key)

	def __repr__(self):
		return str(f'{type(self).__name__}({self.data})')

	def get(self, key, default):
		if key in self:
			return self.__getitem__(key)
		else:
			return default

	def common_connections(self, node1, node2):
		return [i for i in self[node1] if i in self[node2]]


class Node(object):
	# TODO remove graph attribute
	def __init__(self, x, y, graph):
		self.x = x
		self.y = y
		self.graph = graph
	
	def draw(self,surface):
		#pygame.draw.circle(self.graph, (50,100,150), (self.x,self.y), 5)
		# make the font
		pygame.font.init() 
		myfont = pygame.font.SysFont('Comic Sans MS', 20)
		text_content =str(self)
		#text_content = '%.2f'% (angle((self,surface.vertices[0])))
		text = myfont.render(text_content,False,(255, 0, 0))
		surface.blit(text,(self.x,self.y))

	def closest_node(self, nodes):
			nodes.remove(self)
			closer = lambda node:abs(sum(node.pos) - sum(self.pos))
			distances = map(closer, nodes)
			closest = nodes[min(enumerate(distances), key = lambda x:x[1])[0]]
	def connected_to(self,other):
		#true if both nodes are connected
		return other in self.graph.edges[self].keys()

	def __repr__(self):
		return str(f'{type(self).__name__}({self.x},{self.y})')
def signed_distance(p,l):
	# returns the signed distance between point p and line l
	return (p.x - l[0].x)*(l[1].y-l[0].y) - (p.y - l[0].y)*(l[1].x - l[0].x)

def opposite_sides(p1,p2,l):
	# determine if a line passes between two points
	sign = lambda a: (a>0) - (a<0) #returns -1 for a negative, 1 for a positive else 0

	return sign(signed_distance(p1, l)) != sign(signed_distance(p2, l))


def intersect(l1,l2):
	#takes two lines as tuples of Nodes and returns whether they intersect	

	
	#See if both points of one line are on the one side of the other line
	return opposite_sides(l1[0], l1[1], l2) and opposite_sides(l2[0], l2[1], l1)

def line_length(l):
	node1,node2 = l
	return math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)

def determinant(matrix):
	# recursively gets the determinant of a matrix
	# base case
	if len(matrix) == len(matrix[0]) == 2:
		return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
	
	def get_submatrix(matrix, top_index):
		# gets the submatrix that doesn't include the first row
		# or the column of top_index
		return [tuple(map(row.__getitem__,(index for index in range(len(row)) if index != top_index))) for row in matrix[1:]]
	return sum(((-1)**index)*value*determinant(get_submatrix(matrix, index)) for index, value in enumerate(matrix[0]))

class Polygon(object):
	"""a list of points that are made anticlockwise in the begining"""
	def __init__(self, nodes):
		if not nodes:
			raise ValueError('can\'t make polygon with not points')
		self.nodes = nodes

	def make_anticlockwise(self):
		'''sorts the nodes in anticlockwis order around their 'center of mass' 
		starting from the left'''

		# get the center
		centerx, centery = sum(map(lambda node:node.x,self.nodes))/len(self.nodes), sum(map(lambda node:node.y, self.nodes))
		# sort the nodes
		self.nodes.sort(key=lambda node: math.atan2(node.y - centery, node.x - centerx))
		print(self.nodes)


	def get_tangent(self, other, stitch= False):
		# computes the tangent to another polygon
		self.make_anticlockwise()
		pass


class Triangle(Polygon):
	"""three point polygon"""
	def __init__(self, edge):
		super().__init__(edge)

	def contains_point(self, point):
		pass

	def circumcircle_contains_point(self, point):
		pass

class Line(Polygon):
	"""two point polygon"""
	pass

		

class Graph(pygame.Surface):
	'''
	A graph
	Stored as an edge list
	Mutable
	'''

	def __init__(self, vertex_count, size):
		super().__init__(size)
		self.vertex_count = vertex_count
		self.vertices = []
		self.edges = UndirectedGraph({})
		self.populate()
	
	def populate(self):
		'''add the vertices to the graph and connect them'''

		for i in range(self.vertex_count):
			x= random.randint(self.get_rect().left, self.get_rect().right)			
			y= random.randint(self.get_rect().top, self.get_rect().bottom)
			new_node = Node(x, y, self)
			self.vertices.append(new_node)
		def triangulate_(nodes):

			if not nodes:
				return
			if len(nodes) <= 3:
				# base case 
				# found a triangle or line or poinr

				try:
					# found a triangle connect its points
					self.edges.connect(nodes[1],nodes[2])
					self.edges.connect(nodes[2],nodes[0])
					#go anticlockwise around the triangle
					convex_hull = [nodes[0],nodes[1],nodes[2]] if signed_distance(nodes[0],(nodes[1],nodes[2])) > 0 else [nodes[2],nodes[1],nodes[0]]
				except IndexError:
					# there are less than three nodes in the 'triangle'
					# add them to a convex hull in reverse order of their y
					# because we know which side they're on this makes them anticlockwise
					convex_hull = sorted(nodes,key=lambda node:-node.y)
				try:
					#found a line connect it

					self.edges.connect(nodes[0],nodes[1])
				except IndexError:
					pass
				return convex_hull
			else:
				# nearest multiple of 3 to half the nodes
				half_length = round(len(nodes)/6) * 3
				# split into two halves
				half1 = nodes[:half_length]
				half2 = nodes[half_length:]
				# recursivly triangulate each half
				half1_hull = triangulate_(half1)
				half2_hull = triangulate_(half2)
				#get the tangents that connect the halves and stitch them together
				(half1_tt_index, half1_tt), (half2_tt_index,half2_tt) = self.get_tangent('top', half1_hull, half2_hull)				
				(half1_bt_index, half1_bt), (half2_bt_index, half2_bt) = self.get_tangent('bottom', half1_hull, half2_hull)	
				#connect the tangents
				#find the sides to be included in the convex hull
				half1_back = []
				i = half1_tt_index
				while i != half1_bt_index:
					half1_back.append(half1_hull[i])
					i += 1
					i %= len(half1_hull)
				half1_back.append(half1_hull[i])

				half2_back = []
				j = half2_bt_index
				while j != half2_tt_index:
					half2_back.append(half2_hull[j])
					j += 1
					j %= len(half2_hull)
				half2_back.append(half2_hull[j])
				# return the convex hull
				return half1_back + half2_back
		def triangulate(nodes):
			nodes.sort(key=lambda node:(node.x,node.y))
			triangulate_(nodes)

		def delaunify():
			#find all the quadrangles
				# for every node get its connections
			for A, connections in self.edges.items():
				connections = [*connections.keys()]
				print(A, 'connected_to')
				#get its connections
				for B in connections:
					print('		',B,end ='')
					print(' common connections', end=' ')
					common = self.edges.common_connections(A,B)
					pprint(common)
					#filter out the (convex hull) and (triangle within triangle) edges
					possible_cds = [(c,d) for c in common for d in common if intersect((c,d),(A,B))]
					try:
						#get the c,d with the min length to filter out quad inside quad
						C,D = min(possible_cds, key = line_length)
					except ValueError:
						print('			no possible cds')
						# if there are no possible cds
						continue
					print('			possible cds: ', possible_cds)
					print('			C,D: ',C,D)
					print()

					# make them counterclockwise relative to the center
					# A,C,B,D if A,C,B is counterclockwise else
					# D,B,C,A makes sure it's clockwise
					counterclockwise = (A,B,C,D) if signed_distance(C,(A,B)) > 0 else (D,B,C,A)
					# flip if non delaunay

					# get the determinant and see if A
					# is in the circumcircle
					matrix = [
					[B.x-A.x, B.y-A.y, (B.x -A.x)**2+(B.y - A.y)**2],
					[C.x-A.x, C.y-A.y, (C.x -A.x)**2+(C.y - A.y)**2],
					[D.x-A.x, D.y-A.y, (D.x -A.x)**2+(D.y - A.y)**2],]
					det = determinant(matrix)
					if det > 0:
						#pass
						#pprint(self.edges.data)
						self.edges.set_attr(A,B,'color',(0,255,255))
						self.edges.disconnect(A,B)
						self.edges.connect(C,D)
						delaunify()

		triangulate(self.vertices)

	def get_tangent(self, side, poly1, poly2):

		poly1_t_index, poly1_t = max(enumerate(poly1), key = lambda node: node[1].x)
		poly2_t_index, poly2_t  = min(enumerate(poly2), key = lambda node: node[1].x)
		done= False
		index_change = ((side == 'top') - (side == 'bottom'))
		#set the accumulator
		poly1_next_node_i = poly1_t_index + index_change
		poly2_next_node_i = poly2_t_index - index_change
		# make sure in index range 
		poly1_next_node_i %= len(poly1)
		poly2_next_node_i %= len(poly2)
		while not done:
			done = True # assume done until
			
			while signed_distance(poly1[poly1_next_node_i],(poly1_t, poly2_t))* index_change >= 0:


				self.edges.connect(poly1_t, poly2_t)
				poly1_t = poly1[poly1_next_node_i]
				poly1_next_node_i += index_change
				poly1_next_node_i %= len(poly1)

			if len(poly2)>1:
				while signed_distance(poly2[poly2_next_node_i],(poly2_t, poly1_t)) * -index_change >= 0:
					done = False

					self.edges.connect(poly1_t,poly2_t)
					poly2_t = poly2[poly2_next_node_i]
					poly2_next_node_i -= index_change
					poly2_next_node_i %= len(poly2)
	
		self.edges.connect(poly1_t, poly2_t)				
		return (poly1.index(poly1_t), poly1_t), (poly2.index(poly2_t), poly2_t)

	def draw_nodes(self):
		for node in self.vertices:
			#draw the node onto the surface self
			node.draw(self)

	def draw_edges(self):
		for node in self.edges:
			for connected, attrs in self.edges[node].items():
				edge = ((node.x,node.y),(connected.x,connected.y))
				pygame.draw.line(self, attrs.get('color',(0,0,0)),(node.x,node.y),(connected.x,connected.y))



def main():

	# Event loop
	try:
		# Initialise screen
		pygame.init()
		screen = pygame.display.set_mode((800, 600))
		pygame.display.set_caption('Basic Pygame program')

		

		# create the graph
		background = Graph(10, screen.get_size())
		p = Polygon([Node(0,0,background),Node(1,1,background),Node(0,1,background),Node(1,-1,background),Node(-1,1,background),Node(-1,0,background),Node(0,-1,background)])
		p.make_anticlockwise()


		# Blit everything to the screen
		screen.blit(background, (0, 0))
		pygame.display.flip()

		while 1:
			#get all events
			for event in pygame.event.get():
				#on exit
				if event.type == QUIT:
					#close pygame, necessary when running from sublime
					pygame.quit()
					#exit the main func and close the window
					return
				elif event.type == KEYDOWN and event.key == pygame.K_ESCAPE:
					pygame.quit()
					return
			background.fill((250, 250, 250)) # fill the background
			background.draw_nodes() # draw the nodes
			background.draw_edges() # draw the edges
			screen.blit(background, (0, 0))
			pygame.display.flip()
	except Exception as e:
		raise e
		print('Error:' + str(e))
		pygame.quit()
		return

if __name__ == '__main__': 
	main()