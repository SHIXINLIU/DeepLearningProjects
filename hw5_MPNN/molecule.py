class Molecule:
	"""
	Data structure representing a molecule.
	:param	nodes: When they are first parsed, nodes is a list of integers of size
		representing the atomic number of each of the atoms in the molecule
		(e.g for a molecule with atoms C, H, and F, it might look like [6, 1, 9]).
		In preprocess.py, this should be turned into a 2d numpy array
		representing the one-hotted versions of these numbers of size (num_atoms, 119)
		-- 119 is the size of our periodic table, and thus the max of our one-hotted array.
	:param 	edges: A list of tuples --
		each tuple has two integers i and j representing a connection between the ith and jth nodes.
	:param 	label: np.long value of 1 if active against cancer, 0 if not.
	"""
	def __init__(self, nodes, edges, label):
		self.nodes = nodes
		self.edges = edges
		self.label = label
