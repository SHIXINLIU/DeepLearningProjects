from molecule import Molecule
from periodictable import elements
import torch as th
import numpy as np
""" Code provided by CS1470 TA Staff
"""
class SdfIterator:
	""" 
	Iterator Object used on a single .sdf file
	Split the file by the separator ($$$$), and for each molecule generated by this,
	extract the atoms, bonds and label of each.
	:param	file_name: string, name of data file
	"""
	def __init__(self, file_name):
		lines_unstripped = open(file_name, 'r').readlines()
		self.file_lines = [line.strip() for line in lines_unstripped]
		self.line = 0 # Counter for where we are in the file
		self.periodic_table = {} # Need atomic number as feature instead of symbol.
		for el in elements:
			self.periodic_table[el.symbol] = el.number

	def read_molecule(self):
		"""
		Read a single molecule from the .sdf file
		Starting from the line we left off, (or 0 if beginning), read lines until we hit a $$$$.
		Remember what line number we started at, and where we end.
		pass a slice of the array with these bounds to be turned into a Molecule, and return.
		"""
		molecule_line_start = self.line
		if self.line == len(self.file_lines):
			self.line = 0
			return None
		while self.file_lines[self.line] != "$$$$":
			self.line += 1
			if self.line == len(self.file_lines):
				self.line = 0
				return None
		self.line += 1
		molecule_lines = self.file_lines[molecule_line_start:self.line]
		mol = self.create_molecule(molecule_lines)
		return mol

	def create_molecule(self, lines):
		"""
		Creates a molecule object from its file lines.
		Split all the lines by whitespace so we can get the info we care about.
		for atoms, only get the symbol.
		for bonds, only get the first two elements (indices of atoms connected by bond).
		Only other thing we need is the label: 1 == active against cancer, -1 else.
		:param	lines: file lines
		"""
		lines = [l.split() for l in lines]
		atoms = [l for l in lines if len(l) == 10]
		bonds = [l for l in lines if len(l) == 6]
		nodes = [self.periodic_table[a[3]] for a in atoms]
		edges = [(int(b[0])-1, int(b[1])-1) for b in bonds]
		label = int(float(lines[-3][0]))
		if label == -1:
			label = 1
		else:
			label = 0
		label = np.array(label).astype(np.long)
		return Molecule(nodes, edges, label)

