import numpy as np
import sys
from random import shuffle
import sdf_iterator

def read_file(file_name):
	"""
	TA provided helper function to read an .sdf file.
	Given the file, this produces a list of all the molecules in a file
	(see molecule.py for the contents of a molecule).
	:param 	file_name: string, name of data file
	:return a list of molecules. The nodes are only a list of atomic numbers.
	"""
	iterator = sdf_iterator.SdfIterator(file_name)
	number_of_elements = len(iterator.periodic_table)
	mol = iterator.read_molecule()
	molecules = []
	while mol is not None:
		molecules.append(mol)
		mol = iterator.read_molecule()
	return molecules


def get_data(file_name):
	"""
	Loads the NCI dataset from an sdf file.
	After getting back a list of all the molecules in the .sdf file, there's a little more preprocessing to do.
	First, you need to one hot encode the nodes of the molecule to be a 2d numpy array
	of shape (num_atoms, 119) of type np.float32 (see molecule.py for more details).
	After the nodes field has been taken care of, shuffle the list of molecules,
	and return a train/test split of 0.9/0.1.
	:param 	file_name: string, name of data file
	:return train_data, test_data. Two lists of shuffled molecules that
		have had their nodes turned into a 2d numpy matrix, and of split 0.9 to 0.1.
	"""
	test_fraction = 0.1
	number_of_elements = 119
	molecules = read_file(file_name)
	for m in molecules:
		m.nodes = np.eye(number_of_elements, dtype=np.float32)[m.nodes, :]
	shuffle(molecules)
	train_data = molecules[ : -int(test_fraction * len(molecules))]
	test_data = molecules[int((1 - test_fraction) * len(molecules)): ]

	return train_data, test_data

