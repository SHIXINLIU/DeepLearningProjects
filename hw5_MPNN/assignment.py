import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import *
import dgl

class Model(nn.Module):
	"""
	Model class representing your MPNN. This is (almost) exactly like your Model class from TF2,
	but it inherits from nn.Module instead of keras.Model.
	Treat it identically to how you treated your Model class from previous assignments.
	"""
	def __init__(self):
		"""
		Init method for Model class. Instantiate a lifting layer, an optimizer,
		some number of MPLayers (we recommend 3), and a readout layer going from
		the latent space of message passing to the number of classes.
		"""
		super(Model, self).__init__()

		# Initialize hyperparameters
		self.raw_features = 119
		self.lift_features = 300
		self.num_classes = 2
		self.batch_size = 10

		# Initialize trainable parameters
		self.lifting = nn.Linear(self.raw_features, self.lift_features)
		self.mp1 = MPLayer(self.lift_features, self.lift_features)
		self.mp2 = MPLayer(self.lift_features, self.lift_features)
		self.mp3 = MPLayer(self.lift_features, self.lift_features)
		self.read_out = nn.Linear(self.lift_features, self.num_classes)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)

	def forward(self, g):
		"""
		Responsible for computing the forward pass of your network.
		Analagous to "call" methods from previous assignments.
		1)	The batched graph that you pass in hasn't had its node features lifted yet.
			Pop them, run them through your lifting layer. Don't apply an activation function. 
		2) 	After the node features for the graph have been lifted, run them through
				the mp layers (ReLUing the result returned from each).
		3) 	After ReLUing the result of your final mp layer, feed it through the
			readout function in order to get logits.
		:param 	g: The DGL graph you wish to run inference on.
		:return	logits tensor of size (batch_size, 2)
		"""
		lifted_feats = self.lifting(g.ndata.pop("data"))
		feats1 = torch.relu(self.mp1(g, lifted_feats))
		feats2 = torch.relu(self.mp2(g, feats1))
		feats3 = torch.relu(self.mp3(g, feats2))
		logits = self.readout(g, feats3)

		return logits

	def readout(self, g, node_feats):
		""" 
		Responsible for reducing the dimensionality of the graph to num_classes,
		and summing the node features in order to return logits.
		Set your node features to be the output of your readout layer on node_feats,
		then use dgl.sum_nodes to return logits.
		:param	g: The batched DGL graph
		:param 	node_feats: The features at each node in the graph. Tensor of shape
							(num_atoms_in_batched_graph, size_of_node_vectors_from_prev_message_passing)
		:return	logits tensor of size (batch_size, 2)
		"""
		g.ndata['data'] = self.read_out(node_feats)
		logits = dgl.sum_nodes(g, 'data')

		return logits

	def accuracy_function(self, logits, labels):
		"""
		Computes the accuracy across a batch of logits and labels.
		:param 	logits: a 2-D np array of size (batch_size, 2)
		:param 	labels: a 1-D np array of size (batch_size).(1 for if the molecule is active against cancer, else 0).
		:return	mean accuracy over batch.
		"""
		true_pred = np.equal(np.argmax(logits, 1), labels)
		return np.mean(np.array(true_pred))


class MPLayer(nn.Module):
	"""
	A PyTorch module designed to represent a single round of message passing.
	This should be instantiated in your Model class several times.
	"""
	def __init__(self, in_feats, out_feats):
		"""
		Init method for the MPLayer. You should make a layer that will be
		applied to all nodes as a final transformation when you've finished
		message passing that maps the features from size in_feats to out_feats (in case you
		want to change the dimensionality of your node vectors at between steps of message
		passing). You should also make another layer to be used in computing your messages.
		:param	in_feats: The size of vectors at each node of your graph when you begin
						message passing for this round.
		:param	out_feats: The size of vectors that you'd like to have at each of your
						nodes when you end message passing for this round.
		"""
		super(MPLayer, self).__init__()

		self.in_feats = in_feats
		self.out_feats = out_feats
		self.f1 = nn.Linear(self.in_feats, self.out_feats)
		self.f2 = nn.Linear(self.out_feats, self.out_feats)

	def forward(self, g, node_feats):
		"""
		Responsible for computing the forward pass of your network.
		Analagous to "call" methods from previous assignments.
		1) You should reassign g's ndata field to be the node features that were popped off
			in the previous layer (node_feats).
		2) Trigger message passing and aggregation on g using the send and recv functions.
		3) Pop the node features, and then feed it through a linear layer and return.
		You can assign/retrieve node data by accessing the graph's ndata field with some attribute
		that you'd like to save the features under in the graph (e.g g.ndata["h"] = node_feats)
		:param 	g: The batched DGL graph you wish to run inference on.
		:param 	node_feats: Beginning node features for your graph.
							should be a torch tensor (float) of shape (number_atoms_batched_graph, in_feats).
		:return	node_features of size (number_atoms_batched_graph, out_feats)
		"""
		g.ndata['data'] = node_feats
		g.send(g.edges(), self.message)
		g.recv(g.nodes(), self.reduce)
		return self.f2(g.ndata.pop('data'))

	def message(self, edges):
		"""
		A function to be passed to g.send. This function, when called on a group of edges,
		should compute a message for each one of them. Each message will then be sent
		to the edge's "dst" node's mailbox.
		The particular rule to compute the message should be familiar. The message from
		node n1 with node feature v1 to n2 should be ReLU(f(v1)), where f is a feed-forward layer.
		:param 	edges: All the DGL edges in the batched DGL graph.
		:return	A map from some string (you choose) to all the messages computed for each edge.
				These messages can then be retrieved at each destination node's mailbox
				(e.g destination_node.mailbox["string_you_chose"]) once DGL distributes them with the send function.
		"""
		return {'hi': torch.relu(self.f1(edges.src['data']))}

	def reduce(self, nodes):
		"""
		A function to be passed to g.recv. This function, when called on a group of nodes,
		should aggregate (ie. sum) all the messages received in their mailboxes from message passing.
		DGL will then save these new features in each node under the attribute you set (see the return).
		The messages in each node can be accessed like:
		nodes.mailbox['string_you_chose_in_message']
		:param 	nodes: All the DGL nodes in the batched DGL Graph.
		:return	A Map from string to the summed messages for each node.
				The string should be the same attribute you've been using to access ndata this whole time.
				The node data at this attribute will be updated to the summed messages by DGL.
		"""
		return {'data': nodes.mailbox['hi'].sum(1)}


def build_graph(molecule):
	"""
	Constructs a DGL graph out of a molecule from the train/test data.
	:param	molecule: a Molecule object (see molecule.py for more info)
	:return	A DGL Graph with the same number of nodes as atoms in the molecule, edges connecting them,
			and node features applied.
	"""
	# TODO: Initialize a DGL Graph
	G = dgl.DGLGraph()
	# TODO: Call the graph's add_nodes method with the number of nodes in the molecule.
	G.add_nodes(len(molecule.nodes))
	# TODO: Turn molecule's nodes into a tensor, and set it to be the data of this graph.
	G.ndata['data'] = torch.tensor(molecule.nodes)
	# TODO: Construct a tuple of src and dst nodes from the list of edges in molecules.
	#      e.g if the edges of the molecule looked like [(1,2), (3,4), (5,6)] return (1,3,5) and (2,4,6).
	src, dst = tuple(zip(*molecule.edges))
	# TODO: Call the graph's add_edges method to add edges from src to dst and dst to src.
	#       Edges are directed in DGL, but undirected in a molecule, so you have to add them both ways.
	G.add_edges(src, dst)
	G.add_edges(dst, src)

	return G


def train(model, train_data):
	"""
	Trains your model given the training data.
	For each batch of molecules in train data...
		1) Make dgl graphs for each of the molecules in your batch; collect them in a list.
		2) call dgl.batch to turn your list of graphs into a batched graph.
		3) Turn the labels of each of the molecules in your batch into a 1-D tensor of size batch_size
		4) Pass this graph to the Model's forward pass. Run the resulting logits
			and the labels of the molecule batch through nn.CrossEntropyLoss.
		3) Zero the gradient of your optimizer.
		4) Do backprop on your loss.
		5) Take a step with the optimizer.
	Note that nn.CrossEntropyLoss expects Logits, not probabilities. It contains a softmax layer on its own.
	Your model won't train well if you pass it probabilities.
	:param 	model: Model class representing your MPNN.
	:param 	train_data: A 1-D list of molecule objects, representing all the molecules in the training set from get_data
	:return	nothing.
	"""
	loss = nn.CrossEntropyLoss()
	N = len(train_data) // model.batch_size
	for batch in range(N):
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
		if (batch + 1) * model.batch_size > len(train_data): end = len(train_data)
		train_list = train_data[start: end]
		gh = []
		lb = []
		for i in range(0, model.batch_size):
			gh.append(build_graph(train_list[i]))
			lb.append(train_list[i].label)
		graphs = dgl.batch(gh)
		logits = model(graphs)
		labels = torch.tensor(np.array(lb))
		l = loss(logits, labels)
		model.optimizer.zero_grad()
		l.backward()
		model.optimizer.step()


def test(model, test_data):
	"""
	Testing function for our model.
	Batch the molecules in test_data, feed them into your model as described in train.
	After you have the logits: turn them back into numpy arrays, compare the accuracy to the labels,
	and keep a running sum.
	:param	model: Model class representing your MPNN.
	:param 	test_data: A 1-D list of molecule objects, representing all the molecules in your
			testing set from get_data.
	:return	total accuracy over the test set (between 0 and 1)
	"""
	sum_true = 0
	sum_pred = 0
	N = len(test_data) // model.batch_size
	for batch in range(N):
		start = batch * model.batch_size
		end = (batch + 1) * model.batch_size
		if (batch + 1) * model.batch_size > len(test_data): end = len(test_data)
		test_list = test_data[start: end]
		gh = []
		lb = []
		for i in range(0, model.batch_size):
			gh.append(build_graph(test_list[i]))
			lb.append(test_list[i].label)
		graphs = dgl.batch(gh)
		logits = model.forward(graphs).detach().numpy()  # must call detach
		labels = torch.tensor(np.array(lb))
		acc = model.accuracy_function(logits, labels)
		sum_pred += (end - start)
		sum_true += acc * (end - start)
	return sum_true / sum_pred


def main():
	train_data, test_data = get_data('1-balance.sdf')
	model = Model()
	num_epoch = 15
	for epoch in range(num_epoch):
		print('Epoch {}: '.format(epoch))
		train(model, train_data)
		acc = test(model, test_data)
		print('Test Accuracy = {0:.2f} %'.format(acc * 100))


if __name__ == '__main__':
	main()
