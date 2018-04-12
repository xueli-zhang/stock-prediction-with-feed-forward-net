import numpy as np 
import random 


class feed_forward_net(object):
	
	def __init__(self, input_size, hidden_layers_size, output_size, learning_rate = 0.5):

		#create input layers
		self.input_layer = np.zeros(input_size, dtype = float)
		self.input_layer = np.append(self.input_layer, [-1])
		self.learning_rate = learning_rate

		self.weights = []
		hidden_layers_size = [x+1 for x in hidden_layers_size]
		prev_layer_size =np.append([input_size+1], hidden_layers_size)

		self.errors = []


		#create hidden layers
		for i in range(len(hidden_layers_size)):
				y = prev_layer_size[i]
				x = prev_layer_size[i+1]
				layer_weight = np.zeros((y ,x), dtype = float)
				for j in range(y):
					for k in range(x):
						layer_weight[j][k] = random.uniform(0, 1)
				#print("layer weight: ")
				#print(layer_weight)
				self.weights.append(layer_weight)

		#create output layers
		y = prev_layer_size[len(prev_layer_size) - 1]
		x = output_size
		layer_weight = np.zeros((y ,x), dtype = float)
		for j in range(y):
			for k in range(x):
				layer_weight[j][k] = random.uniform(0, 1)
		#print("layer weight: ")
		#print(layer_weight)
		self.weights.append(layer_weight)

		self.output_layer = np.zeros(output_size, dtype = float)

		print("input layer: ")
		print(self.input_layer)

		print("weights: ")
		print(self.weights)

		print("output layer: ")
		print(self.output_layer)



	def feed_forward(self, inputs):

		print("inputs: ")
		print(inputs)
		self.activations = []
		outputs = self.input_layer


		if len(inputs)+1 != len(self.input_layer):
			print("feeding inputs demsion not matching input layer demsion...")
			return

		for i in range(len(inputs)):
			outputs[i] = inputs[i]/10000

		for layer_weight in self.weights:
			outputs = np.dot(outputs, layer_weight)
			outputs = self.sigmoid(outputs)
			self.activations.append(outputs.reshape(len(outputs), 1))
			#print("outputs: ")
			#print(outputs)
		self.activations = self.activations[:-1]
		#print("activations: ")
		#print(self.activations)
		#print("output_layer before: ")
		#print(self.output_layer)
		for i in range(len(outputs)):
			self.output_layer[i] = outputs[i]
		#print("output_layer: ")
		#print(self.output_layer)


	def backpropagation(self, labels):

		labels = np.divide(labels, 10000).reshape(len(labels), 1)
		#print("labels: ")
		#print(labels)

		outputs = self.output_layer.reshape(len(self.output_layer), 1)
		#print("outputs: ")
		#print(outputs)
		#print("outputs 1:")
		#print(self.output_layer)

		last_neuron_error = np.subtract(labels, outputs)
		#print("last_neuron_error: ")
		#print(last_neuron_error)

		self.errors.append(last_neuron_error)

		last_neuron_correction = np.subtract(1, outputs)
		#print("last_neuron_correction: ")
		#print(last_neuron_correction)

		layer_corrections = []


		layer_corrections.append(last_neuron_correction)
		#print("layer corrections: ")
		#print(layer_corrections)

		for i in reversed(range(len(self.weights))):
			if i != 0:
				new_correction = np.multiply(np.multiply(self.activations[i-1], np.subtract(1, self.activations[i-1])),
					np.dot(self.weights[i], layer_corrections[len(self.weights)-1-i]))
				layer_corrections.append(new_correction)

		layer_corrections = layer_corrections[::-1]
		print("layer corrections: ")
		print(layer_corrections)

		#update weights
		input_and_hidden = []
		input_and_hidden.append(self.input_layer.reshape(len(self.input_layer), 1))

		for i in self.activations:
			input_and_hidden.append(i)


		print("input_and_hidden: ")
		print(input_and_hidden)

		for i in range(len(input_and_hidden)):

			for j in range(len(input_and_hidden[i])):

				for k in range(len(layer_corrections[i])):

					self.weights[i][j][k] += self.learning_rate * input_and_hidden[i][j][0] * layer_corrections[i][k][0]

		print("new weights:")
		print(self.weights)

	def sigmoid(self, inputs):
		return 1 / (1 + np.exp(-inputs))


	def train(self, train_dataset, )



def main():
	nn = feed_forward_net(3,[3, 5, 2], 2)

	inputs = [1201, 2290, 1293]
	nn.feed_forward(inputs)

	labels = [4784, 2031]
	nn.backpropagation(labels)

if __name__ == '__main__':
	main()

