import numpy as np 
import random 


class feed_forward_net(object):
	
	def __init__(self, input_size, hidden_layers_size, output_size):

		#create input layers
		self.input_layer = np.zeros(input_size, dtype = float)
		self.input_layer = np.append(self.input_layer, [-1])

		self.weights = []
		hidden_layers_size = [x+1 for x in hidden_layers_size]
		prev_layer_size =np.append([input_size+1], hidden_layers_size)


		#create hidden layers
		for i in range(len(hidden_layers_size)):
				y = prev_layer_size[i]
				x = prev_layer_size[i+1]
				layer_weight = np.zeros((y ,x), dtype = float)
				for j in range(y):
					for k in range(x):
						layer_weight[j][k] = random.uniform(0, 1)
				print("layer weight: ")
				print(layer_weight)
				self.weights.append(layer_weight)

		#create output layers
		y = prev_layer_size[len(prev_layer_size) - 1]
		x = output_size
		layer_weight = np.zeros((y ,x), dtype = float)
		for j in range(y):
			for k in range(x):
				layer_weight[j][k] = random.uniform(0, 1)
		print("layer weight: ")
		print(layer_weight)
		self.weights.append(layer_weight)

		self.output_layer = np.zeros(output_size, dtype = float)



	def feed_forward(self, inputs):

		print("inputs: ")
		print(inputs)
		outputs = self.input_layer

		if len(inputs)+1 != len(self.input_layer):
			print("feeding inputs demsion not matching input layer demsion...")
			return

		for i in range(len(inputs)):
			outputs[i] = inputs[i]/10000

		for layer_weight in self.weights:
			outputs = np.dot(outputs, layer_weight)
			outputs = self.sigmoid(outputs)
			print("outputs: ")
			print(outputs)
		print("output_layer before: ")
		print(self.output_layer)
		for i in range(len(outputs)):
			self.output_layer[i] = outputs[i]
		print("output_layer: ")
		print(self.output_layer)


	def backpropagation(self, labels):

		labels = np.divide(labels, 10000).reshape(len(labels), 1)
		print("labels: ")
		print(labels)

		outputs = self.output_layer.reshape(len(self.output_layer), 1)
		print("outputs: ")
		print(outputs)
		print("outputs 1:")
		print(self.output_layer)

		last_neuron_error = np.subtract(labels, outputs)
		print("last_neuron_error: ")
		print(last_neuron_error)

		last_neuron_correction = np.subtract(1, outputs)
		print("last_neuron_correction: ")
		print(last_neuron_correction)

		



	def sigmoid(self, inputs):
		return 1 / (1 + np.exp(-inputs))



	def sigmoid_p(self, inputs):
		return self.sigmoid(inputs)*(1 - self.sigmoid(inputs))


def main():
	nn = feed_forward_net(3,[3, 5, 2], 2)

	inputs = [1201, 2290, 1293]
	nn.feed_forward(inputs)

	labels = [4784, 2031]
	nn.backpropagation(labels)

if __name__ == '__main__':
	main()

