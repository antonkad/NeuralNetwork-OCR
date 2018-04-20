import numpy
import scipy.special


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes


        #   W11     W21     W31
        #   W12     w22     W32
        #   W13     W23     W33
        #Init des poids autour de 0, déviation, nombres de poids = inodes x hnodes
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # Sigmoid, special case of the logistic function (0,1)
    def sigmoid(x): # sigmoid(x) = 1/(1+exp(-x))
        return scipy.special.expit(x)

    # Gradiant stronger than sigmoid (-1,1)
    def tanh(x): # tanh(x) = 2/(1+exp(-2x)) -1  =>  tanh(x) = 2sigmoid(2x) - 1
        return np.tanh(x)

    # less computationally expensive (0,inf)
    def relu(x):
        return np.maximum(x, 0)


    def train(self, inputs_list, targets_list):
        #Convertir en tableau à dimention
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #Multiplcation matriciel couche Entrée vers couche cachée
        hidden_inputs = numpy.dot(self.wih, inputs)
        #Fonction d'activation
        hidden_outputs = self.activation_function(hidden_inputs)

        #Caché vers Output
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #Target : 0.1-0.99
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        #Update des poids
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        #for index, value in enumerate(final_outputs):
        #    print(index, ' ', value)
        pass


    def query(self, inputs_list):
        # convertion
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
