from script import neuralNetwork
import sys
import numpy
from numpy import genfromtxt
import pathlib
import matplotlib.pyplot as plt

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

# number of input, hidden and output nodes
input_nodes = 28*28
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

data_file = open(sys.argv[2])
data_list = data_file.readlines()
data_file.close()

def train():
    i = 10
    for e in range(i):
        for record in data_list:
            all_values = record.split(',')
            #On ramène les valeurs de 0-255 à 0.1-1.0
            # numpy.asfarray : Renvoie un array avec des float
            scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # numpy.zeros : Return a new array of given shape and type, filled with zeros.
            targets = numpy.zeros(output_nodes) + 0.01
            # on met 0.99 à la valeur qui correspond à la bonne réponse
            targets[int(all_values[0])] = 0.99
            print("label", all_values[0])
            n.train(scaled_input, targets)
            pass
        pass
    numpy.savetxt("who.csv", n.who, delimiter=",")
    numpy.savetxt("wih.csv", n.wih, delimiter=",")




def run():
    p = pathlib.Path('who.csv')
    if p.is_file():  # or p.is_dir() to see if it is a directory
        n.who = genfromtxt('who.csv', delimiter=',')
        n.wih = genfromtxt('wih.csv', delimiter=',')

    scorecard = []
    for record in data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        print(outputs)
        label = numpy.argmax(outputs)
        print("-----label ", label)
        print("-----Correct label ", correct_label, "\n")

        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        pass
    print(scorecard)
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)

if (sys.argv[1] == '-t'):
    train()
elif (sys.argv[1] == '-r'):
    run()
else:
    print("Aucun arguments")
