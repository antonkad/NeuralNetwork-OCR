import sys
import numpy
import matplotlib.pyplot as plt

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument List:', str(sys.argv))

data_file = open(sys.argv[1])
data_list = data_file.readlines()
data_file.close()

all_values = data_list[1].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')

scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)

print("List size :", len(data_list))
print(image_array)

plt.show()
