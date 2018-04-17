#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import tensorflow as tf

tf.summary.FileWriterCache.clear()

batch_size = 100
learning_rate = 0.01
training_epochs = 10

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

x =  tf.placeholder(tf.float32, [None, 784], name="x-input")
y_ =  tf.placeholder(tf.float32, [None, 10], name="y-input")

# Reseau dans couche caché les vecteurs vont des 784 inputs -> 10 outputs
W = tf.Variable(tf.zeros([784, 10]))
#Bias (à revoir)
b = tf.Variable(tf.zeros([10]))
#Softmax: exp(y)/somme(exp(y)
#matmul: multiplication (pareil que dot, mais ne gère pas scalar multiplication
y = tf.nn.softmax(tf.matmul(x,W) + b)
#reduce mean: moyenne arithmetique
#reduce sum: somme sur la deuxième dimention (0 vertical/colone - 1 horizontal/ligne)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#equal: revoie un boolean si les deux valeur sont egale ou pas
#argmax: renvoie l'index de la valeur la plus haute sur la dimention definit
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#cast: change le type d'un tensor, ici bool vers float32
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op =  tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
   init = tf.global_variables_initializer()
   sess.run(init)
   writer = tf.summary.FileWriter('/mnt/c/Users/Anton/Documents/NeuralNetwork/temp/...', sess.graph)
   
   for epoch in range(training_epochs):
      batch_count = int(mnist.train.num_examples/batch_size)
      for i in range(batch_count):
         batch_x, batch_y = mnist.train.next_batch(batch_size)

         sess.run([train_op], feed_dict={x: batch_x, y_:batch_y})

      if epoch % 2 == 0:
         print("Epoch: ", epoch)
   print("Accuracy: ", accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels}))
   print("done")
