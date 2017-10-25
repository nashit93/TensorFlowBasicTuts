import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

batch_size = 100

n_hidden_layers_1 = 500
n_hidden_layers_2 = 500
n_hidden_layers_3 = 500

n_output = 10

x = tf.placeholder('float',[None, 784])
y= tf.placeholder('float')


def neural_network_model(data):
    hl1 = {"weights":tf.Variable(tf.random_normal([784,n_hidden_layers_1])),
    "biases":tf.Variable(tf.random_normal([n_hidden_layers_1]))}

    hl2 = {"weights":tf.Variable(tf.random_normal([n_hidden_layers_1,n_hidden_layers_2])),
    "biases":tf.Variable(tf.random_normal([n_hidden_layers_2]))}

    hl3 = {"weights":tf.Variable(tf.random_normal([n_hidden_layers_2,n_hidden_layers_3])),
    "biases":tf.Variable(tf.random_normal([n_hidden_layers_3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_hidden_layers_3,n_output])),
    "biases": tf.Variable(tf.random_normal([n_output]))}

    l1 = tf.add(tf.matmul(data,hl1["weights"]),hl1["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hl2["weights"]),hl2["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hl3["weights"]),hl3["biases"])
    l3 = tf.nn.relu(l3)


    output = tf.add(tf.matmul(l3,output_layer["weights"]),output_layer["biases"])


    return output

def train_neural_net(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimiser = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 100

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = sess.run([optimiser,cost],feed_dict = {x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch,' completed out of ', hm_epochs,' Loss : ',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy : ',accuracy.eval({x:mnist.test.images,y: mnist.test.labels}))
    





train_neural_net(x)
