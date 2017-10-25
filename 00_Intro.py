import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

num1=tf.constant(6)
num2=tf.constant(5)

result=tf.multiply(num1,num2)

#print(result)

print("##############################Result over here!###################################################")

with tf.Session() as sess:
    output = sess.run(result)
    print(output)


print(output)

