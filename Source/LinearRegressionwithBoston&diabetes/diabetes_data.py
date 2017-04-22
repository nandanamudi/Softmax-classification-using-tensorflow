#importing the libraries required

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#importing numpy.genfromtxt to read the numerical values from a text file
from numpy import genfromtxt

#importing the diabetes dataset from sklearn
from sklearn import linear_model
from sklearn.datasets import load_diabetes


#reading the dataset to numpy.arrray
def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

#reading the diabetes datset to numpy.array and returing features and labels
def read_diabetes_data():
    diabetes = load_diabetes()
    features = np.array(diabetes.data)
    labels = np.array(diabetes.target)
    return features, labels

#features of the dataset is normalized
def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma


#appending biased term to the normalized features
def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l


#dividing into training and testing

features,labels = read_diabetes_data()
normalized_features = feature_normalize(features)
f, l = append_bias_reshape(normalized_features,labels)
n_dim = f.shape[1]

rnd_indices = np.random.rand(len(f)) < 0.80

x_train = f[rnd_indices]
y_train = l[rnd_indices]
x_test  = f[~rnd_indices]
y_test  = l[~rnd_indices]


learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

#inserts a placeholder for tensor to  and used to feed data to session.run() using feed_dict

X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,1])


# create a shared variable
W = tf.Variable(tf.ones([n_dim,1]))

# Initializing the variables
init = tf.initialize_all_variables()


y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))

# construct an optimizer to minimize cost and fit line to my data
training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_op,feed_dict={X:x_train,Y:y_train})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: x_train,Y: y_train}))


#plot the curve

plt.plot(range(len(cost_history)), cost_history,color='g')
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

pred_y = sess.run(y_, feed_dict={X: x_test})
mse = tf.reduce_mean(tf.square(pred_y - y_test))

#prinitng the mean squared error
print("MSE: %.4f" % sess.run(mse))


#Printing the variance
print('Variance: %.2f' % regr.score(x_test, y_test))


#ploting the curve showing measured vs predicted values
fig, ax = plt.subplots()
ax.scatter(y_test, pred_y)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, color='r')
ax.set_xlabel('Measured Value')
ax.set_ylabel('Predicted Value')
plt.show()

sess.close()