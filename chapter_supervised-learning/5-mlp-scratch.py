'''
Multilayer perception machine without gluon
'''
from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd
import mxnet
import numpy as np
import utils   # in other files

root="~/.mxnet/datasets/fashion-mnist"
def transform_mnist(data, label):
    # transform a batch of examples
    if resize:
        n = data.shape[0]
        new_data = nd.zeros((n, resize, resize, data.shape[3]))
        for i in range(n):
            new_data[i] = image.imresize(data[i], resize, resize)
        data = new_data
    # change data from batch x height x weight x channel to batch x channel x height x weight
    return nd.transpose(data.astype('float32'), (0,3,1,2))/255, label.astype('float32')

batch_size = 256
mnist_train = gluon.data.vision.FashionMNIST(root=root, train=True, transform=transform_mnist)
mnist_test = gluon.data.vision.FashionMNIST(root=root, train=False, transform=transform_mnist)
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

num_inputs = 28*28
num_outputs = 10
num_hidden = 256
weight_scale = .01
W1 = nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale)
b1 = nd.zeros(num_hidden)
W2 = nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale)
b2 = nd.zeros(num_outputs)
params = [W1 b1 W2 b2]

for param in params:
    param.attach_grad()
    
def RELU(X):
    return nd.maximum(X, 0)
def net(X):
    X = X.reshape((-1, num_inputs))
    h1 = RELU(nd.dot(X, W1) + b1)
    output = nd.dot(h1, W2) + b2
    return output
    
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

learning_rate = .5
for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        SGD(params, learning_rate/batch_size)
        
        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)
    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))
