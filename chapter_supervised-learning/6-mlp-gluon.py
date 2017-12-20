'''
Multilayer perception machine with Gluon
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

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(256, activation="relu"))
    net.add(gluon.nn.Dense(10))
net.initialize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += utils.accuracy(output, label)

    test_acc = utils.evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data), test_acc))
