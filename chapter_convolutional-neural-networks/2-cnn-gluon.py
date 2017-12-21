'''
CNN without gluon
'''
import maxnet as mx
from maxnet import autograd
from maxnet import ndarray as nd
from maxnet import gluon

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')
# root='./'
mnist_train = gluon.data.vision.FashionMNIST(root='~/.mxnet/datasets/fashion-mnist', train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(root='~/.mxnet/datasets/fashion-mnist', train=False, transform=transform)
batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
   
from mxnet.gluon import nn
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(10)
    )
net.initialize()
    
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

from utils import accuracy, evaluate_accuracy
learning_rate = .2

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    for data, label in train_data:
        with autograd.record():
            output = net(data)
            Loss = loss(output, label)
        Loss.backward()
        trainer.step(batch_size)

        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    test_acc = evaluate_accuracy(test_data, net)
    print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data), test_acc))
