'''
Linear Regression with Gluon
'''
from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

# step 1 : generate dataset
X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

# step 2 : read data with batch size
batch_size = 10
dataset = gluon.data.ArrayDataset(X, y)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# step 3 : Define the model
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

# step 4 : Initialize the model parameters
net.initialize()

# step 5 : Loss function
square_loss = gluon.loss.L2Loss()

# step 6 : Optimize function
trainer = gluon.Trainer(
  net.collect_params(), 'sgd', {'learning_rate': 0.1})

# step 7 : Train
epochs = 5
batch_size = 10
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
    
dense = net[0]
dense.weight.data()
dense.bias.data()
