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

net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))

net.initialize()

square_loss = gluon.loss.L2Loss()

trainer = gluon.Trainer(
  net.collect_params(), 'sgd', {'learning_rate': 0.1})

