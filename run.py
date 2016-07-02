import mnist_loader
import network3 
from network3 import Network 
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer 


mnist_loader.load_data_wrapper()

training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 100
net = Network([FullyConnectedLayer(n_in=784, n_out=4096),FullyConnectedLayer(n_in=4096, n_out=4096),FullyConnectedLayer(n_in=4096, n_out=4096),FullyConnectedLayer(n_in=4096, n_out=4096),FullyConnectedLayer(n_in=4096, n_out=10)], mini_batch_size)
net.SGD(training_data, 1000, mini_batch_size, 0.003, xvalidation_data, test_data)



