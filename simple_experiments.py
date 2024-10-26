from models import CNN
from experiments import Experiment
from data import get_mnist, get_fashion_mnist

#cnn = CNN()
#fashion_mnist_train_loader, fashion_mnist_test_loader = get_fashion_mnist()
#exp = Experiment("train_fashion_mnist", cnn, fashion_mnist_train_loader, fashion_mnist_test_loader, False, False, False, 20)
#exp.run()

cnn = CNN()
fashion_mnist_train_loader, fashion_mnist_test_loader = get_fashion_mnist()
exp = Experiment("train_fashion_mnist_sam", cnn, fashion_mnist_train_loader, fashion_mnist_test_loader, True, True, False, 20)
exp.run()

'''
cnn = CNN()
mnist_train_loader, mnist_test_loader = get_mnist()
exp = Experiment("train_mnist", cnn, mnist_train_loader, mnist_test_loader, False, False, False)
exp.run()

cnn = CNN()
mnist_train_loader, mnist_test_loader = get_mnist()
exp = Experiment("train_mnist_sam", cnn, mnist_train_loader, mnist_test_loader, True, False, False)
exp.run()
'''