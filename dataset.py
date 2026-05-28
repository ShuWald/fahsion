import mnist_reader
X_train, y_train = mnist_reader.load_mnist('dataset', kind='train')
X_test, y_test = mnist_reader.load_mnist('dataset', kind='t10k')
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)