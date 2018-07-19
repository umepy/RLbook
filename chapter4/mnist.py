import torch
from sklearn.datasets import fetch_mldata
from six.moves import urllib
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
try:
    mnist = fetch_mldata('MNIST original')
except:
    print("Could not download MNIST data from mldata.org, trying alternative...")

    # Alternative method to load MNIST, if mldata.org is down
    from scipy.io import loadmat
    mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
    mnist_path = "./mnist-original.mat"
    response = urllib.request.urlopen(mnist_alternative_url)
    with open(mnist_path, "wb") as f:
        content = response.read()
        f.write(content)
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }
    print("Success!")

X = mnist.data / 255
y = mnist.target

plt.imshow(X[0].reshape(28, 28), cmap='gray')
print('label is {:.0f}'.format(y[0]))