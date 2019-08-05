import tflearn.datasets.mnist as mnist
from random import randrange, choice

class MNISTDatasets:
    
    def __init__(self):
        mnist_data = mnist.load_data(one_hot = True)
        self.training_images  = self.create_dataset (mnist_data[:2])
        self.testing_images   = self.create_dataset (mnist_data[2:])
        
        self.image_size     = 28 * 28 #--- size of input (mnist images are 28x28)
        self.num_categories = 10      #--- size of output (images can be either 0, 1, 2, 3, 4, 5, 6, 7, 8 9)
        
        return
    
    def create_dataset(self, mnist_dataset):
        return (self.get_images(mnist_dataset), self.get_categories(mnist_dataset))
    
    def get_images(self, mnist_dataset):
        return mnist_dataset[0]
    
    def get_categories(self, mnist_dataset):
        return mnist_dataset[1]
    
    
    def get_random_digit (self, datasets=None):
        
        if not datasets:
            datasets = [self.training_images, self.testing_images]
        dataset    = choice(datasets)
        images     = self.get_images(dataset)
        categories = self.get_categories(dataset)
        
        i = randrange(len(images))

        return (images[i].reshape([28,28]), categories[i])