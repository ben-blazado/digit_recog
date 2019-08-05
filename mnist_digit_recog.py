import tflearn

class MNISTDigitRecog:
    
    def __init__(self, mnist_datasets, hidden_layers_sizes=[128, 32]):
    
        self.mnist_datasets = mnist_datasets
        input_size = self.mnist_datasets.image_size
        output_size = self.mnist_datasets.num_categories

        net = tflearn.input_data([None, input_size])

        for size in hidden_layers_sizes:
            net = tflearn.fully_connected(net, size, activation='ReLU')

        net = tflearn.fully_connected(net, output_size, activation='softmax')
        net = tflearn.regression(net, optimizer='sgd', learning_rate=0.01, loss='categorical_crossentropy')

        self.model = tflearn.DNN(net)
        
        self.losses     = []
        self.accuracies = []
        
        return
    
    
    def update_training_history(self, training_state):
        self.losses     += [training_state.loss_value]
        self.accuracies += [training_state.acc_value]

        return
    
    
    def train(self, n_epoch=100):
        
        class training_callback(tflearn.callbacks.Callback):

            def __init__(self, target):
                self.target = target
                return
            
            def on_epoch_end(self, training_state):
                self.target.update_training_history(training_state)
                return
        
        self.losses     = []
        self.accuracies = []
        training_callback = training_callback(self)
        
        training_images, training_categories = self.mnist_datasets.training_images
        self.model.fit(training_images, training_categories, callbacks=training_callback, 
                  validation_set=0.1, show_metric=True, batch_size=100, n_epoch=n_epoch)
        
        return
    
    
    def identify(self, image):
        
        prediction = self.model.predict(image.reshape([1, -1]))
        
        return prediction.argmax()