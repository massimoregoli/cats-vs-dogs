from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import applications
from time import time
import pandas as pd
import abc


class AbstractModel(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_instance(params):
        return ModelFactory.get_instance(params)

    def __init__(self, params):

        self._params = params

        self._classifier = None
        self._training_set = None
        self._test_set = None

    @abc.abstractmethod
    def build_model(self):
        return


    def prepare_train_data(self):

        # configures data generators, during training data augmentation strategy is adopted
        train_data_generator = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, shear_range = 0.2, horizontal_flip = True)
        val_data_generator = ImageDataGenerator(rescale = 1./255)

        # configures data flow for training and validation
        self._training_set = train_data_generator.flow_from_directory(self._params['dataset_dir'] + '/train', shuffle = True, target_size = (self._params['img_width'], self._params['img_height']), batch_size = self._params['batch_size'], class_mode = 'binary')
        self._val_set = val_data_generator.flow_from_directory(self._params['dataset_dir'] + '/val', target_size = (self._params['img_width'], self._params['img_height']), batch_size = self._params['batch_size'], class_mode = 'binary')

        print 'training and validation data configured'

    def fit_data(self):

        # saves TensorBoard logs
        tensorboard = TensorBoard(log_dir='logs/' + self._params['net'] + '_{}'.format(time()))

        # saves the best model obtained during training
        checkpointer = ModelCheckpoint(filepath='models/' + self._params['model_to_save_name'], verbose = 1, save_best_only=True)

        self._classifier.fit_generator(self._training_set, epochs = self._params['epochs'], steps_per_epoch = self._params['train_set_size'] // self._params['batch_size'], validation_data = self._val_set, validation_steps = self._params['val_set_size'] // self._params['batch_size'], callbacks=[checkpointer,tensorboard])

        print 'training process complete!'

    def load_model(self):

        # loads a previously stored model
        self._classifier = load_model('models/' + self._params['model_to_load_name'])

        print 'model ' + self._params['model_to_load_name'] + ' successfully loaded'

    def prepare_test_data(self):

        # configures test data generator
        test_data_generator = ImageDataGenerator(rescale = 1./255)

        # configures test data flow
        self._test_set = test_data_generator.flow_from_directory(self._params['dataset_dir'] + '/test', target_size = (self._params['img_width'], self._params['img_height']), batch_size = 1, class_mode = None)

        print 'test data configured'

    def test_model(self):

        # provides an array of probabilities predicting whether test images represent dogs rather than cats
        # note that predictions are ordered according to the related image filename and NOT to the image ID ('10.jpg' comes before '2.jpg' etc.)
        predictions = self._classifier.predict_generator(self._test_set, verbose = 1)

        print 'test completed' 

        if self._params['save_predictions'] == True:

            # in order to store the (id,pred) pairs in increasing order of 'id', predictions need to be reordered
            image_names = self._test_set.filenames
            image_ids = []
            for i in range(len(image_names)):
                image_ids.append(int(image_names[i].split('/')[1].split('.')[0]))

            image_ids_index = sorted(range(len(image_ids)), key=lambda k: image_ids[k])
            ordered_predictions = []
            for i in range(len(image_ids_index)):
                ordered_predictions.append(predictions.T[0][image_ids_index[i]])

            ordered_ids = sorted(image_ids)

            # stores (id,predictions) pairs into a CSV file  
            results = pd.DataFrame({"id": ordered_ids, "pred" :ordered_predictions})
            results.to_csv('predictions/' + self._params['predictions_file_name'], index = False)

            print 'predictions ' + self._params['predictions_file_name'] + ' successfully saved' 

class ModelFactory:

    @staticmethod
    def get_instance(params):
        if params['net'] == 'custom':
            return CustomModel(params)
        else:
            return VGG16Model(params)


class CustomModel(AbstractModel):


    def build_model(self):

        # builds the custom network

        classifier = Sequential()

        classifier.add(Conv2D(32, (3, 3), input_shape = (self._params['img_width'], self._params['img_height'], 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        classifier.add(Flatten())
        classifier.add(Dense(units = 64, activation = 'relu'))
        classifier.add(Dropout(0.5))

        classifier.add(Dense(units = 1, activation = 'sigmoid'))

        # compiles the model
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        self._classifier = classifier

        print 'custom network model built'


class VGG16Model(AbstractModel):

    def build_model(self):

        # imports the convolutional structure and weights of the VGG16 network
        vgg16_conv_net = applications.VGG16(weights='imagenet', include_top=False, input_shape=(self._params['img_width'], self._params['img_height'], 3))

        # sets all the VGG16 layers as non-trainable 
        for layer in vgg16_conv_net.layers:
            layer.trainable = False

        # builds a fully-connected (trainable) network to put on top of the VGG16 convolutional network
        fc_net = Sequential()
        fc_net.add(Flatten(input_shape=vgg16_conv_net.output_shape[1:]))
        fc_net.add(Dense(256, activation='relu'))
        fc_net.add(Dropout(0.5))

        fc_net.add(Dense(1, activation='sigmoid'))

        # add the fully-connected network on top of the VGG16 convolutional network
        self._classifier = Model(inputs= vgg16_conv_net.input, outputs= fc_net(vgg16_conv_net.output))

        # compiles the model
        self._classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        print 'VGG16 network model built'




