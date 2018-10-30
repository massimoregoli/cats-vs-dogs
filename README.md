# Description

This framework allows to train and test neural networks on the Kaggle dataset (link here) for the cats vs dogs classification task. Two different networks are available:
* A custom-made network composed of three convolutional-pooling layers and a fully-connected layer
* A network composed of the convolutional part of the VGG16 network (pretrained and with freezed weights) and a fully-connected layer placed on top of it 

# Project overview

The project includes the following files/folders:

* `run_train.py` and `run_test.py` are used to run, respectively, the training and testing process on the dataset (further details below), while `modules/` contains their dependencies and `cfgs/` the related configuration files
* `scripts/prepare_dataset.py` takes the Kaggle dataset folder as input and copy/reorganize data into a different folder tree in order to be processed during training
* as output of the training, `models/` is meant to store the best model obtained during the training process, while `logs/` stores the TensorBoard log. In addiction, they currently contain, for both networks, the best network models and logs I have obtained so far.
* `predictions/` stores, after test is completed, the CSV file containing all the (id,prediction) pairs. Morover, It currently contains the predictions obtained with the best network models trained so far (contained in `models/`)
* `plots/` contains a few plots related to the best results obtained using both networks

# How to run

This is a Keras based project, so it is required to successfully run the training and testing scripts. Specifically, version 2.2.4 has been used for the experiments.

## Preparing the dataset

In order to prepare the dataset the following steps need to be performed:
* download the Kaggle dataset (here)
* run `python scripts/prepare_dataset.py --in_dir <in_dir> --out_dir <out_dir> --perc_train <perc_train>`, where:
  * `<in_dir>` is the folder where you have stored the Kaggle dataset (it should contain the `train` and `test` folders)
  * `<out_dir>` is the folder where you want the data to be copied (and reorganized). If it does not exist, it will be created.
  * `<perc_train>` is the percentage of Kaggle training images that will be maintained as training data in our dataset, while the other training images will be used as validation set (in my case it has been set to 0.88, so 22000 images are used for training and 3000 for validation)
After copying is done, you will get the following folder tree (which reflects the way Keras expects data):
  * `<out_dir>/train/cats/`: 12500 * `<perc_train>` images
  * `<out_dir>/train/dogs/`: 12500 * `<perc_train>` images
  * `<out_dir>/val/cats/`: 12500 * (1 - `<perc_train>`) images
  * `<out_dir>/val/dogs/`12500 * (1 - `<perc_train>`) images
  * `<out_dir>/test/cats_and_dogs/`: 12500 images

## Setting the configuration file

Before running the scripts it is important to properly set the dataset path in the configuration file. Two examples of configuration files are contained in the `cfgs/` folder, one for each network. All settings are intuitive and can be left as they are, just make sure that`dataset_dir` is set to the folder you specified in the `--out_dir` parameter of the `prepare_dataset.py` script.

## Run training

To run the training process launch `python run_train.py --from <cfg_file>`, where `<cfg_file>` is the config file you are using, specifically:
* use `params_custom.yml` (`cfgs/` should NOT be included) to run the training using the custom network
* use `params_vgg16.yml` to run the training using the vgg16-based network
During the training progress TensorBoard logs the training status into the `logs/` folder, while at the end of each epoch the current trained model is stored into the `models/` folder if improved with respect to the previous epoch.

## Run testing

To run the testing process launch `python run_test.py --from <cfg_file>`, where, again, `<cfg_file>` is the config file you are using. As they share most of parameters, both training and testing use the same configuration file.
At the end of the testing process, in the `predictions/` folder will be stored the predictions on the test set in a CSV file according to the Kaggle submission format.

# Results

In my experiments I obtained the best results int the following settings:
* Using the custom-made network I achieved a validation accuracy of 90.8% after 50 epochs and: batch_size = 32, optimizer = 'adam', input images resized to 128x128 and learning rate = 0.001.
* Using the VGG16-based network I achieved a validation accuracy of 95.0% after 45 epochs and: batch_size = 32, optimizer = 'adam', input images resized to 224x224 and learning rate = 0.001.
Structural hyperparameters (number of layers, nodes, activation functions) can be explored having a look at the classes `CustomModel` and `VGG16Model` defined in `modules/models.py`.
Furthermore, in both cases I adopted the data-augmentation strategy, which consisted in image zooming, shearing and horizontal flip.
