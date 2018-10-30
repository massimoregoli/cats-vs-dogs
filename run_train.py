from modules.models import AbstractModel
import yaml
import sys
import argparse



def main(argv=None):

    # parses arguments
    parser = argparse.ArgumentParser(description='Trains a model for dogs vs cats classification')
    parser.add_argument('--from', dest='cfg_file_name', action='store')
    args = parser.parse_args()

    # loads configuration file
    with open('cfgs/' + args.cfg_file_name,'r') as yml_file:
        params = yaml.load(yml_file)

    # instantiates the desired model class according to the 'net' parameter set in the yaml configuration file:
    #   - net: custom -> uses the custom network
    #   - net: vgg16 -> uses the VGG16-based network
    model = AbstractModel.get_instance(params)

    # builds the desired network, layer by layer
    model.build_model()

    # configures the source and type of training and validation data
    model.prepare_train_data()

    # trains the network to fit the data
    model.fit_data()


if __name__ == "__main__":
    main(sys.argv)
