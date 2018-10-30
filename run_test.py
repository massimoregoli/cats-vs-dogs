from modules.models import AbstractModel
import yaml
import sys
import argparse



def main(argv=None):

    # parses arguments
    parser = argparse.ArgumentParser(description='Loads a model trained for cats vs dogs classification and tests it on a given dataset')
    parser.add_argument('--from', dest='cfg_file_name', action='store')
    args = parser.parse_args()

    # loads configuration file
    with open('cfgs/' + args.cfg_file_name,'r') as yml_file:
        params = yaml.load(yml_file)

    # instantiates the desired model class according to the 'net' parameter set in the yaml configuration file:
    #   - net: custom -> uses the custom network
    #   - net: vgg16 -> uses the VGG16-based network
    model = AbstractModel.get_instance(params)

    # loads a pretrained model from file
    model.load_model()
    
    # configures the source and type of test data
    model.prepare_test_data()

    # test the model on the specified path
    model.test_model()


if __name__ == "__main__":
    main(sys.argv)
