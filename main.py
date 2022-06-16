import os
import argparse
import yaml

from solver import Solver
from data_pipeline import DataPipeline
from Model import BreedClassifier
from torch.backends import cudnn


def make_train_directory(config):
    # Create directories if not exist.
    if not os.path.exists(config['TRAINING']['TRAIN_DIR']):
        os.makedirs(config['TRAINING']['TRAIN_DIR'])
    if not os.path.exists(os.path.join(config['TRAINING']['MODEL_DIR'])):
        os.makedirs(os.path.join(config['TRAINING']['MODEL_DIR']))


def main(config):
    assert config['TRAINING']['MODE'] in ['train', 'test']

    cudnn.benchmark = True
    # Define train and valid data
    train_loader, valid_loader, _, _ = DataPipeline(config).get_data_loader()
    # Define Model
    model = BreedClassifier(config)
    # Start training
    solver = Solver(config, train_loader, valid_loader, model)
    print('{} is started'.format(config['TRAINING']['MODE']))
    if config['TRAINING']['MODE'] == 'train':
        solver.train()
    elif config['TRAINING']['MODE'] == 'test':
        pass
    print('{} is finished'.format(config['TRAINING']['MODE']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='specifies config yaml file')

    params = parser.parse_args()

    if os.path.exists(params.config):
        config = yaml.load(open(params.config, 'r'), Loader=yaml.FullLoader)
        make_train_directory(config)
        main(config)
    else:
        print("Please check your config yaml file")

