# -*- coding: utf-8 -*-
import os

SEED = 2017

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Database URLs
URL_IRIS_DATABASE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
URL_CAR_DATABASE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
URL_WINE_DATABASE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

# Data sets filenames
FILENAME_IRIS_DATABASE = 'iris.data'
FILENAME_CAR_DATABASE = 'car.data'
FILENAME_WINE_DATABASE = 'wine.data'
FILENAME_RUNNER_DATABASE = 'Runner_num.txt'
FILENAME_POLINOMIO_DATABASE = 'Polinomio.txt'
FILENAME_CNAE_DATABASE = 'CNAE-9_reduzido.txt'
FILENAME_NEBULOSA_TRAIN_DATABASE = 'nebulosa_train.txt'
FILENAME_NEBULOSA_TEST_DATABASE = 'nebulosa_test.txt'
