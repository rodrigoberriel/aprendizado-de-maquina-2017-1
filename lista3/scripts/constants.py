# -*- coding: utf-8 -*-
import os

SEED = 2017
MAX_FLOAT = float('inf')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
LETTERS_DIR = os.path.join(DATA_DIR, 'letras')

# Database URLs
URL_CONCRETE_DATABASE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'

# Data sets filenames
FILENAME_CONCRETE_DATABASE = 'Concrete_Data.xls'
FILENAME_KNN_DATABASE = 'knn.data'
