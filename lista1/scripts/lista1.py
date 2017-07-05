import argparse
import importlib


def lista1(exercicio):
    exercicio_str = 'exercicio{}'.format(exercicio)
    getattr(importlib.import_module(exercicio_str), exercicio_str)()
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lista 1')
    parser.add_argument('-e', '--exercicio', type=int, help='Numero do exercicio da lista',
                        required=True, choices=[1, 2, 3, 5, 6, 8, 9, 10])
    args = parser.parse_args()
    lista1(args.exercicio)
