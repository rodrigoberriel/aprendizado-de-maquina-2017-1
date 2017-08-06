import sys
import argparse
import importlib


choices = {
    1: [1, 2, 3, 5, 6, 8, 9, 10],
    2: [1, 2, 5, 6, 7],
    3: [1, 2, 3],
}


def main(lista, exercicio):
    sys.path.append('lista{}/scripts'.format(lista))
    exercicio_str = 'exercicio{}'.format(exercicio)
    getattr(importlib.import_module(exercicio_str), exercicio_str)()
    exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lista 2')
    parser.add_argument('-l', '--lista', type=int, help='Numero da lista', required=True, choices=list(choices.keys()))
    parser.add_argument('-e', '--exercicio', type=int, help='Numero do exercicio da lista', required=True, choices=set(choices[1]+choices[2]))
    args = parser.parse_args()

    if args.exercicio not in choices[args.lista]:
        parser.error('Para a lista {}, apenas os seguintes exercicios podem ser escolhidos: {}'.format(
            str(args.lista), ', '.join(map(str, choices[args.lista]))
        ))

    main(args.lista, args.exercicio)
