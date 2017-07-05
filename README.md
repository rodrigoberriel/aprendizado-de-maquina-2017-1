## Aprendizado de Máquina - 2017/1

Professor: Dr. Patrick Marques Ciarelli

Esse repositório será usado para armazenar os códigos-fontes e relatórios das listas de exercícios da disciplina.

### Lista 1
Esse código foi testado tanto no Linux (Ubuntu 14.04) quanto no Windows 10. Para rodar os scripts dos exercícios, é necessário ter Python 2.7 ou Python 3.x e algumas dependências. Para instalar as dependências, basta rodar o comando abaixo na raiz do repositório:

    $ pip install -r lista1/requirements.txt

Para executar o script de um exercicio em específico (por exemplo, o exercício 1), basta executar:

    $ cd lista1/scripts
    $ python lista1.py --exercicio=1

Ou então, se preferir, você pode executar o script diretamente. Por exemplo:

    $ cd lista1/scripts
    $ python exercicio1.py

As bases de dados Iris, Car Evaluation e Wine serão baixadas automaticamente nas questões em que foram usadas. Todas as bases de dados estão armazenadas na pasta `data`. Ao executar os scripts, os gráficos que estão neste relatório serão exibidos e salvos na pasta `output`. O codigo-fonte desse relatório (em LaTeX) também se encontra na pasta `report`. Para garantir a reprodutibilidade dos resultados e gráficos, foi usado como seed o valor `2017` definido no arquivo `lista1/scripts/constants.py`.
