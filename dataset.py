import configparser
import pymysql
import pymysql.cursors
import fasttext

from nltk.tokenize import RegexpTokenizer
import numpy as np


config = configparser.ConfigParser()
config.read('application.cfg')

mysql_config = config['mysql_local']

conn = pymysql.connect(host=mysql_config['host'],
                       user=mysql_config['user'],
                       password=mysql_config['password'],
                       db=mysql_config['db'],
                       charset=mysql_config['charset'],
                       cursorclass=pymysql.cursors.DictCursor)

tokenizer = RegexpTokenizer(r'\w+')

embedded_words = fasttext.load_model('resources/wiki.pt/wiki.pt.bin')

padding_array = np.zeros(300).tolist()


def get_label(label: str) -> list:
    from_to = {'NG': [1, 0, 0], 'NE': [0, 1, 0], 'PO': [0, 0, 1]}
    return from_to[label]


def get_max_paragraph_length() -> int:

    max_len_pargraph = 0

    cursor = conn.cursor()
    cursor.execute("""select paragrafo
                      from noticias.noticias_x_paragrafo ncp
                      join noticias.noticias n
                      on n.id_noticia = ncp.id_noticia
                      where ncp.polaridade in ('NG','NE','PO')
                      and n.ind_corpus = 'S'
                    """)

    for row in cursor:

        len_paragraph = len(tokenizer.tokenize(row['paragrafo']))

        if len_paragraph > max_len_pargraph:
            max_len_pargraph = len_paragraph

    return max_len_pargraph


def get_fold_sample(folds: list) -> (list, list):

    cursor = conn.cursor()
    query = """select paragrafo, polaridade, fold, entidade, id_perfil
               from noticias.noticias_x_paragrafo ncp
               join noticias.noticias n
               on n.id_noticia = ncp.id_noticia
               where ncp.polaridade in ('NG','NE','PO')
               and n.ind_corpus = 'S'
               and ncp.fold IN (%s)
                """
    in_clause = ', '.join(list(map(lambda x: '%s', folds)))
    query = query % in_clause
    cursor.execute(query, folds)

    paragraphs = []
    labels = []

    for row in cursor:
        paragraph = tokenizer.tokenize(row['paragrafo'])
        labels.append(get_label(row['polaridade']))
        paragraphs.append(paragraph)

    return (paragraphs, labels)


def format_matrix(paragraphs: list, paragraph_length: int) -> np.ndarray:

    padded_paragraphs = []

    for (i, paragraph) in enumerate(paragraphs):

        len_paragraph = len(paragraph)
        padded_paragraph = []

        for j in range(0, paragraph_length):

            if j < len_paragraph:
                padded_paragraph.append(embedded_words[paragraph[j]])
            else:
                padded_paragraph.append(padding_array)

        padded_paragraphs.append(padded_paragraph)

    return np.array(padded_paragraphs)
