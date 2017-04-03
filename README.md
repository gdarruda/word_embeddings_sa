# Sentiment Analysis for portuguese

A deep learning sentiment analysis for portuguese using Python 3.5 + Keras 2.0 + Tensorflow 1.0 + Fasttext pre-trained vectors

## Instructions

This is not a "production-ready" solution, just [an experiment](https://gdarruda.github.io/2017/deep-leraning-sentiment-analysis/) using an corpus of annotated news.

## MySQL

The corpus is stored in a MySQL database, it's not publicy avaliabe on this format, but I can send the dump of this dataset. Just sent me an e-mail (gda.gabriel@gmail.com).

The database connections are set on an a file named `application.cfg` with this layout:

~~~python
[mysql_local]
user: # Database user
password: # Password user
host: # hostname
db: # database "noticias"
charset: # "charset "utf8" by default
~~~

## Fasttext pre-trained vectors

The pre-trained vectors used in the experiment are [publicy avaliable](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md). You can download portuguese version and put the `wiki.pt.bin` and `wiki.pt.vec` under `resources\wiki.pt` folder.
