# Challenge | Manuel Valentim Loureiro

## Instructions

Clone this repository, create a new virtual environment and run:

```console
$ pip install -r requirements.txt
```

To generate `output/results.csv` run:

```console
$ python predict.py
```

Alternatively run jupyter notebook `Challenge.ipynb`, where a summary of metrics
is also provided.

## Summary

This repository contains a cvssV3 classifier using cvssV2 data. It is built on
top of `lightgbm` and there are features extracted using tfidf.

__Note:__ Due to a limit in processing capacity the tfidf features were limited
to 1000. However, it is possible to increase this number to improve results.