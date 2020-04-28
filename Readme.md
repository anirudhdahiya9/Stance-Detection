# Stance Detection
Exploratory project towards stance detection on the Fake News Challenge Dataset. Details of the task  and dataset at [fakenewschallenge.org](http://fakenewschallenge.org/).

`train.py` implements the preprocessing, training and evaluation for training a pretrained Roberta model on the Stance Detection dataset. Enter `python train.py -h` to see all configurable options.


`train_hier.py` implements the same for a hierarchical prediction model, as inspired from [Zhang et al. 2019](https://dl.acm.org/doi/10.1145/3308558.3313724)


`model.py` defines the hierarchical model built up from Roberta representations.

