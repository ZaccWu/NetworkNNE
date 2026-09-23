
"""
Converted from main.m
This script orchestrates set_up -> training -> generation -> analysis.
"""
from set_up import set_up
from nne_train import nne_train
from nne_gen import nne_gen


def main():
    training_set = set_up()
    ml_data = nne_gen(training_set)
    nne_train(ml_data)


if __name__ == '__main__':
    main()
