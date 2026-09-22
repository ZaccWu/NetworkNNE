# main.py
"""
Converted from main.m
This script orchestrates set_up -> read_data -> training -> generation -> analysis.
"""
from set_up import set_up
from read_data import read_data
from nne_train import nne_train
from nne_gen import nne_gen
from Moments import Moments
from Descriptive import Descriptive
from Clustering_global import clustering_global
from Test_error_summary import Test_error_summary

def main():
    config = set_up()
    data = read_data(config)

    # The MATLAB code may pass many items; here we assume read_data returns a dict
    # and nne_train expects (data, config)
    net, guild, track = nne_train(data, config)

    # generate using trained result if nne_gen implemented
    try:
        generated = nne_gen(net, config)
    except Exception:
        generated = None

    # analytics
    try:
        moments = Moments(net, guild)
    except Exception:
        moments = None

    try:
        desc = Descriptive(data)
    except Exception:
        desc = None


    return net, guild, track

if __name__ == '__main__':
    main()
