import yaml
import pprint
import numpy as np

class Config:
    def __init__(self, configFilename):
        self.value = self.load(configFilename)

    def load(self, configFilename):
        with open(configFilename) as configFile:
            configDict = yaml.safe_load(configFile)
            print(f"load config...")
            pprint.pprint(configDict)
        return configDict

def diff_angle(a, b):
    """returns the CCW difference between angles a and b, i.e. the amount
    that you'd neet to rotate from b to get to a.  The result is in the
    range (-pi,pi]"""
    d = a - b
    while d <= -np.pi:
        d = d + np.pi * 2
    while d > np.pi:
        d = d - np.pi * 2
    return d




