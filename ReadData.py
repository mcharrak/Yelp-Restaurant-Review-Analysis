import gzip
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import datetime
import folium
from folium import plugins

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)


def readData(fileName="dataset/reviewShuffled.json",breakCondition=5000000):
    data=[]
    count=0
    with open(fileName,"r")as f:
        for line in f:
            line=json.loads(line)
            data.append(line)
            if(count==breakCondition):
                break
            count+=1
    print("Data Loaded")
    print("Total Data: ",count)
    return data
