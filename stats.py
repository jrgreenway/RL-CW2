import matplotlib.pyplot as plt
import os
import json
import numpy as np

files = os.listdir("data/")    

stats = {}

for file in files:
    with open("data/"+file, "r") as f:
        data = json.load(f)
    data = data["data"]["scores"]
    stat = max(data)
    stats[file+"_max"] = stat

print(stats)
    