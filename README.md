# Instalation
```
git clone https://github.com/Evintkoo/SOM-DNAClustering.git
```

A library to cluster DNA strands with Self Organizing Map clustering.

### How to use

Using som-dnaclustering is easy and simple, which could be folowed by:
1. Import the som-dnaclustering modules
```python
from modules.dna_clustering import best_som_fit
```

2. Now use your DNA strands list as your train data
```python
data = ["AaCATAGCATgcTCgTCAGAGACT", 
        "acgcgcgcactcagcatcgactgcatcgactagcatgcatcga", 
        "actactcgcatgcatcgctaccctagctggatcgatc",
        "actcattcatcatcastagcatatcatcatasactac"]
best_model = best_som_fit(X=data)
```

Note: this library have already implemented data cleaning, so you do not need to worry about your data, and you could use this library's features such as correlation analysis by using feature_selection = None or an integer