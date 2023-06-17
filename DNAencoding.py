import numpy as np
import sklearn_som.som as SOM
import math

# Count k-mers is an encoding method that counts total number of its substring that appear in the string with length of K
def count_kmers(sequence:str, k_size:int) -> dict():
    # initialize empty dictionary
    data = {}
    size = len(sequence)
    
    for i in range(size - k_size + 1):
        kmer = sequence[i: i + k_size]
        try:
            data[kmer] += 1
        except KeyError:
            data[kmer] = 1
    return data

# dnaStrandEncoding is encode the list of DNA strand -> list(string) return the substrands and matrix of encoded DNA 
def dnaStrandEncoding(dnaStrandList: list()) -> list():
    #ensure that all of the dna that inserted are unique
    dnaStrandList = np.unique(dnaStrandList)
    
    #calculate the average length of the DNA strands
    average_dna_length = sum([len(strands) for strands in dnaStrandList])/len(dnaStrandList)
    
    # calculate the K value for the Counting K-mers
    substrand_length = int(math.log(average_dna_length,4))
    
    # convert each DNA strand into dictionary
    encoded_dna = [count_kmers(strands,substrand_length) for strands in dnaStrandList]
    
    # initialize the empty dictionary as a holder
    substrand_keys = {}
    
    # merge the all of the key from all of the DNA list
    for dicts in encoded_dna:
        substrand_keys.update(dicts)
    substrand_keys = list(substrand_keys.keys())
    
    # initialize the parameter of the list
    parameters = substrand_keys
    
    #initialize empty list of DNA that would be encoded
    encoded_dna_list = list()
    for dicts in encoded_dna:
        dict_to_List = list()
        
        # try to append the value that has the same substrand, if substrand cannot be found, append 0
        for keys in substrand_keys:
            try:
                dict_to_List.append(dicts[keys])
            except:
                dict_to_List.append(0)
        encoded_dna_list.append(dict_to_List)
    
    return parameters, encoded_dna_list