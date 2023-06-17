from modules.dna_encoder import encodeStrand
import modules.som_finder as finder

x = encodeStrand(["AaCATAGCATgcTCgTCAGAGACT", 
                  "acgcgcgcactcagcatcgactgcatcgactagcatgcatcga", 
                  "actactcgcatgcatcgctaccctagctggatcgatc",
                  "actcattcatcatcastagcatatcatcatasactac"])
print(x)
a = finder.find_model(x.values, total_rep= 2, random_state = 1)
print(a)
