from modules.dna_clustering import best_som_fit
from modules.dna_encoder import encodeStrand


data = ["AaCATAGCATgcTCgTCAGAGACT", 
        "acgcgcgcactcagcatcgactgcatcgactagcatgcatcgatagcgtacgt", 
        "actactcgcatgcatcgctaccctagctggatcgatyhccatgacgtacgtc",
        "actcattcatcatcastagcatatcatcatasactatacaaaacagtgactgacdgtc",
        "actactatatactagatcgtactgcatgcatgcatgcagtcagtactgcatg",
        "aaacacacaacaaaaaaaaaahthcathcathcathctahctatcthachtacth",
        "aaaaaaaaacacacataccgactgagctagtgtaaaaaaaaaattgctagtggctatc"]
best_model = best_som_fit(X=data,feature_selection=3)
print("result of encoding:")
print(encodeStrand(data))
print(best_model.model)
print("Model with matrix", best_model.model.m, "by", best_model.model.n)
print("Model dimension:", best_model.model.dim)
print("Model features:", best_model.params_)
print("Silhouette score:", best_model.score)
print("Feature Correlation Matrix")
print(best_model.param_corr)
print("Prediciton of the data:")
print(best_model.predict(data))
print("Input clustering matrix")
print(best_model.input_matrix(data))