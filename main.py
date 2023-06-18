from modules.dna_clustering import best_som_fit


data = ["AaCATAGCATgcTCgTCAGAGACT", 
        "acgcgcgcactcagcatcgactgcatcgactagcatgcatcga", 
        "actactcgcatgcatcgctaccctagctggatcgatc",
        "actcattcatcatcastagcatatcatcatasactac"]
best_model = best_som_fit(X=data,feature_selection=3)
print(best_model.model)
print("Model with matrix", best_model.model.m, "by", best_model.model.n)
print("Model dimension:", best_model.model.dim)
print("Model features:", best_model.params_)
print("Silhouette score:", best_model.score)
print("Feature Correlation Matrix")
print(best_model.param_corr)