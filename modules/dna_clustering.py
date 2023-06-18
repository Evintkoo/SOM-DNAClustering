from modules.dna_encoder import encodeStrand
import modules.som_finder as finder
import modules.feature_analysis as features
import numpy as np

class selfOrganizingMaps():
    def __init__(self, model, params, shs, data_correlation) -> None:
        self.model = model
        self.params_ = params
        self.score = shs
        self.param_corr = data_correlation

def best_som_fit(X: np.ndarray, feature_selection=None, corelation_treshold = 0.0, total_rep = 5, random_state = 5, max_iter = 3000, epoch = 1, learning_rate = 1, sigma = 1):
    """
    som_dna_clustering is a DNA clustering method which clusters the DNA strands using Self Orgnaizing Maps

    Args:
        X (np.ndarray): _description_
        corelation_treshold (float, optional): _description_. Defaults to 0.0.
        total_rep (int, optional): _description_. Defaults to 5.
        random_state (int, optional): _description_. Defaults to 5.
        max_iter (int, optional): _description_. Defaults to 3000.
        epoch (int, optional): _description_. Defaults to 1.
        learning_rate (int, optional): _description_. Defaults to 1.
        sigma (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    
    # encode the strands of the DNA
    encoded_strands = encodeStrand(X)
    
    if not feature_selection:
        X_train = encoded_strands
        print("no corr analysis")
    else :
        # reduce the dimension of the data by using correlation analysis
        reducted_dimension = features.dim_reduction(encoded_strands, corelation_treshold, total_features=feature_selection)
        X_train = reducted_dimension
    
    # find the best model of the Self Organizing Maps
    model, shs = finder.find_model(X_train.values, total_rep= 2, random_state = 1)
    return selfOrganizingMaps(model = model, params= X_train.columns, shs = shs, data_correlation = X_train.corr())
    