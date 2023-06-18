from modules.dna_encoder import encodeStrand
import modules.som_finder as finder
import modules.feature_analysis as features
import numpy as np

class selfOrganizingMaps():
    """
    class of the model
    """
    def __init__(self, model, params, shs, data_correlation) -> None:
        self.model = model
        self.params_ = params
        self.score = shs
        self.param_corr = data_correlation
        
    def predict(self, X):
        encoded_strands = encodeStrand(X)
        X_predict = encoded_strands[self.params_]
        model = self.model
        pred = model.predict(X_predict.values)
        return pred

def best_som_fit(X: np.ndarray, feature_selection=None, corelation_treshold = 0.0, total_rep = 5, random_state = 5, max_iter = 3000, epoch = 1, learning_rate = 1, sigma = 1):
    """
    som_dna_clustering is a DNA clustering method which clusters the DNA strands using Self Orgnaizing Maps

    Args:
        X (np.ndarray): 
            Training data. Must have shape (n, self.dim) where n is the number
            of training samples.
        feature_selection (None or int,optional):
            Number of feature pair that would be selected in the correlation matrix of data. 
            Default to None, means that this feature will not be used.
        corelation_treshold (float, optional): 
            Minimum pair feature correlation value. 
            Defaults to 0.0.
        total_rep (int, optional): 
            Maximum iteration of the matrix size. 
            Defaults to 5.
        random_state (int, optional): 
            Jumping value of the matrix size. 
            Defaults to 5.
        max_iter (int, optional): 
            Optional parameter to stop training if you reach this many interation while training the SOM.  
            Defaults to 3000.
        epoch (int, optional): 
            The number of times to loop through the training data when fitting the SOM. 
            Defaults to 1.
        learning_rate (int, optional): 
            The initial step size for updating the SOM weights. 
            Defaults to 1.
        sigma (int, optional):
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate).
            Defaults to 1.

    Returns:
        <selfOrganizingMaps object>: an object with subclass of :
            .model: <modules.som.SOM object>
                SOM class, the best performance among all of the trials
            .params_ : list of substrand 
                features (substrand) that used in the clustering
            .score : silhouette score of the model
            .param_corr: correlation matrix
                the correlation matrix of the model's parameters
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
    