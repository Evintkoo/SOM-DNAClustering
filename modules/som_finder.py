import numpy as np
import modules.som as som
from sklearn.metrics import silhouette_score 
import modules.dna_encoder
import modules.quadatic_regression as quadres
    
# factorize is find numbers that could divide the input
def factorize(num: int) -> list():
    return [i for i in range(1,int((num+1)**0.5)+1) if num%i == 0]

# pick the best SOM model with the size of matrix_size
def best_matrix_size(X:list(), matrix_size: int, max_iter = 3000 , epoch = 1, learning_rate = 1, sigma = 1):
    factors = factorize(matrix_size)
    models = list()
    silhouette_scores = list()
    for num in factors:
        clustering_model = som.SOM(m=num, n=int(matrix_size/num), dim=X.shape[1], max_iter=max_iter, lr = learning_rate, sigma=sigma)
        clustering_model.fit(X, epochs= epoch)
        predictions = clustering_model.predict(X)
        models.append(clustering_model)
        silhouette_scores.append(silhouette_score(X, predictions))
    return models[np.array(silhouette_scores).argmax()], max(silhouette_scores)

def som_peak_test(X, som_hist, som_scores, max_iter=3000, epoch = 1, learning_rate = 1, sigma = 1):
    X_quadreg = [(model.m * model.n) for model in som_hist]
    y_quadreg = som_scores
    quad_model = quadres.quadratic_regression(X_quadreg,y_quadreg)
    best_size = int(quad_model.peak_point()[0])
    models_hist = list()
    models_silhouetterScore = list()
    if best_size-2 <= 1: 
        min_size = best_size
        if best_size < 2:
            min_size = 2
    else:
        min_size = best_size-2
    if min_size + 3 > X.shape[1]-1:
        max_size = X.shape[1]-1
    else: 
        max_size = min_size + 3
    for matrix_size in range(min_size,max_size):
        model, shs = best_matrix_size(X, matrix_size, max_iter=max_iter, epoch= epoch, learning_rate = learning_rate, sigma = sigma)
        models_hist.append(model)
        models_silhouetterScore.append(shs)
    return models_hist[np.array(models_silhouetterScore).argmax()], max(models_silhouetterScore)

def test_som(X, array_n_cluster, max_iter=3000, epoch = 1, learning_rate = 1, sigma = 1):
    models_hist = list()
    models_silhouetterScore = list()
    for n_cluster in array_n_cluster:
        model, silhouette_score = best_matrix_size(X, n_cluster, max_iter=max_iter, epoch= epoch, learning_rate = learning_rate, sigma = sigma)
        predictions = model.predict(X)
        models_hist.append(model)
        models_silhouetterScore.append(silhouette_score)
        
    # perform quadratic regression based on matrix size and silhouette score
    best_model, shs = som_peak_test(X, models_hist, models_silhouetterScore, max_iter=max_iter, epoch= epoch, learning_rate = learning_rate, sigma = sigma)
    return best_model, models_silhouetterScore
    # return models_hist[np.array(models_silhouetterScore).argmax()], max(models_silhouetterScore)

def find_model(X: np.ndarray, total_rep = 5, random_state = 5, max_iter = 3000, epoch = 1, learning_rate = 1, sigma = 1):
    """
    find_model() is a function to find the best size of matrix within range start from random_state value until random_state*total_rep
    
    Example:
    find_model(X, total_rep = 5, random_state = 5) would try matrix size of 5, 10, 15, 20, and 25

    Args:
        X (np.ndarray): 
            Training data. Must have shape (n, m) where n is the number
            of training samples, and m is the number of the features.
        total_rep (int, optional): 
            Maximum iteration of the matrix size. 
            Defaults to 5.
        random_state (int, optional): 
            Jumping value of the matrix size. 
            Defaults to 5.
        max_iter (int, optional): 
            Optional parameter to stop training if you reach this many interation. 
            Defaults to 3000.
        epoch (int, optional): 
            The number of times to loop through the training data when fitting. 
            Defaults to 1.
        learning_rate (int, optional): 
            The initial step size for updating the SOM weights. 
            Defaults to 1.
        sigma (int, optional): 
            Optional parameter for magnitude of change to each weight. Does not
            update over training (as does learning rate).
            Defaults to 1.

    Raises:
        ValueError: error if the total sample is less than total_reps*random_state.

    Returns:
        <modules.som.SOM object>: the highest silhouette score among all of the matrix size.
    """
    if X.shape[0] < total_rep*random_state:
        raise ValueError("The sample of the data not enough to iterates, minimum number of data is total_rep * random_state")
    else:
        # generates the list of matrix size
        if random_state == 1:
            array_n_cluster = [random_state*i for i in range(2,total_rep+2)]
        else :
            array_n_cluster = [random_state*i for i in range(1,total_rep+2)]
    
    models, shs = test_som(X, array_n_cluster, max_iter=max_iter, epoch = epoch, learning_rate = learning_rate, sigma=sigma)
    return models, shs


