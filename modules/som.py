import numpy as np
import sklearn_som.som as SOM #will built by myself
from sklearn.metrics import silhouette_score #will built by myself

# factorize is find numbers that could divide the input
def factorize(num: int) -> list():
    return [i for i in range(1,int((num+1)**0.5)+1) if num%i == 0]

# pick the best SOM model with the size of matrix_size
def best_matrix_size(X:list(), matrix_size: int, max_iter = 3000 , epoch = 1, learning_rate = 1):
    factors = factorize(matrix_size)
    models = list()
    silhouette_scores = list()
    for num in factors:
        clustering_model = SOM(m=num, n=int(matrix_size/num), dim=X.shape[1], max_iter=max_iter, lr = learning_rate)
        clustering_model.fit(X, epochs= epoch)
        predictions = clustering_model.predict(X)
        models.append(clustering_model)
        silhouette_scores.append(silhouette_score(X, predictions))
    return models[np.array(silhouette_scores).argmax()], max(silhouette_scores)

def test_som(X, array_n_cluster, max_iter=3000, epoch = 1, lr = 1):
    models_hist = list()
    models_silhouetterScore = list()
    hist = list()
    for n_cluster in array_n_cluster:
        model, silhouette_score = best_matrix_size(X, n_cluster, max_iter=max_iter, epoch= epoch, lr = lr)
        predictions = model.predict(X)
        models_hist.append(model)
        models_silhouetterScore.append(silhouette_score)
        cluster_center = model.cluster_centers_.shape[0]*model.cluster_centers_.shape[1]
        hist.append([model.m, model.n, cluster_center, model.dim, silhouette_score])
    return models_hist, models_silhouetterScore

def find_SOM(X, random_state = 5, total_rep = 10, max_iter = 3000, epoch = 1, lr = 1):
    if random_state == 1:
        array_n_cluster = [random_state*i for i in range(2,total_rep+2)]
    else :
        array_n_cluster = [random_state*i for i in range(1,total_rep+2)]
    
    models, shs = test_som(X, array_n_cluster, max_iter=max_iter, epoch = epoch, lr = lr)
    return models[np.array(shs).argmax()]