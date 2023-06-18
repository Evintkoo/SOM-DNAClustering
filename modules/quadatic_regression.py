import numpy as np

class quadatic_function():
    def __init__(self, a, b, c) -> None:
        self.a = a
        self.b = b 
        self.c = c
    
    def predict(self, X):
        return self.a*X*X + self.b*X + self.c
    
    def peak_point(self):
        return (-self.b/(self.a*2), (self.b*self.b - 4*self.a*self.c)/(4*self.a))
    
def quadratic_regression(X: list(), y:list()):
    coefficients = np.polyfit(X, y, 2)
    model = quadatic_function(coefficients[0], coefficients[1], coefficients[2])
    return model