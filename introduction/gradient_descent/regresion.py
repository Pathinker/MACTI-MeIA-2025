import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

LEARNING_RATE = 0.01
EPOCHS = 100

IRIS = load_iris(as_frame=True)
DATA_FRAME = IRIS.frame
DATA_FRAME.columns = [col.lower().replace(" (cm)", "").replace(" ", "_") for col in DATA_FRAME.columns]

X = DATA_FRAME[['sepal_width', 'petal_width', 'sepal_length']]
Y = DATA_FRAME['petal_length']

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2, random_state=42)

class linear_regresion:

    def __init__(self, X, Y, EPOCHS, LEARNING_RATE):

        self.X = X
        self.Y = Y
        self.X_COLUMNS = len(X.columns)
        self.REGISTERS = len(X)

        self.EPOCHS = EPOCHS
        self.LEARNING_RATE = LEARNING_RATE

        self.slope = np.zeros(self.X_COLUMNS, dtype=np.float64)
        self.bias = np.float64(0)
        self.error = np.zeros(self.EPOCHS, dtype=np.float64)

    def predict(self, x):

        predicted_values = np.zeros(self.REGISTERS, dtype=np.float64)

        for i in range(self.REGISTERS):
            for j in range(self.slope.size):
                predicted_values[i] += self.slope[j] * x.iloc[i, j]
            predicted_values[i] += self.bias

        return predicted_values
 
    def gradient_descent(self, x, predict, y):

        updated_slope = np.zeros(self.X_COLUMNS, dtype=np.float64)
        update_bias = 0.0

        for i in range(self.REGISTERS):
            error = predict[i] - y.iloc[i]
            for j in range(self.X_COLUMNS):
                updated_slope[j] += error * x.iloc[i, j]
            update_bias += error

        for j in range(self.X_COLUMNS):
            self.slope[j] -= self.LEARNING_RATE * updated_slope[j] / self.REGISTERS
        self.bias -= self.LEARNING_RATE * update_bias / self.REGISTERS
    
    def fitness_function(self, predicted_values, expected_values):

        MINIMUM_SQUARE_ERROR = 0.0

        for i in range(len(expected_values)):
            MINIMUM_SQUARE_ERROR += (expected_values.iloc[i] - predicted_values[i]) ** 2
        
        return MINIMUM_SQUARE_ERROR / self.REGISTERS

    def fit(self):
        
        for i in range(EPOCHS):

            predicted_values = self.predict(self.X)
            self.gradient_descent(self.X, predicted_values, self.Y)
            self.error[i] = self.fitness_function(predicted_values, self.Y)
        
        return self.error
        
if __name__ == "__main__":

    train_iris_estimator = linear_regresion(X_TRAIN, Y_TRAIN, EPOCHS, LEARNING_RATE)
    train_error = train_iris_estimator.fit()

    validation_iris_estimator = linear_regresion(X_TEST, Y_TEST, EPOCHS, LEARNING_RATE)
    validation_error = validation_iris_estimator.fit()

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_TRAIN, Y_TRAIN)
    prediction_sklearn = sklearn_model.predict(X_TRAIN)

    print("\nCUSTOM MODEL PARAMETERS")
    print(f"Slopes: {train_iris_estimator.slope}")
    print(f"Bias (intercept): {train_iris_estimator.bias}")
    print(f"Final MSE: {train_error[-1]:.6f}")

    print("\nSKLEARN MODEL PARAMETERS")
    print(f"Slopes: {sklearn_model.coef_}")
    print(f"Bias: {sklearn_model.intercept_}")
    print(f"Final MSE: {np.mean((Y_TRAIN - prediction_sklearn)**2):.6f}")
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(EPOCHS), train_error, label="Train", color='BLUE')
    plt.plot(range(EPOCHS), validation_error, label="Validation", color='RED')
    plt.hlines(np.mean((Y_TRAIN - prediction_sklearn)**2), xmin=0, xmax=EPOCHS, colors='red', linestyles='--', label='Error MSE sklearn')
    plt.title("Fitness Function Minimum Square Error")
    plt.xlabel("Época")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()