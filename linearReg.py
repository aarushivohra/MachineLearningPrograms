import pandas as pd
import numpy as np

def linearRegression(x, y):

    n = len(x)
    xy = np.multiply(x, y)
    x_square = x*x

    # b = Sxy/Sxx
    # Sxy = sum(xy) - (sum(x)*sum(y))/n
    # Sxx = sum(x_square) - (sum(x)*sum(x))/n

    sum_xy = np.sum(xy)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x_square = np.sum(x_square)

    Sxy = sum_xy - (sum_x * sum_y) / n
    Sxx = sum_x_square - (sum_x * sum_x) / n

    b = Sxy / Sxx 
    coeff = []
    coeff.append(b)

    # a = y_mean - b * x_mean
    y_mean = np.mean(y)
    x_mean = np.mean(x)

    a = y_mean - b * x_mean
    coeff.append(a)
    return coeff


def main():
    df = pd.read_csv('LR.csv')
    x = df['Income'].to_numpy()
    y  = df['Food Expenditure'].to_numpy()
    coeff = linearRegression(x, y)

    print("a =", coeff[1])
    print("b =", coeff[0])


if __name__ == "__main__":
    main()