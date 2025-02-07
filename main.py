
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

def generate_synthetic_data(num_samples=100):
    np.random.seed(42)
    data = pd.DataFrame({
        'id': range(1, num_samples + 1),
        'age': np.random.randint(20, 70, num_samples),
        'gender': np.random.choice(['Male', 'Female'], num_samples),
        'risk_score': np.random.uniform(0, 1, num_samples),
        'treatment': np.random.choice([0, 1], num_samples)
    })
    return data

def risk_set_matching(data, covariates, treatment_col):
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]


    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[covariates])

    distances, indices = nn.kneighbors(treated[covariates])
    matched_control = control.iloc[indices.flatten()]


    matched_data = pd.concat([treated, matched_control]).reset_index(drop=True)
    return matched_data


def visualize_matching(data, covariate, treatment_col):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data[data[treatment_col] == 1][covariate], label='Treated', fill=True)
    sns.kdeplot(data=data[data[treatment_col] == 0][covariate], label='Control', fill=True)
    plt.title(f'Distribution of {covariate} Before Matching')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    data = generate_synthetic_data(200)
    print("Synthetic data generated:")
    print(data.head())

    data.to_csv("synthetic_data.csv", index=False)


    matched_data = risk_set_matching(data, ['age', 'risk_score'], 'treatment')
    print("\nMatched data:")
    print(matched_data.head())

  
    matched_data.to_csv("matched_data.csv", index=False)


    visualize_matching(data, 'age', 'treatment')
