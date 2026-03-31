import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import pickle

# Load configuration
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Load data
df = pd.read_csv(config["data"]["path"])
X = df.drop(columns=['sales'])
y = df['sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["training"]["test_size"],
    random_state=config["training"]["random_state"]
)
