import pandas as pd
import os

# Get the directory where train_model.py is located
base_dir = os.path.dirname(__file__)

# Correct path to the CSV
file_path = os.path.join(base_dir, "data", "air_quality_data.csv")

# Load the data
data = pd.read_csv(file_path)
