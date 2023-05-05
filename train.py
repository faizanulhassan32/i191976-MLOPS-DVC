import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv("Dataset.csv")
df.dropna(inplace=True)
# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
# Define the features and target variables
features = ['Open', 'High', 'Low', 'Volume_(BTC)', 'Volume_(Currency)']
target = 'Weighted_Price'
# Train the decision tree regressor
tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(train_data[features], train_data[target])
# Evaluate the model on the testing set
y_pred = tree.predict(test_data[features])
mse = mean_squared_error(test_data[target], y_pred)
mae = mean_absolute_error(test_data[target], y_pred)
# Save the metrics to a JSON file using DVC
metrics = {'mean_squared_error': mse, 'mean_absolute_error': mae}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)