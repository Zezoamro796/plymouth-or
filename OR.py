import pandas as pd  # Import the pandas library for data manipulation
import pulp  # Import the pulp library for linear programming

# Load the dataset from the CSV file
df = pd.read_csv('production_data.csv')

# Print the first few rows of the dataset to understand its structure
print("Dataset head:")
print(df.head())

# Convert relevant columns to lists for processing
resources = df["Resource"].tolist()  # List of resources (e.g., flour, sugar)
product_1 = df["Product 1"].tolist()  # List of Product 1 values
product_2 = df["Product 2"].tolist()  # List of Product 2 values
product_3 = df["Product 3"].tolist()  # List of Product 3 values
resource_limit = df["Resource Limit"].tolist()  # List of resource limits (e.g., max resource usage)

# Define the optimization model using pulp
model = pulp.LpProblem("Production_Optimization", pulp.LpMaximize)

# Define decision variables for each product and resource (0 or 1 for selection)
x1 = pulp.LpVariable.dicts("x1", resources, lowBound=0, upBound=1, cat='Continuous')
x2 = pulp.LpVariable.dicts("x2", resources, lowBound=0, upBound=1, cat='Continuous')
x3 = pulp.LpVariable.dicts("x3", resources, lowBound=0, upBound=1, cat='Continuous')

# Objective function: Maximize profit by selecting products based on resource availability
model += pulp.lpSum([ 
    product_1[i] * x1[resources[i]] + 
    product_2[i] * x2[resources[i]] + 
    product_3[i] * x3[resources[i]] 
    for i in range(len(resources)) 
])

# Add constraints to ensure that the sum of products selected for each resource does not exceed its limit
for i in range(len(resources)):
    model += x1[resources[i]] + x2[resources[i]] + x3[resources[i]] <= resource_limit[i]

# Solve the optimization problem
model.solve(pulp.PULP_CBC_CMD(msg=False))

# Print the optimization status (whether the problem was solved successfully or not)
print("\nOptimization Status:", pulp.LpStatus[model.status])

# Collect and print the selected production tasks (products chosen for each resource)
selected = []
for i in range(len(resources)):
    if x1[resources[i]].value() > 0:  # If Product 1 is selected for the resource
        selected.append(f"Product 1 for {resources[i]}")
    if x2[resources[i]].value() > 0:  # If Product 2 is selected for the resource
        selected.append(f"Product 2 for {resources[i]}")
    if x3[resources[i]].value() > 0:  # If Product 3 is selected for the resource
        selected.append(f"Product 3 for {resources[i]}")

# Print the list of selected tasks
print("\nSelected Production Tasks:", selected)
