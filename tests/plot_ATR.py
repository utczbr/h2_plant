import pandas as pd
import matplotlib.pyplot as plt
import math

# Load the dataset
filename = 'ATR_linear_regressions.csv'
df = pd.read_csv(filename)

# Identify the independent variable 'x' and the dependent variables
x_col = 'x'
y_cols = [col for col in df.columns if col != x_col]

# Calculate grid size for subplots
num_plots = len(y_cols)
cols = 4  # Number of columns in the plot grid
rows = math.ceil(num_plots / cols)

# Create the figure
fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
axes = axes.flatten()

# Loop through each dependent variable and plot it against 'x'
for i, col in enumerate(y_cols):
    axes[i].plot(df[x_col], df[col], linewidth=2)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].set_xlabel(x_col)
    axes[i].set_ylabel(col)
    axes[i].grid(True, linestyle='--', alpha=0.6)

# Remove any empty subplots if the grid is larger than the number of plots
for i in range(num_plots, len(axes)):
    fig.delaxes(axes[i])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig('linear_interpolations_grid.png')

# Show the plot
plt.show()