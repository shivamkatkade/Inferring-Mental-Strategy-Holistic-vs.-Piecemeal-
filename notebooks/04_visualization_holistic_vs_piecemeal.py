import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys

# Function to install a package
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully.")

# Install necessary libraries for plotting
install_package("matplotlib")
install_package("seaborn")

# --- Step 1: Define the HMM Parameters from Your Output ---

holistic_means = np.array([[2.12e+02, 3.28e-01],
                           [4.41e+02, 4.07e-01],
                           [1.06e+02, 2.96e-01]])

piecemeal_means = np.array([[2.08e+02, 1.74e-01],
                            [1.23e+03, 3.63e-01],
                            [3.70e+02, 2.70e-01]])

# --- Step 2: Create a DataFrame for Easy Plotting ---

holistic_df = pd.DataFrame(holistic_means, columns=['Fixation Duration', 'Fixation Dispersion'])
holistic_df['State'] = ['State 1', 'State 2', 'State 3']
holistic_df['Group'] = 'Holistic'

piecemeal_df = pd.DataFrame(piecemeal_means, columns=['Fixation Duration', 'Fixation Dispersion'])
piecemeal_df['State'] = ['State 1', 'State 2', 'State 3']
piecemeal_df['Group'] = 'Piecemeal'

# Combine the data for plotting
combined_df = pd.concat([holistic_df, piecemeal_df])

# --- Step 3: Plot the Emission Means for Each Group ---

sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

# Plot for Holistic Group
plt.subplot(1, 2, 1)
sns.barplot(x='State', y='Fixation Duration', data=holistic_df, palette='viridis')
plt.title('Holistic Group - Fixation Duration by State')
plt.ylabel('Average Fixation Duration (ms)')
plt.xlabel('Hidden State')

plt.subplot(1, 2, 2)
sns.barplot(x='State', y='Fixation Dispersion', data=holistic_df, palette='plasma')
plt.title('Holistic Group - Fixation Dispersion by State')
plt.ylabel('Average Fixation Dispersion')
plt.xlabel('Hidden State')

plt.tight_layout()
plt.savefig('Holistic_HMM_States.png')
plt.show()

# Plot for Piecemeal Group
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='State', y='Fixation Duration', data=piecemeal_df, palette='viridis')
plt.title('Piecemeal Group - Fixation Duration by State')
plt.ylabel('Average Fixation Duration (ms)')
plt.xlabel('Hidden State')

plt.subplot(1, 2, 2)
sns.barplot(x='State', y='Fixation Dispersion', data=piecemeal_df, palette='plasma')
plt.title('Piecemeal Group - Fixation Dispersion by State')
plt.ylabel('Average Fixation Dispersion')
plt.xlabel('Hidden State')

plt.tight_layout()
plt.savefig('Piecemeal_HMM_States.png')
plt.show()

print("Charts have been created and saved as 'Holistic_HMM_States.png' and 'Piecemeal_HMM_States.png'.")
