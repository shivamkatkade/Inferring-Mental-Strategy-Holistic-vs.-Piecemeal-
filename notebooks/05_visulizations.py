import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- HMM Parameters and DataFrames from Your Analysis ---
# These are the outputs from the analysis and are defined here for the purpose of visualization.

holistic_means = np.array([[2.12e+02, 3.28e-01],
                           [4.41e+02, 4.07e-01],
                           [1.06e+02, 2.96e-01]])

holistic_transmat = np.array([[0.97, 0.  , 0.02],
                              [0.01, 0.98, 0.01],
                              [0.02, 0.01, 0.97]])

piecemeal_means = np.array([[2.08e+02, 1.74e-01],
                            [1.23e+03, 3.63e-01],
                            [3.70e+02, 2.70e-01]])

piecemeal_transmat = np.array([[0.99, 0.  , 0.01],
                               [0.01, 0.99, 0.  ],
                               [0.01, 0.  , 0.98]])

# Assume clustering_df is available from a previous data processing step.
# We will create a sample one here to make the script runnable.
# In a real scenario, this would be loaded from a file or another script.
data = {'avg_fixation_duration': [250, 450, 150, 200, 400, 100],
        'gaze_dispersion': [0.35, 0.45, 0.25, 0.18, 0.38, 0.28],
        'cluster': [0, 1, 0, 0, 1, 0]}
clustering_df = pd.DataFrame(data)

# Create dataframes for HMM state visualization
holistic_df = pd.DataFrame(holistic_means, columns=['Fixation Duration', 'Fixation Dispersion'])
holistic_df['State'] = ['State 1', 'State 2', 'State 3']
holistic_df['Group'] = 'Holistic'

piecemeal_df = pd.DataFrame(piecemeal_means, columns=['Fixation Duration', 'Fixation Dispersion'])
piecemeal_df['State'] = ['State 1', 'State 2', 'State 3']
piecemeal_df['Group'] = 'Piecemeal'

combined_df = pd.concat([holistic_df, piecemeal_df])

# --- Function to Create and Plot HMM Diagram (from visual3.py) ---
def plot_hmm_graph(means, transmat, title, filename):
    plt.figure(figsize=(10, 8))
    G = nx.MultiDiGraph()
    
    # Add nodes with labels
    for i, mean in enumerate(means):
        label = f"State {i+1}\nDuration: {mean[0]:.0f} ms\nDispersion: {mean[1]:.2f}"
        G.add_node(i, label=label)

    # Add edges with weights (probabilities)
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            prob = transmat[i, j]
            if prob > 0:
                G.add_edge(i, j, weight=prob)

    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes and labels
    node_labels = nx.get_node_attributes(G, 'label')
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=2000)
    nx.draw_networkx_labels(G, pos, labels=node_labels)

    # Draw edges and edge labels
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, width=2)
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.savefig(filename)
    plt.show()

# --- Plotting All the Visuals ---

# 1. Participant Clusters Scatter Plot (from visual4.py)
print("--- Generating Participant Clusters Scatter Plot ---")
plt.figure(figsize=(10, 8))
sns.scatterplot(x='avg_fixation_duration', y='gaze_dispersion', hue='cluster', data=clustering_df, palette='viridis', s=100)
plt.title('Participant Clusters based on Eye-Tracking Features')
plt.xlabel('Average Fixation Duration (ms)')
plt.ylabel('Gaze Dispersion')
plt.legend(title='Cluster', labels=['Piecemeal', 'Holistic'])
plt.grid(True)
plt.savefig('Participant_Clusters.png')
plt.show()

# 2. HMM State Bar Plots (from visual1.py)
print("\n--- Generating HMM State Bar Plots ---")
sns.set_style("whitegrid")
plt.figure(figsize=(12, 6))

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

# 3. HMM Transition Matrix Heatmap (from visual2.py)
print("\n--- Generating HMM Transition Matrix Heatmaps ---")
sns.set_style("white")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.heatmap(holistic_transmat, ax=axes[0], annot=True, cmap="YlGnBu", fmt=".2f",
            xticklabels=['State 1', 'State 2', 'State 3'],
            yticklabels=['State 1', 'State 2', 'State 3'])
axes[0].set_title('Holistic Model Transition Matrix')
axes[0].set_xlabel('To Hidden State')
axes[0].set_ylabel('From Hidden State')
sns.heatmap(piecemeal_transmat, ax=axes[1], annot=True, cmap="YlGnBu", fmt=".2f",
            xticklabels=['State 1', 'State 2', 'State 3'],
            yticklabels=['State 1', 'State 2', 'State 3'])
axes[1].set_title('Piecemeal Model Transition Matrix')
axes[1].set_xlabel('To Hidden State')
axes[1].set_ylabel('From Hidden State')
plt.tight_layout()
plt.savefig('Transition_Matrices_Comparison.png')
plt.show()

# 4. HMM State Transition Diagrams (from visual3.py)
print("\n--- Generating HMM State Transition Diagrams ---")
plot_hmm_graph(holistic_means, holistic_transmat,
               'Holistic Model: State Transition Diagram', 'Holistic_Transition_Diagram.png')
plot_hmm_graph(piecemeal_means, piecemeal_transmat,
               'Piecemeal Model: State Transition Diagram', 'Piecemeal_Transition_Diagram.png')

# 5. HMM State Insights & Plots (from visual5.py)
print("\n--- Generating Additional HMM State Insight Plots ---")

# Insight 1: Group Comparison Bar Plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='State', y='Fixation Duration', hue='Group', data=combined_df, palette='Set2')
plt.title('Fixation Duration Comparison (Holistic vs Piecemeal)')
plt.ylabel('Average Fixation Duration (ms)')
plt.xlabel('Hidden State')
plt.subplot(1, 2, 2)
sns.barplot(x='State', y='Fixation Dispersion', hue='Group', data=combined_df, palette='Set1')
plt.title('Fixation Dispersion Comparison (Holistic vs Piecemeal)')
plt.ylabel('Average Fixation Dispersion')
plt.xlabel('Hidden State')
plt.tight_layout()
plt.savefig("Group_Comparison.png")
plt.show()

# Insight 2: Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=combined_df, 
                x='Fixation Duration', 
                y='Fixation Dispersion', 
                hue='Group', 
                style='State', 
                s=150, palette='Dark2')
plt.title("Fixation Duration vs Dispersion (Holistic vs Piecemeal)")
plt.xlabel("Fixation Duration (ms)")
plt.ylabel("Fixation Dispersion")
plt.legend()
plt.savefig("Scatter_Duration_vs_Dispersion.png")
plt.show()

# Insight 3: Heatmaps
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
sns.heatmap(holistic_df[['Fixation Duration','Fixation Dispersion']].set_index(holistic_df['State']),
            annot=True, cmap="Blues", cbar=False)
plt.title("Holistic Group - State Profiles")
plt.subplot(1, 2, 2)
sns.heatmap(piecemeal_df[['Fixation Duration','Fixation Dispersion']].set_index(piecemeal_df['State']),
            annot=True, cmap="Oranges", cbar=False)
plt.title("Piecemeal Group - State Profiles")
plt.tight_layout()
plt.savefig("State_Profiles_Heatmap.png")
plt.show()

# --- New Visualizations Added Below ---

# New Visual 1: Box Plot of Fixation Duration
print("\n--- Generating Box Plot for Fixation Duration ---")
plt.figure(figsize=(8, 6))
sns.boxplot(x='Group', y='Fixation Duration', data=combined_df, palette='viridis')
plt.title('Fixation Duration Distribution by Group')
plt.xlabel('Group')
plt.ylabel('Fixation Duration (ms)')
plt.grid(True)
plt.savefig('Boxplot_Duration.png')
plt.show()

# New Visual 2: Violin Plot of Fixation Dispersion
print("\n--- Generating Violin Plot for Fixation Dispersion ---")
plt.figure(figsize=(8, 6))
sns.violinplot(x='Group', y='Fixation Dispersion', data=combined_df, palette='plasma')
plt.title('Fixation Dispersion Distribution by Group')
plt.xlabel('Group')
plt.ylabel('Fixation Dispersion')
plt.grid(True)
plt.savefig('Violinplot_Dispersion.png')
plt.show()

# New Visual 3: Grouped Bar Plot of Both Metrics
print("\n--- Generating Grouped Bar Plot for Both Metrics ---")
plt.figure(figsize=(12, 8))
combined_melted = combined_df.melt(id_vars=['Group', 'State'], var_name='Metric', value_name='Value')
sns.barplot(x='State', y='Value', hue='Group', data=combined_melted, palette='coolwarm')
plt.title('HMM State Metric Comparison (Holistic vs Piecemeal)')
plt.ylabel('Value')
plt.xlabel('Hidden State')
plt.legend(title='Group')
plt.tight_layout()
plt.savefig('Grouped_Bar_Plot.png')
plt.show()

# New Visual 4: Scatter Plot with Regression Line
print("\n--- Generating Scatter Plot with Regression Line ---")
plt.figure(figsize=(10, 8))
sns.lmplot(x='Fixation Duration', y='Fixation Dispersion', hue='Group', data=combined_df, 
           palette='Set2', markers=['o', 's'])
plt.title('Relationship between Duration and Dispersion')
plt.xlabel('Fixation Duration (ms)')
plt.ylabel('Fixation Dispersion')
plt.grid(True)
plt.tight_layout()
plt.savefig('Scatter_Regression.png')
plt.show()

# New Visual 5: State Clustering with Annotations
print("\n--- Generating State Clustering Plot with Annotations ---")
features = combined_df[['Fixation Duration','Fixation Dispersion']]
scaled = StandardScaler().fit_transform(features)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
combined_df['Cluster'] = kmeans.fit_predict(scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(data=combined_df, 
                x='Fixation Duration', 
                y='Fixation Dispersion', 
                hue='Cluster', 
                style='Group', 
                s=150, palette='Set1')

for i, row in combined_df.iterrows():
    plt.text(row['Fixation Duration']+10, row['Fixation Dispersion']+0.005, 
             f"{row['Group'][0]}-{row['State'][-1]}", fontsize=8)

plt.title("Clustering of States Across Groups")
plt.xlabel("Fixation Duration (ms)")
plt.ylabel("Fixation Dispersion")
plt.legend(title='Cluster', loc='upper right')
plt.tight_layout()
plt.savefig("Cluster_States_Annotated.png")
plt.show()

print("\nAll new plots have been successfully generated and saved.")
