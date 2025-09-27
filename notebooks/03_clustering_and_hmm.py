import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# --- LOAD FEATURE DATA ---
features_df = pd.read_csv('participant_features.csv')
features_df = features_df.dropna()
print(f"Loaded features for {len(features_df)} participants.")

# --- CLUSTERING ---
features_to_cluster = features_df[['num_fixations', 'avg_fixation_duration', 'gaze_dispersion', 'avg_pupil_size']]
scaled_features = StandardScaler().fit_transform(features_to_cluster)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
features_df['cluster'] = kmeans.fit_predict(scaled_features)

features_df.to_csv('participant_clusters.csv', index=False)
print("Clustering complete. Results saved to 'participant_clusters.csv'.")
print("\nCluster Summary:")
print(features_df.groupby('cluster').mean())

# --- HMM TRAINING ---
holistic_sequences = []
piecemeal_sequences = []

for participant_id in features_df['participant_id']:
    try:
        cluster = features_df.loc[features_df['participant_id'] == participant_id, 'cluster'].iloc[0]
        ivt_file = os.path.join(r'D:\cleaneddata', str(participant_id), 'IVT_cleaned.xlsx')
        ivt_df = pd.read_excel(ivt_file)

        fixation_features = ivt_df[['Fixation Duration', 'Fixation Dispersion']].dropna()
        sequence = fixation_features.values

        if len(sequence) == 0:
            continue  # Skip empty sequences

        if cluster == 0:
            holistic_sequences.append(sequence)
        else:
            piecemeal_sequences.append(sequence)

    except FileNotFoundError:
        print(f"Participant {participant_id} file not found. Skipping...")
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")

# Train HMMs
if holistic_sequences:
    holistic_model = hmm.GaussianHMM(n_components=3, n_iter=100)
    holistic_model.fit(np.concatenate(holistic_sequences), [len(s) for s in holistic_sequences])

if piecemeal_sequences:
    piecemeal_model = hmm.GaussianHMM(n_components=3, n_iter=100)
    piecemeal_model.fit(np.concatenate(piecemeal_sequences), [len(s) for s in piecemeal_sequences])

# Save HMM parameters
with open('hmm_analysis_output.txt', 'w') as f:
    f.write("--- HMM PARAMETERS ---\n")
    if holistic_sequences:
        f.write("\n--- HOLISTIC MODEL ---\n")
        f.write("Emission Means:\n")
        f.write(np.array2string(holistic_model.means_, precision=2, separator=', ') + "\n")
        f.write("Transition Matrix:\n")
        f.write(np.array2string(holistic_model.transmat_, precision=2, separator=', ') + "\n")
    if piecemeal_sequences:
        f.write("\n--- PIECEMEAL MODEL ---\n")
        f.write("Emission Means:\n")
        f.write(np.array2string(piecemeal_model.means_, precision=2, separator=', ') + "\n")
        f.write("Transition Matrix:\n")
        f.write(np.array2string(piecemeal_model.transmat_, precision=2, separator=', ') + "\n")

print("\nHMM training complete. Parameters saved to 'hmm_analysis_output.txt'.")
