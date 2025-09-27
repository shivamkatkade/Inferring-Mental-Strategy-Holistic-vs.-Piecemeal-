import pandas as pd
import numpy as np
import os

# --- PHASE 1: Feature Engineering ---

print("--- Phase 1: Feature Engineering ---")

all_participant_features = []

for participant_id in range(1, 39):
    try:
        folder_path = r'D:\cleaneddata'
        ivt_file = os.path.join(folder_path, str(participant_id), 'IVT_cleaned.xlsx')
        eye_file = os.path.join(folder_path, str(participant_id), 'EYE_cleaned.xlsx')

        # Load data
        ivt_df = pd.read_excel(ivt_file)
        eye_df = pd.read_excel(eye_file)

        # --- IVT Features ---
        fixation_df = ivt_df.groupby('Fixation Index').agg(
            fixation_duration=('Fixation Duration', 'first'),
            fixation_x=('Fixation X', 'mean'),
            fixation_y=('Fixation Y', 'mean')
        ).reset_index()

        num_fixations = len(fixation_df)
        avg_fixation_duration = fixation_df['fixation_duration'].mean()
        std_fixation_duration = fixation_df['fixation_duration'].std()
        gaze_dispersion = np.sqrt(fixation_df['fixation_x'].std()**2 + fixation_df['fixation_y'].std()**2)
        fixation_x_std = fixation_df['fixation_x'].std()
        fixation_y_std = fixation_df['fixation_y'].std()

        # --- Eye Features ---
        # Average pupil size across both eyes
        avg_pupil_size = pd.concat([eye_df['ET_PupilLeft'], eye_df['ET_PupilRight']]).mean()
        pupil_std = pd.concat([eye_df['ET_PupilLeft'], eye_df['ET_PupilRight']]).std()

        # Compile participant features
        participant_features = {
            'participant_id': participant_id,
            'num_fixations': num_fixations,
            'avg_fixation_duration': avg_fixation_duration,
            'std_fixation_duration': std_fixation_duration,
            'gaze_dispersion': gaze_dispersion,
            'fixation_x_std': fixation_x_std,
            'fixation_y_std': fixation_y_std,
            'avg_pupil_size': avg_pupil_size,
            'pupil_std': pupil_std
        }
        all_participant_features.append(participant_features)

    except FileNotFoundError:
        print(f"Participant {participant_id} files not found. Skipping...")
    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")

# Create a DataFrame
features_df = pd.DataFrame(all_participant_features)
print("Feature engineering complete. DataFrame created with participant-level summary features.")
print(features_df.head())
