import pandas as pd
import glob
import os

# ✅ Define columns to keep for each dataset type
eye_cols = ["UnixTime", "ET_GazeLeftX", "ET_GazeLeftY",
            "ET_GazeRightX", "ET_GazeRightY",
            "ET_PupilLeft", "ET_PupilRight",
            "ET_ValidityLeft", "ET_ValidityRight"]

ivt_cols = ["UnixTime",
            "Interpolated Gaze X", "Interpolated Gaze Y", "Interpolated Distance",
            "Gaze Velocity", "Gaze Acceleration",
            "Fixation Index", "Fixation X", "Fixation Y",
            "Fixation Start", "Fixation End", "Fixation Duration", "Fixation Dispersion"]

external_cols = ["Timestamp",
                 "Combined Event Source",
                 "SlideEvent",
                 "StimType",
                 "CollectionPhase",
                 "Source StimuliName",
                 "InputEventSource"]

# ✅ Input & Output
input_base = "D:\copy_dataset\STData"        # folder that contains 38 folders
output_base = "D:\cleaneddata"

os.makedirs(output_base, exist_ok=True)

# ✅ Loop through all 38 folders
for folder in glob.glob(os.path.join(input_base, "*")):
    if os.path.isdir(folder):  # only process folders
        folder_name = os.path.basename(folder)
        output_folder = os.path.join(output_base, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # ---- Process EYE dataset ----
        eye_file = os.path.join(folder, "EYE.csv")
        if os.path.exists(eye_file):
            df_eye = pd.read_csv(eye_file)
            df_eye = df_eye[[c for c in eye_cols if c in df_eye.columns]]
            df_eye.to_csv(os.path.join(output_folder, "EYE_cleaned.csv"), index=False)

        # ---- Process IVT dataset ----
        ivt_file = os.path.join(folder, "IVT.csv")
        if os.path.exists(ivt_file):
            df_ivt = pd.read_csv(ivt_file)
            df_ivt = df_ivt[[c for c in ivt_cols if c in df_ivt.columns]]
            df_ivt.to_csv(os.path.join(output_folder, "IVT_cleaned.csv"), index=False)

        # ---- Process EXTERNAL dataset ----
        ext_file = os.path.join(folder, "EXTERNAL.csv")
        if os.path.exists(ext_file):
            df_ext = pd.read_csv(ext_file)
            df_ext = df_ext[[c for c in external_cols if c in df_ext.columns]]
            df_ext.to_csv(os.path.join(output_folder, "EXTERNAL_cleaned.csv"), index=False)

        print(f"✅ Cleaned datasets saved in: {output_folder}")
