# Inferring Mental Strategy: Holistic vs. Piecemeal 🧠

This project investigates **how individuals process information**—whether they adopt a **holistic strategy** (seeing the big picture) or a **piecemeal strategy** (focusing on individual components). The analysis combines **data preprocessing, feature engineering, Hidden Markov Models (HMM), clustering, and visualization** to classify and understand mental strategies.

---

## 🔬 Objective
- Identify patterns in human decision-making strategies  
- Classify mental strategies as **Holistic** or **Piecemeal**  
- Provide visual insights for research and educational purposes  

---

## 📊 Methodology

### 1. Data Collection
- Collected raw datasets from eye-tracking, behavioral tests, and related experiments  
- Stored in `.csv` format for processing  

### 2. Data Preprocessing
- Handled missing and noisy values  
- Normalized data to improve model performance  
- Split dataset into **training** and **testing sets**  

### 3. Feature Engineering
- Extracted meaningful features such as:  
  - Fixation duration  
  - Saccade distance  
  - Attention shifts  
- Selected features that are most predictive of holistic or piecemeal strategies  

### 4. Model Training

- **Hidden Markov Model (HMM)** was trained on the processed features to classify participants as **Holistic** or **Piecemeal**.
- The HMM captures temporal patterns in the data, such as sequences of eye movements and attention shifts, making it ideal for this type of behavioral analysis.
- **Training Process:**
  - Dataset split into **training (80%)** and **testing (20%)** sets.
  - Model parameters (number of states, transition probabilities) were optimized using the training set.
  - Evaluated performance using metrics such as **accuracy, precision, recall**, and **F1-score**.
- **Model Saving:**
  - The trained model was saved as a `.pkl` file for reproducibility and future predictions:
  ```python
  import pickle
  with open('hmm_model.pkl', 'wb') as file:
      pickle.dump(hmm_model, file)

<img width="880" height="150" alt="Screenshot 2025-09-26 103326" src="https://github.com/user-attachments/assets/fc816c82-1287-4217-ac02-3a2af1705ce3" />

### 5. Clustering
- Applied clustering algorithms to identify groups of similar strategies  
- Compared clusters with ground truth to validate classification  

### 6. Visualization 📊
- Heatmaps for eye-tracking patterns  
- Scatter plots for feature clustering  
- Interactive plots to explore mental strategy behavior  

---

## 🖼 Visualization: Holistic vs. Piecemeal
The following visualizations compare **holistic** and **piecemeal** strategies:


| Holistic Strategy |   


<img width="1919" height="949" alt="Screenshot 2025-09-26 181939" src="https://github.com/user-attachments/assets/e248fd7f-3a4d-448b-9e8d-3d3d97de2a65" />


| Piecemeal Strategy |   


<img width="1916" height="957" alt="Screenshot 2025-09-26 181953" src="https://github.com/user-attachments/assets/85ae2137-7cc0-4cbc-8a21-c42aa814262b" />
## 🖼 Visualization:


| Heatmap | 


<img width="1916" height="943" alt="Screenshot 2025-09-26 182005" src="https://github.com/user-attachments/assets/9b0539cb-7bf7-4f0b-9fe1-60ed3c17ed13" />


| other Visualization |


<img width="1919" height="981" alt="Screenshot 2025-09-26 182029" src="https://github.com/user-attachments/assets/6897b20f-39d3-4563-8f4b-35e84b155563" />


<img width="1903" height="945" alt="Screenshot 2025-09-26 182043" src="https://github.com/user-attachments/assets/08c0484d-f62c-4ca3-b2a3-fc935685f602" />


<img width="1905" height="914" alt="Screenshot 2025-09-26 182051" src="https://github.com/user-attachments/assets/fd7dc3a4-06cd-4410-a80f-a8d579ffd82c" />


---

## 🛠 Tools & Libraries
- **Python**  
- **pandas, numpy** – Data handling  
- **scikit-learn** – Feature selection, clustering  
- **hmmlearn** – Hidden Markov Model implementation  
- **matplotlib, seaborn, plotly** – Visualization  

---
---

## 🧩 Final Analysis: HMM Parameters

The final analysis examines the trained Hidden Markov Models for **Holistic** and **Piecemeal** strategies. Below are the parameters exactly as obtained from the models:

--- HOLISTIC MODEL ---
**Emission Means (Feature Averages for each Hidden State):**

**Transition Matrix (Probability of moving between states):**

--- PIECEMEAL MODEL ---
**Emission Means (Feature Averages for each Hidden State):**

<img width="597" height="542" alt="Screenshot 2025-09-26 184917" src="https://github.com/user-attachments/assets/a3486eab-8f2a-4de7-acbe-47fe5290271f" />

> **Explanation:**  
> - **Emission Means** show the average values of the features for each hidden state, summarizing typical behavior.  
> - **Transition Matrix** shows how likely the model is to move from one hidden state to another.  
> - Together, these parameters describe the patterns captured by the HMM for holistic and piecemeal strategies.

---

