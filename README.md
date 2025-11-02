 **HiLabs Hackathon 2025 – Patient Risk Identification**

 **Overview**

This project was developed as part of the HiLabs Hackathon 2025 challenge — a real-world simulation of value-based healthcare analytics.    
The objective is to predict patient risk levels using multi-source healthcare data to enable targeted and proactive care management.

In the U.S. Value-Based Care (VBC) model, healthcare providers and payers collaborate to improve outcomes and reduce costs by identifying high-risk patients early.    
Our solution uses clinical, demographic, and utilization data to generate a patient-level risk score that represents the likelihood of adverse health outcomes or care gaps.  

The model helps care teams prioritize patients who require immediate attention, optimize resource allocation, and improve overall care quality.

 **Project Structure**

The repository is organized as follows:

├── create\_env.sh \# Bash script to create and set up the Python environment  
├── requirements.txt \# Python dependencies required to run the project  
├── /notebooks/ \# All Jupyter Notebooks used for analysis and model building  
│ ├── HilabsRound2PS.ipynb  
├── /data/ \# Raw and processed datasets
| ├──/test
│ | ├── patient.csv  
│ | ├── diagnosis.csv  
│ | ├── care.csv     
│ | └── visit.csv  
| ├──/train
│ | ├── patient.csv  
│ | ├── diagnosis.csv  
│ | ├── care.csv  
│ | ├── visit.csv  
│ | └── risk.csv 
├── /results/ \# Generated outputs and prediction files  
│ └── Hivise2.0\_HiLabs\_Risk\_Score.csv  
└── README.md \# Setup and execution documentation

## **1\. Overall Approach & Data Architecture**

Our strategy was to transform the five raw, disconnected CSVs into a single, flat "master feature table" where each row represents one patient. This master table serves as the single source of truth for training the model.

Our data architecture and workflow followed these 6 steps:

1. Data Ingestion: Loaded the 5 raw CSVs (`patient.csv`, `visit.csv`, `care.csv`, `diagnosis.csv`, `risk.csv`) into Pandas DataFrames.  
2. Data Cleaning: Handled all data quality issues, including:  
   * Placeholder Dates: Converted `0001-01-01` and `8888-12-31` to `NaN` to be treated as nulls.  
   * Messy Headers: Corrected the misaligned header in `diagnosis.csv`.  
   * Flag Conversion: Converted all 't'/'f' string flags to binary integers (1/0).  
3. Hierarchical Feature Engineering: This was the core of our approach. We built features in three levels:  
   * Level 3 (Raw Features): Raw counts and recency metrics (e.g., `total_visits`, `days_since_last_visit`).  
   * Level 2 (Ingredient Features): Domain-specific metrics built from L3 features (e.g., `visit_acceleration_ratio`, `disease_burden_index (DBI)`).  
   * Level 1 (Composite Indices): High-level scores that combine multiple L2 features (e.g., `utilizationindex`, `chronicburdenindex`).  
4. Data Aggregation: All 1:N tables (`visit`, `care`, `diagnosis`) were aggregated by `patient_id` and joined against the main `patient.csv` table.  
5. Model Training & Selection: We trained an XGBoost Regressor on the final feature set, using early stopping and cross-validation to find the best parameters and prevent overfitting.  
6. Prediction: The final, tuned model was re-trained on 100% of the labeled data and used to predict risk scores for the unlabeled test set.

---

## **Feature Engineering**

### **Objective**

Feature engineering aimed to transform raw clinical, demographic, and utilization data into meaningful indicators that reflect patient health behavior, chronic disease burden, and overall care adherence.  
 Our focus was to design interpretable, domain-driven variables that quantify risk dimensions such as chronicity, care gaps, readmissions, and utilization trends.

---

### **Patient-Level Features**

Source: `patient.csv`

| Feature | Description | Logic / Formula | Rationale |
| ----- | ----- | ----- | ----- |
| is\_hot\_spotter | Indicates if the patient was previously identified as a high-risk (“hot spotter”) case. | 1 if `hot_spotter_identified_at` is not null | Prior high-risk identification implies chronic care complexity. |
| days\_since\_hotspotter | Days elapsed since last hotspotter identification. | Current date – `hot_spotter_identified_at` | Reflects time since the patient was last monitored for risk. |
| hotspotterScore | Recency-weighted score for hotspotter status. | `exp(-days_since_hotspotter / 365)` | Recent high-risk identification → higher weight. |
| hot\_spotter\_readmission\_flag | Binary indicator for readmission history. | ‘t’ \= 1, ‘f’ \= 0 | History of readmission is a major driver of future risk. |
| hot\_spotter\_chronic\_flag | Indicates if the patient has chronic care involvement. | ‘t’ \= 1, ‘f’ \= 0 | Chronic conditions correlate with long-term care risk. |
| agerisk | Normalized age factor. | `age / 100` | Older patients typically have higher baseline risk. |
| baseriskindex | Weighted combination of core patient attributes. | `0.3*agerisk + 0.3*chronic_flag + 0.25*readmission_flag + 0.15*hotspotterScore` | Creates a baseline patient risk before integrating clinical data. |

---

### **Visit-Level Features**

Source: `visit.csv`

| Feature | Description | Logic / Formula | Rationale |
| ----- | ----- | ----- | ----- |
| has\_follow\_up | Whether a valid follow-up date exists. | `1 if follow_up_dt != NaN` | Presence of follow-up indicates engagement. |
| visit\_duration\_days | Duration of each visit. | `visit_end_dt – visit_start_dt` | Captures inpatient vs. outpatient nature of visit. |
| readmission\_rate | Ratio of readmissions to total visits. | `total_readmissions / total_visits` | High values signal unstable medical conditions. |
| visit\_acceleration\_ratio | Growth in visit frequency (last 30 vs 365 days). | `visits_last_30_days / (visits_last_365_days + 1)` | Detects sudden spikes in care needs. |
| Followup\_Compliance\_Index | Share of visits with valid follow-ups. | `total_followups / (total_visits + 1)` | Proxy for adherence to medical advice. |
| inpatient\_to\_total\_visit\_ratio | Fraction of inpatient encounters. | `inpatient_visits / total_visits` | Reflects care intensity and hospitalization frequency. |
| ed\_visit\_ratio | Fraction of emergency department visits. | `ED_visits / total_visits` | Emergency visits are early warning signs for deterioration. |
| utilizationindex | Weighted composite of visit-level risks. | `0.2*visit_acceleration + 0.2*ed_ratio + 0.15*inpatient_ratio + 0.15*(1-FCI) + 0.3*readmission_rate` | Captures utilization trends and compliance behavior holistically. |

---

### **Care-Level Features**

Source: `care.csv`

| Feature | Description | Logic / Formula | Rationale |
| ----- | ----- | ----- | ----- |
| care\_gap\_ind | Indicates presence of missed or delayed care. | ‘t’ \= 1, ‘f’ \= 0 | Unaddressed gaps increase future risk. |
| days\_since\_last\_care | Recency of last care event. | `today – last_care_dt` | Delayed care may signal disengagement. |
| has\_scheduled\_care | Whether the patient has upcoming scheduled care. | `1 if next_care_dt not null` | Indicates proactive management. |
| care\_adherence\_index | Fraction of care gaps per total events. | `care_gaps / total_care_events` | Lower value \= better adherence. |
| care\_engagement\_ratio | Proportion of patients with future appointments. | `scheduled_care / total_care_events` | Measures commitment to ongoing care. |
| care\_recency\_index | Exponential decay based on care recency. | `exp(-days_since_last_care / 365)` | Penalizes long gaps in care. |
| hba1c\_control\_index | Normalized glucose control indicator (if HbA1c available). | `1 - (avg_hba1c - 5.5)/5, clipped(0,1)` | Quantifies diabetes management control. |
| carebehaviorscore | Composite measure of care engagement and adherence. | `0.4*(1-adherence) + 0.3*recency + 0.3*(1-engagement)` | Synthesizes behavior-related care risk factors. |

---

### **Diagnosis-Level Features**

Source: `diagnosis.csv`

| Feature | Description | Logic / Formula | Rationale |
| ----- | ----- | ----- | ----- |
| total\_diagnoses | Count of total diagnosis entries per patient. | — | Higher counts → higher disease burden. |
| total\_chronic\_diagnoses | Number of chronic diagnoses. | Sum of `is_chronic` | Core indicator of long-term disease load. |
| chronic\_ratio | Ratio of chronic diagnoses to total. | `chronic / total` | Chronic dominance signals sustained risk. |
| has\_diabetes / has\_cancer / has\_hypertension | Binary flags for major chronic diseases. | Regex search in `condition_name` | Identifies specific high-impact conditions. |
| disease\_burden\_index (DBI) | Weighted health burden score. | `0.5*chronic_ratio + 0.2*diabetes + 0.3*cancer` | Reflects cumulative severity of conditions. |
| comorbidity\_flag | Indicates co-existence of 2+ chronic diseases. | `(diabetes + cancer + hypertension) ≥ 2` | Captures multi-disease complexity. |
| chronicburdenindex | Combined index of chronic burden and comorbidity. | `0.5*DBI + 0.3*log(chronic_count+1) + 0.2*comorbidity_flag` | A comprehensive disease load indicator. |

---

### **Feature Selection for Modeling**

We chose XGBoost for its:

* **Performance:** Ability to model complex, non-linear relationships.  
* **Robustness:** Natively handles `NaN` values (like in `avg_hba1c`), which we interpreted as a "missing information" feature.  
* **Interpretability:** Provides clear "feature importance" scores.

After feature generation, correlation and interpretability analysis were conducted to select the most informative predictors for the model.  
 The final set of features used in the XGBoost model were:

\[  
 'hotspotterScore',  
 'readmission\_rate',  
 'hot\_spotter\_chronic\_flag',  
 'visit\_acceleration\_ratio',  
 'care\_adherence\_index',  
 'care\_engagement\_ratio',  
 'chronic\_ratio',  
 'comorbidity\_flag'  
\]

These features were selected because they collectively represent:

* Historical patient status (`hotspotterScore`, `chronic_flag`)

* Utilization behavior (`visit_acceleration_ratio`, `readmission_rate`)

* Care engagement (`care_adherence_index`, `care_engagement_ratio`)

* Clinical burden (`chronic_ratio`, `comorbidity_flag`)

---

### **Summary of Feature Engineering Approach**

Our approach followed a hierarchical aggregation strategy:

1. Cleaned and standardized individual datasets (`patient`, `care`, `visit`, `diagnosis`).

2. Derived interpretable intermediate features (flags, ratios, recency indices).

3. Aggregated patient-level metrics across sources using `groupby(patient_id)`.

4. Combined multi-domain indices (behavioral, clinical, utilization) into a unified patient profile.

5. Selected explainable and high-variance features to improve model interpretability and robustness.

This multi-layered engineering ensured that the risk model not only predicts accurately but also aligns with real-world clinical reasoning used in value-based care frameworks.
