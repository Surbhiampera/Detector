import os
import pandas as pd

# ---------------- CONFIG ----------------
FILES = {
    "DEMO": "DEMO25Q2.txt",
    "DRUG": "DRUG25Q2.txt",
    "REAC": "REAC25Q2.txt",
    "OUTC": "OUTC25Q2.txt",
    "INDI": "INDI25Q2.txt",
    "RPSR": "RPSR25Q2.txt",
    "THER": "THER25Q2.txt"
}

separator = "$"

# Columns to keep
demo_keep = ["primaryid","caseid","caseversion","fda_dt","age","sex","reporter_country","rept_cod"]
drug_keep = ["primaryid","caseid","drug_seq","role_cod","drugname","prod_ai"]
reac_keep = ["primaryid","caseid","pt"]
outc_keep = ["primaryid","caseid","outc_cod"]
indi_keep = ["primaryid","caseid","indi_drug_seq","indi_pt"]
rpsr_keep = ["primaryid","caseid","rpsr_cod"]
ther_keep = ["primaryid","caseid","dsg_drug_seq","start_dt","end_dt","dur","dur_cod"]

# ---------------- FOLDERS ----------------
DATA_FOLDER = "/home/ampara/Downloads/ASCII"   # Updated to your folder
OUTPUT_FOLDER = "./cleaned_data"              # Folder to save cleaned files
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- LOAD FILE FUNCTION ----------------
def load_file(file_path, keep_cols):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
        return pd.DataFrame(columns=keep_cols)
    
    df = pd.read_csv(file_path, sep=separator, header=0, encoding="latin-1", dtype=str, low_memory=False)
    return df[keep_cols]

# ---------------- LOAD ALL FILES ----------------
demo = load_file(os.path.join(DATA_FOLDER, FILES["DEMO"]), demo_keep)
drug = load_file(os.path.join(DATA_FOLDER, FILES["DRUG"]), drug_keep)
reac = load_file(os.path.join(DATA_FOLDER, FILES["REAC"]), reac_keep)
outc = load_file(os.path.join(DATA_FOLDER, FILES["OUTC"]), outc_keep)
indi = load_file(os.path.join(DATA_FOLDER, FILES["INDI"]), indi_keep)
rpsr = load_file(os.path.join(DATA_FOLDER, FILES["RPSR"]), rpsr_keep)
ther = load_file(os.path.join(DATA_FOLDER, FILES["THER"]), ther_keep)

# ---------------- SAMPLE UNIQUE PATIENTS / DRUGS ----------------
# 5k unique patients
unique_patients = demo["primaryid"].dropna().unique()
sample_patients = pd.Series(unique_patients).sample(n=min(5000, len(unique_patients)), random_state=42)
demo_sampled = demo[demo["primaryid"].isin(sample_patients)]

# 10k unique drugs
unique_drugs = drug["prod_ai"].dropna().unique()
sample_drugs = pd.Series(unique_drugs).sample(n=min(10000, len(unique_drugs)), random_state=42)
drug_sampled = drug[drug["prod_ai"].isin(sample_drugs)]

# ---------------- FILTER OTHER FILES BASED ON SAMPLED PATIENTS ----------------
reac_sampled = reac[reac["primaryid"].isin(sample_patients)]
outc_sampled = outc[outc["primaryid"].isin(sample_patients)]
indi_sampled = indi[indi["primaryid"].isin(sample_patients)]
rpsr_sampled = rpsr[rpsr["primaryid"].isin(sample_patients)]
ther_sampled = ther[ther["primaryid"].isin(sample_patients)]

# ---------------- SAVE FILES WITH ACTUAL DATA ----------------
demo_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "DEMO_cleaned.csv"), index=False)
drug_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "DRUG_cleaned.csv"), index=False)
reac_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "REAC_cleaned.csv"), index=False)
outc_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "OUTC_cleaned.csv"), index=False)
indi_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "INDI_cleaned.csv"), index=False)
rpsr_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "RPSR_cleaned.csv"), index=False)
ther_sampled.to_csv(os.path.join(OUTPUT_FOLDER, "THER_cleaned.csv"), index=False)

print("✅ All files reduced and saved with actual data rows successfully in 'cleaned_data' folder.")
