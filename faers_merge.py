import pandas as pd
import os
from datetime import datetime

# ---------------- CONFIG ----------------
DATA_FOLDER = "/home/ampara/Downloads/ASCII"
# OUTPUT_FOLDER = "/home/ampara/Downloads/ASCII/output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

# ---------------- KEEP COLS ----------------
demo_keep = ["primaryid","caseid","caseversion","fda_dt","age","sex","reporter_country","rept_cod"]
drug_keep = ["primaryid","caseid","drug_seq","role_cod","drugname","prod_ai"]
reac_keep = ["primaryid","caseid","pt"]
outc_keep = ["primaryid","caseid","outc_cod"]
indi_keep = ["primaryid","caseid","indi_drug_seq","indi_pt"]
rpsr_keep = ["primaryid","caseid","rpsr_cod"]
ther_keep = ["primaryid","caseid","dsg_drug_seq","start_dt","end_dt","dur","dur_cod"]

# ---------------- GENERIC LOADER ----------------
def load_faers_file(file_path, keep_cols, reduce=True, threshold=100000, random_sample=True):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        return pd.DataFrame(columns=keep_cols)

    df = pd.read_csv(
        file_path,
        sep=separator,
        header=0,
        encoding="latin-1",
        dtype=str,
        low_memory=False
    )

    # reduce dataset size if very large
    if reduce and len(df) > threshold:
        quarter = len(df) // 4
        if random_sample:
            df = df.sample(n=quarter, random_state=42)
            print(f"⚠️ Large file ({len(df)} rows). Randomly sampled 1/4: {quarter} rows.")
        else:
            df = df.iloc[:quarter, :]
            print(f"⚠️ Large file ({len(df)} rows). Using first 1/4: {quarter} rows.")

    df = df[keep_cols]
    return df

# ---------------- LOAD FILES ----------------
print("Loading FAERS text files from:", DATA_FOLDER)

demo = load_faers_file(os.path.join(DATA_FOLDER, FILES["DEMO"]), demo_keep)
drug = load_faers_file(os.path.join(DATA_FOLDER, FILES["DRUG"]), drug_keep)
reac = load_faers_file(os.path.join(DATA_FOLDER, FILES["REAC"]), reac_keep)
outc = load_faers_file(os.path.join(DATA_FOLDER, FILES["OUTC"]), outc_keep)
indi = load_faers_file(os.path.join(DATA_FOLDER, FILES["INDI"]), indi_keep)
rpsr = load_faers_file(os.path.join(DATA_FOLDER, FILES["RPSR"]), rpsr_keep)
ther = load_faers_file(os.path.join(DATA_FOLDER, FILES["THER"]), ther_keep)

print("Files loaded successfully.\n")

# ---------------- MERGE FILES ----------------
print("Merging files...")

df = demo.merge(drug, on=["primaryid","caseid"], how="inner") \
         .merge(reac, on=["primaryid","caseid"], how="left") \
         .merge(outc, on=["primaryid","caseid"], how="left") \
         .merge(indi, on=["primaryid","caseid"], how="left") \
         .merge(rpsr, on=["primaryid","caseid"], how="left") \
         .merge(ther, left_on=["primaryid","caseid","drug_seq"], right_on=["primaryid","caseid","dsg_drug_seq"], how="left")

if "dsg_drug_seq" in df.columns:
    df = df.drop(columns=["dsg_drug_seq"])

df = df.drop_duplicates()

print("Files merged successfully.\n")

# ---------------- CLEANUP + RENAME ----------------
df = df.rename(columns={
    "sex": "SEX",
    "reporter_country": "RPT_COUNTRY",
    "rept_cod": "RPSR_COD"
})

# ---------------- SHOW OUTPUT ----------------
print("First 10 rows of the merged dataset:\n")
print(df.head(10))

# ---------------- SAVE TO CSV ----------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(OUTPUT_FOLDER, f"FAERS_merged_25Q2_{timestamp}.csv")
df.to_csv(output_file, index=False)
print(f"\nMerged dataset saved as: {output_file}")
