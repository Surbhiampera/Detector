import pandas as pd
import os

def sample_csv(input_file, output_file=None, fraction=0.25, nrows=None, random_state=42):
    """
    Read a large CSV file and return/save a smaller sample.

    Args:
        input_file (str): Path to the large CSV file.
        output_file (str, optional): Path to save the sampled CSV. If None, won't save.
        fraction (float, optional): Fraction of rows to sample (default: 0.25).
        nrows (int, optional): Fixed number of rows to sample. Overrides fraction if provided.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Sampled dataframe.
    """
    # Get row count without loading whole file
    total_rows = sum(1 for _ in open(input_file)) - 1  # excluding header

    # Decide sample size
    if nrows is not None:
        sample_size = min(nrows, total_rows)
    else:
        sample_size = int(total_rows * fraction)

    print(f"📂 Input file: {input_file}")
    print(f"➡️ Total rows: {total_rows}")
    print(f"🎯 Sampling: {sample_size} rows")

    # Use pandas sampling
    df = pd.read_csv(input_file)
    df_sample = df.sample(n=sample_size, random_state=random_state)

    # Save if path provided
    if output_file:
        df_sample.to_csv(output_file, index=False)
        print(f"✅ Sample saved to: {output_file}")

    return df_sample


# ---------------- USAGE ----------------
if __name__ == "__main__":
    input_path = "/home/ampara/Downloads/FAERS_merged_25Q2_20250922.csv"      # change to your file
    output_path = "fears_sampled_dataset.csv"   # output file

    # Example 1: Get 1/4th of dataset
    sample_csv(input_path, output_file=output_path, fraction=0.25)

    # Example 2: Get exactly 10000 rows
    # sample_csv(input_path, output_file="sample_10k.csv", nrows=10000)
