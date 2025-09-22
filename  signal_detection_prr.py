import pandas as pd
import numpy as np
from io import StringIO

# ---  FAERS-style dataset ---
data_path = "data/synthetic_reports.csv"

# Load into DataFrame
df = pd.read_csv(data_path, parse_dates=["date_reported"])
df = df.drop_duplicates(["patient_id","drug_name","adverse_event"])

# --- CONFIG ---
TIME_FREQ = "M"  # monthly
PRR_THRESH = [1.0, 2.0]  # thresholds for state assignment
ALARM_P_TRANS = 0.3
ALARM_PI = 0.2

# Create time windows
df['period'] = df['date_reported'].dt.to_period(TIME_FREQ).dt.to_timestamp()

# Aggregate counts
def agg_counts(df_period):
    total_reports = df_period['patient_id'].nunique()
    d_counts = df_period.groupby('drug_name')['patient_id'].nunique().rename('N_d')
    e_counts = df_period.groupby('adverse_event')['patient_id'].nunique().rename('N_e')
    de_counts = df_period.groupby(['drug_name','adverse_event'])['patient_id'].nunique().rename('N_de')
    return total_reports, d_counts, e_counts, de_counts

# Example: check (Furosemide, Hallucinations)
drugX = "Furosemide"
eventY = "Seizure"

periods = sorted(df['period'].unique())
rows = []
for p in periods:
    sub = df[df['period']==p]
    N_total, d_counts, e_counts, de_counts = agg_counts(sub)
    N_de = de_counts.get((drugX,eventY), 0)
    N_d = d_counts.get(drugX, 0)
    N_e = e_counts.get(eventY, 0)

    if N_d == 0 or N_e == 0:
        prr = 0.0
    else:
        prr = ((N_de+0.5)/(N_d+0.5)) / ((N_e+0.5)/(N_total+0.5))

    rows.append((p, N_de, N_d, N_e, N_total, prr))

ts = pd.DataFrame(rows, columns=['period','N_de','N_d','N_e','N_total','PRR']).sort_values('period')

# Map PRR to discrete state
def prr_to_state(prr):
    if prr < PRR_THRESH[0]:
        return 0
    elif prr < PRR_THRESH[1]:
        return 1
    else:
        return 2

ts['state'] = ts['PRR'].apply(prr_to_state)

# Build transition matrix
trans_counts = np.zeros((3,3), dtype=int)
states = ts['state'].values
for a,b in zip(states[:-1], states[1:]):
    trans_counts[a,b] += 1

P = np.zeros_like(trans_counts, dtype=float)
for i in range(3):
    s = trans_counts[i].sum()
    if s > 0:
        P[i,:] = trans_counts[i,:] / s

# Stationary distribution
evals, evecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(evals - 1.0))
pi = np.real(evecs[:,idx])
pi = pi / pi.sum()

print("Time series:\n", ts, "\n")
print("Transition matrix P:\n", P)
print("Stationary distribution π:", pi)

# Alarm check
if len(states) > 0:
    cur_state = states[-1]
    p_to_signal = P[cur_state, 2] if cur_state < 2 else 0
    if cur_state == 2:
        print("⚠️ ALARM: Current state is SIGNAL")
    elif p_to_signal > ALARM_P_TRANS:
        print(f"⚠️ ALARM: High probability to transition to SIGNAL soon (P={p_to_signal:.2f})")
    elif pi[2] > ALARM_PI:
        print(f"⚠️ ALARM: Long-run probability of SIGNAL is high (π_signal={pi[2]:.2f})")
    else:
        print("No alarm detected.")