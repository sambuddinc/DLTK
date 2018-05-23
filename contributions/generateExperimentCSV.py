import os
import pandas as pd
import numpy as np

data_dir = "insert appropriate"
all_data_csv_path = "dhcp_all_subjects.csv"
all_data = pd.read_csv(data_dir + all_data_csv_path, dtype=object, keep_default_na=False, na_values=[]).as_matrix()

# Experiment 1: (All data) T2 -> EMt
output_data = []
for i, row in enumerate(all_data):
    subj_id = row[0]
    subj_path = "sub-" + subj_id
    ses_data_path = data_dir + "annotated/" + subj_path + "/" + subj_path + "_sessions.tsv"
    if subj_path in os.listdir(data_dir + "annotated/"):
        subj_ses_data = pd.read_csv(ses_data_path, sep='\t', dtype=object, keep_default_na=False,
                                    na_values=[]).as_matrix()
        ses_num = subj_ses_data[0][0]
        # Build path to subject data folder
        subj_data_path = data_dir + "annotated/" + subj_path + "/" + "ses-" + str(ses_num) + "/anat/"
        subj_data_prefix = "sub-" + subj_id + "_ses-" + str(
            ses_num) + "_"  # Reader file looks for post-fix e.g T2w_restore_brain.nii
        subj_row = [i + 1, subj_data_path, subj_data_prefix]
        output_data.append(subj_row)

df = pd.DataFrame(output_data, columns=["subject_id", "path", "prefix"])
df.to_csv(data_dir + "experiment_1.csv", index=False)
