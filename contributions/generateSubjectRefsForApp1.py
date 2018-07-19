import os
import pandas as pd
import numpy as np

data_dir = "/home/sb17/DLTK/contributions/data/dhcp_mr_brains/automatic/"
all_data_csv_path = "dhcp_all_subjects.csv"
all_data = pd.read_csv(data_dir + all_data_csv_path, dtype=object, keep_default_na=False, na_values=[]).as_matrix()

# Experiment 1: (All data) T2 -> EMt
output_data = []
for i, row in enumerate(all_data):
    subj_id = row[0]
    subj_path = "sub-" + subj_id
    ses_tsv = subj_path + '_sessions.tsv'
    ses_data_path = data_dir + "annotated/" + ses_tsv
    if ses_tsv in os.listdir(data_dir + "annotated/"):
        subj_ses_data = pd.read_csv(ses_data_path, sep='\t', dtype=object, keep_default_na=False,
                                    na_values=[]).as_matrix()
        ses_num = subj_ses_data[0][0]
        # Build path to subject data folder
        subj_data_path = data_dir + "annotated/" + "ses-" + str(ses_num) + "/anat/"
        subj_data_prefix = "sub-" + subj_id + "_ses-" + str(
            ses_num) + "_"  # Reader file looks for post-fix e.g T2w_restore_brain.nii
        if subj_data_prefix + "T1w_restore_brain.nii.gz" in os.listdir(subj_data_path):
            subj_row = [i + 1, subj_data_path, subj_data_prefix]
            if (i < 5):
                subj_row.append(1)
                subj_row.append(0)
            else:
                subj_row.append(0)
                subj_row.append(0)
            output_data.append(subj_row)


df = pd.DataFrame(output_data, columns=["subject_id", "path", "prefix", "bs_exists", "gs_exists"])
df.to_csv("/home/sb17/DLTK/contributions/applications/AL_framework/applications/app1/data/" + "subject_data.csv", index=False)
