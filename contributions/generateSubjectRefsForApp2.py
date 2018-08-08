import os
import pandas as pd
import numpy as np

man_dir = "/home/sb17/DLTK/contributions/data/dhcp_mr_brains/manual/"
data_dir = "/home/sb17/DLTK/contributions/data/dhcp_mr_brains/automatic/"
all_data_csv_path = "/home/sb17/DLTK/contributions/data/dhcp_mr_brains/automatic/dhcp_all_subjects.csv"
all_data = pd.read_csv(all_data_csv_path, dtype=object, keep_default_na=False, na_values=[]).as_matrix()


def is_entry_manually_annotated(subject_id):
    files = os.listdir(man_dir)
    subj_found = [subject_id in x and 'excl' not in x and 'skel' not in x and 'over' not in x for x in files]
    return any(subj_found)


def get_row_for_manual_entry(sid, subject_id):
    files = os.listdir(man_dir)
    subj_found = [subject_id in x and 'excl' not in x and 'skel' not in x and 'over' not in x for x in files]
    if any(subj_found):
      print("Found entry for: " + str(subject_id))
      file_name = [x for x in files if subject_id in x and 'excl' not in x and 'skel' not in x and 'over' not in x][0]
      print(file_name)
      parts = file_name.split('-')
      subject_row = [sid, parts[0], parts[1], parts[2][1:], man_dir + file_name]
      return subject_row
    else:
      print("Should not see this")
      return []


# Experiment 3: T2 -> manual, then maybe t1 aswell
output_data = []
split_counter = 0
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
            if is_entry_manually_annotated(subj_id):
                subj_row = get_row_for_manual_entry(i, subj_id)
                # subj_row = [i + 1, subj_data_path, subj_data_prefix]
                subj_row.append(subj_data_path)
                subj_row.append(subj_data_prefix)
                if split_counter < 5:  # Initial Training Set
                    subj_row.append(1)
                    subj_row.append(0)
                    subj_row.append(0)
                    subj_row.append(0)
                elif 5 <= split_counter < 7:  # Validation set
                    subj_row.append(0)
                    subj_row.append(1)
                    subj_row.append(0)
                    subj_row.append(0)
                elif 7 <= split_counter < 10:  # Test Set
                    subj_row.append(0)
                    subj_row.append(0)
                    subj_row.append(1)
                    subj_row.append(0)
                elif 10 <= split_counter < 30:  # Unannotated rest of data to be queried for new annotations
                    subj_row.append(0)
                    subj_row.append(0)
                    subj_row.append(0)
                    subj_row.append(1)
                else:  # Unused data for now
                    subj_row.append(0)
                    subj_row.append(0)
                    subj_row.append(0)
                    subj_row.append(0)
                output_data.append(subj_row)
                split_counter = split_counter + 1

df = pd.DataFrame(output_data, columns=["id", "subject_id", "session", "slice","man_path", "subj_data_path", "subj_data_prefix", "train", "val", "test", "unannotated"])
df.to_csv(man_dir + "subject_data.csv", index=False)
