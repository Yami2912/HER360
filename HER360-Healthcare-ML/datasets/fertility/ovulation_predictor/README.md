# Ovulation predictor dataset notes (v1)

Status: college major project use only. Not validated for clinical or production use.

- raw/: source file as supplied (do not edit).
- processed/ovulation_model_table_v1.csv: output of preprocessing/fertility/ovulation_prep.py
- processed/ovulation_prep_log_v1.json: what the preparation changed.

Known limits
- Label (estimated ovulation day) is source-estimated, not independently confirmed.
- ~157 participants; about 96% of rows share one reproductive category code (meaning undocumented).
- Cannot validate performance for irregular cycles or PCOS.
- Licence / provenance of the source data still to be confirmed.
- Before any real deployment: replace with a professionally verified dataset.
