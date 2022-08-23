# # rename images
# for p in source.iterdir():
#     case_id = p.name.rsplit("_", 1)[0]
#     if len(case_id) == 5:
#         new_case_id = case_id
#     else:
#         new_case_id = f"{case_id[:-1]}_{case_id[-1]}"
#     os.rename(p, p.parent / f"{new_case_id}_0000.nii.gz")

# # rename labels
# for p in source.iterdir():
#     case_id = p.name.rsplit(".")[0]
#     if len(case_id) == 5:
#         new_case_id = case_id
#     else:
#         new_case_id = f"{case_id[:-1]}_{case_id[-1]}"
#     if p.name.endswith(".json"):
#         os.rename(p, p.parent / f"{new_case_id}.json")
#     else:
#         os.rename(p, p.parent / f"{new_case_id}.nii.gz")
