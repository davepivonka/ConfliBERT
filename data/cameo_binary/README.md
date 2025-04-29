`cameo_binary` data comes from [javierosorio/keep_it_local/1_data/en/binary/en_annotations_binary_complete.tsv](https://github.com/javierosorio/keep_it_local/blob/9adc00fd46dfbe0a27b6bbf3daddcda226d0863e/1_data/en/binary/en_annotations_binary_complete.tsv)

> Osorio, J. et al. (2024) ‘Keep it Local: Comparing Domain-Specific LLMs in Native and Machine Translated Text using Parallel Corpora on Political Conflict’, 2024 2nd International Conference on Foundation and Large Language Models (FLLM). 2024 2nd International Conference on Foundation and Large Language Models (FLLM), IEEE. doi:10.1109/fllm63129.2024.10852489.

Transformed to fit the training data format and split up 70-15-15:
```python
# Preprocess the 'cameo_binary' dataset:
import os
import pandas as pd

# 1) Remove the header/top row from the original TSV
input_file = "/content/en_annotations_binary_complete.tsv"
with open(input_file, "r", encoding="utf-8") as fin:
    fin.readline()            # skip the first line
    data_lines = fin.readlines()
with open(input_file, "w", encoding="utf-8") as fout:
    fout.writelines(data_lines)

# 2) Create the dataset directory
data_dir = "./data/cameo_binary"
os.makedirs(data_dir, exist_ok=True)

# 3) Load the full file (no header) and shuffle
df = pd.read_csv(input_file, sep="\t", header=None)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 4) Compute split indices for 70% train, 15% eval, 15% test
n = len(df)
train_end = int(0.70 * n)
eval_end  = train_end + int(0.15 * n)

# 5) Slice into train / eval / test
train_df = df.iloc[:train_end]
eval_df  = df.iloc[train_end:eval_end]
test_df  = df.iloc[eval_end:]

# 6) Save each split as TSV (no header)
train_df.to_csv(os.path.join(data_dir, "train.tsv"), sep="\t", header=False, index=False)
eval_df.to_csv( os.path.join(data_dir, "dev.tsv"),  sep="\t", header=False, index=False)
test_df.to_csv( os.path.join(data_dir, "test.tsv"),  sep="\t", header=False, index=False)

print(f"Saved splits in {data_dir}:")
print(f"  train  = {len(train_df)} rows")
print(f"  dev   = {len(eval_df)} rows")
print(f"  test   = {len(test_df)} rows")
```