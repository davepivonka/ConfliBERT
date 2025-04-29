This data was already packaged with ConfliBERT, but unusable because of its structure and missing files required for fine-tuning for NER tasks.

First, convert from space separated to tab separated format, and create labels.json:
```python
# FOR CAMEO NER

import os

def convert_spaces_to_tabs(input_file, output_file=None):
    """Convert a space-separated NER file to tab-separated format."""
    if output_file is None:
        output_file = input_file + ".tmp"
        overwrite = True
    else:
        overwrite = False

    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:  # Empty line
                    f_out.write('\n')
                    continue

                # Split by space and get the word and label
                parts = line.split()
                if len(parts) >= 2:
                    word = parts[0]
                    label = parts[-1]
                    # Write with tab separation
                    f_out.write(f"{word}\t{label}\n")
                else:
                    f_out.write(line + '\n')

    if overwrite:
        os.replace(output_file, input_file)

def create_labels_json(data_dir):
    """Create labels.json file from the data files"""
    import json

    labels = set()
    for filename in ["train.txt", "test.txt"]:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) == 2:
                            labels.add(parts[1])

    labels_list = sorted(list(labels))
    labels_file = os.path.join(data_dir, "labels.json")
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels_list, f)

# Convert files
data_dir = "./data/cameo_ner/"
convert_spaces_to_tabs(os.path.join(data_dir, "train.txt"))
convert_spaces_to_tabs(os.path.join(data_dir, "test.txt"))
create_labels_json(data_dir)
```

Second part, create dev.txt for evaluation:
```python
# FOR CAMEO NER

import os
import random

def analyze_and_split_data(input_file, train_output, dev_output, dev_ratio=0.2, seed=42):
    """
    Analyzes the NER data file structure and splits it into train and dev sets,
    preserving sentence boundaries.

    Args:
        input_file: Path to the input file (original training data)
        train_output: Path to write the new training data
        dev_output: Path to write the development data
        dev_ratio: Proportion of sentences to use for dev set (default: 0.2)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    # Read the entire file and split into sentences
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines to get sentences
    # (normalize newlines first to handle different formats)
    normalized_content = content.replace('\r\n', '\n')
    sentences = normalized_content.split('\n\n')

    # Remove any empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Count sentences
    total_sentences = len(sentences)
    print(f"Total number of sentences: {total_sentences}")

    # Determine how many sentences for dev set
    dev_count = int(total_sentences * dev_ratio)
    train_count = total_sentences - dev_count
    print(f"Splitting into {train_count} training sentences and {dev_count} dev sentences")

    # Randomly select sentences for dev set
    dev_indices = random.sample(range(total_sentences), dev_count)
    dev_sentences = [sentences[i] for i in dev_indices]
    train_sentences = [sentences[i] for i in range(total_sentences) if i not in dev_indices]

    # Write the files
    with open(train_output, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_sentences))
        f.write('\n\n')  # Add final newline

    with open(dev_output, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(dev_sentences))
        f.write('\n\n')  # Add final newline

    print(f"Created {train_output} with {len(train_sentences)} sentences")
    print(f"Created {dev_output} with {len(dev_sentences)} sentences")

if __name__ == "__main__":
    data_dir = "./data/cameo_ner/"
    train_file = os.path.join(data_dir, "train.txt")

    # Create backup of original training file
    if not os.path.exists(os.path.join(data_dir, "train_full.txt")):
        import shutil
        shutil.copy(train_file, os.path.join(data_dir, "train_full.txt"))
        print(f"Created backup of original training file: train_full.txt")

    # Split the data
    analyze_and_split_data(
        os.path.join(data_dir, "train_full.txt"),
        os.path.join(data_dir, "train.txt"),
        os.path.join(data_dir, "dev.txt")
    )

    # The files will already be in the correct tab-separated format if
    # they were previously converted from space-separated format
```