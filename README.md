
# BERT-based Named Entity Recognition (NER) with SimpleTransformers

## Project Overview
This project implements Named Entity Recognition (NER) using the BERT model. We use the `simpletransformers` library to fine-tune a pre-trained BERT model on a custom NER dataset, classifying words into categories like person names, organizations, locations, and more.

## Installation

To run this project, you'll need the following Python packages:
- `simpletransformers`
- `scikit-learn`
- `pandas`
- `matplotlib`

You can install the dependencies using: 'requirements.txt' file provided


## Dataset
The dataset used is a custom NER dataset provided in CSV format with columns:
- `sentence_id`: Unique identifier for sentences.
- `words`: The words in each sentence.
- `labels`: The named entity labels corresponding to each word.

Ensure the dataset is available in the correct format and path.

## Model Training

We use a BERT model pre-trained on the English language, fine-tuning it on the NER task. The following are the key hyperparameters:
- `learning_rate`: 1e-4
- `num_train_epochs`: 1
- `train_batch_size`: 32
- `eval_batch_size`: 32

To start training the model, run the following code:
```python
from simpletransformers.ner import NERModel, NERArgs

# Prepare the arguments for training
args = NERArgs()
args.num_train_epochs = 1
args.learning_rate = 1e-4
args.train_batch_size = 32
args.eval_batch_size = 32
args.overwrite_output_dir = True

# Define the model
model = NERModel('bert', 'bert-base-cased', labels=label, args=args, use_cuda=False)

# Train the model
model.train_model(train_data, eval_data=test_data)

# Saving the README.md file
readme_path = "/mnt/data/README.md"
with open(readme_path, "w") as f:
    f.write(readme_content)

readme_path  # Return the path to the generated README file


