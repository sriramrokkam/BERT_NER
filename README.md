
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
