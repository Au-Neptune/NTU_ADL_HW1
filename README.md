# ADL HW1

## Project Description
This project is part of the NTU Advanced Deep Learning course (Fall 2023) and focuses on Chinese Extractive Question Answering (QA). The objective is to develop a model that can accurately determine the span of text within a given paragraph that answers a specific question. The project involves two main tasks:
1.  **Paragraph Selection**: Identifying the relevant paragraph that contains the answer to the question.
2.  **Span Selection**: Determining the exact start and end positions of the answer within the selected paragraph.

## Key Components
-   **Pre-trained Language Models**: The project utilizes pre-trained language models such as BERT to facilitate both paragraph and span selection. These models are fine-tuned on the provided dataset to improve their performance on the specific task of extractive QA.
-   **Dataset**: The dataset includes various paragraphs and corresponding questions where the answers are always spans within the correct paragraph. The dataset is used for both training and evaluation of the models.
-   **Evaluation Metrics**: The models are evaluated based on their performance on a Kaggle leaderboard, where they are ranked according to their ability to accurately predict answer spans.

for more information please refer to [NTU ADL 2023 Fall - HW1](./NTU%20ADL%202023%20Fall%20-%20HW1.pdf)

## Enviroments
```bash
pip install -r requirements.txt
```
## Quick Start
download the zip of models, tokenizers and data.
```bash
bash ./download.sh
```
In the zip file, there are some folders:
* `data`: the folder contains the original context.json and test.json.
* `rawdata`: the folder contains the original Kaggle dataset.
* `multiple-choice`: the folder contains multiple-choice models.
* `question-answering`: the folder contains question-answering models.

unzip, convert and use my trained chinese-roberta-wwm-ext to reproduce result.
```bash
bash ./run.sh ./data/context.json ./data/test.json ./data/submission.csv
```

---
## Training
### Context Selection
### Data Format
```bash
python ./convert_mc.py \
--context_name rawdata/context.json \
--train_name rawdata/train.json \
--valid_name rawdata/valid.json \
--test_name rawdata/test.json \
--output_dir ./newdata
```
* `context_name`: Path to context.json.
* `train_name`: Path to train.json.
* `valid_name`: Path to valid.json.
* `test_name`: Path to test.json.
* `output_dir`: The output directory where the formatted files will be stored.

after running this script, you should see `mc_train.json`, `mc_valid.json` and `mc_test.json` in your output directory.

---
### Start Training
for example:
```bash
python ./train_mc.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--train_file newdata/mc_train.json \
--validation_file newdata/mc_valid.json \
--max_seq_length 512 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--num_train_epochs 1 \
--output_dir mc/chinese-roberta-wwm-ext
```
* `model_name_or_path`: Path to pretrained model or model from huggingface.co/models.
* `train_file`: Path to `mc_train.json`.
* `validation_file`: Path to `mc_valid.json`.
* `output_dir`: The output directory where the model will be stored.

---
### Question Answering
### Data Format
```bash
python ./convert_qa.py \
--context_name rawdata/context.json \
--train_name rawdata/train.json \
--valid_name rawdata/valid.json \
--output_dir ./newdata
```
* `context_name` Path to context.json.
* `train_name` Path to train.json.
* `valid_name` Path to valid.json.
* `output_dir` The output directory where the formatted files will be stored.

after running this script, you should see `qa_train.json` and `qa_valid.json` in your output directory.
### Start Training
```bash
python ./train_qa.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--do_train \
--train_file newdata/qa_train.json \
--do_eval \
--validation_file newdata/qa_valid.json \
--max_seq_length 512 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--learning_rate 3e-5 \
--num_train_epochs 6 \
--output_dir qa/chinese-roberta-wwm-ext
```
* `model_name_or_path`: Path to pretrained model or model from huggingface.co/models.
* `train_file`: Path to `mc_train.json`.
* `validation_file`: Path to `mc_valid.json`.
* `output_dir`: The output directory where the model will be stored.

## Testing
### Context Selection
```bash
python ./inference_mc.py \
--model_name_or_path ./mc/chinese-roberta-wwm-ext \
--test_name ./newdata/mc_test.json \
--output_path ./newdata/mc_result.json
```
* `model_name_or_path`: Path to pretrained model.
* `test_name`: Path to `mc_test.json`.
* `output_path`: The output path where the result will be stored（json format）.


### Question Answering
```bash
python ./inference_qa.py \
--model_name_or_path ./qa/chinese-roberta-wwm-ext \
--do_predict \
--test_file ./newdata/mc_result.json \
--max_seq_length 512 \
--output_dir ./qa/chinese-roberta-wwm-ext

mv ./qa/chinese-roberta-wwm-ext/test_submission.csv ./
```
* `model_name_or_path`: Path to pretrained model.
* `test_file`: Path to `mc_result.json`.
* `output_path`: The output path where the result will be stored.
* `mv`: move `test_submission.csv` to root directory.
---
