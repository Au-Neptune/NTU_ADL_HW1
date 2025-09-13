# 中文抽取式問答系統 | NTU 應用深度學習

這是一個針對**中文抽取式問答 (Extractive Question Answering)** 的專案，本專案利用深度學習模型，使其能從給定的文章中，自動找出問題的答案。

## 專案亮點

*   **先進的 NLP 處理技術**：採用了以 BERT 為基礎的 `chinese-roberta-wwm-ext` 等預訓練語言模型，並對其進行微調，以適應中文問答任務。
*   **完整的機器學習流程**：專案涵蓋從資料前處理、模型訓練、推論到結果評估的完整端到端 (End-to-End) 流程，展現了獨立開發的能力。
*   **優異的表現**：在課程 Kaggle 競賽中取得了 `0.78` 的高分，證明了模型架構與訓練策略的有效性。
*   **模組化的程式碼**：專案的程式碼結構清晰，易於理解和擴展，並包含了完整的訓練和測試腳本。

## 核心任務

本專案將問答任務拆解為兩個核心挑戰：

1.  **文章段落篩選 (Paragraph Selection)**：從多個段落中，準確篩選出包含答案的關鍵段落。
2.  **答案區間預測 (Span Prediction)**：在目標段落中，精確地標示出答案的起始與結束位置。

![Task Description](./images/Task%20Description.png)

## 技術棧 (Tech Stack)

*   **程式語言**: Python
*   **主要函式庫**: PyTorch, Transformers, Datasets, tqdm
*   **核心模型**: `hfl/chinese-roberta-wwm-ext` (BERT-based model)

## 快速開始

### 1. 環境設置

```bash
pip install -r requirements.txt
```

### 2. 下載資料與預訓練模型

下載預先處理好的資料、以及訓練好的模型權重。

```bash
bash ./download.sh
```

### 3. 重現結果

執行以下指令，即可快速重現本專案在測試集上的預測結果。

```bash
# bash ./run.sh <path_to_context.json> <path_to_test.json> <path_to_output.csv>
bash ./run.sh data/context.json data/test.json submission.csv
```

## 專案結構

```
.
├── data/                 # 包含用於推論的 context.json 和 test.json
├── multiple-choice/      # 包含「段落篩選」模型的相關檔案
├── question-answering/   # 包含「問答」模型的相關檔案
└── rawdata/              # 包含原始 Kaggle 資料集
```

*   `data`: 存放官方提供的 `context.json` 和 `test.json`。
*   `multiple-choice`: 存放**段落篩選**任務的相關模型與腳本。
*   `question-answering`: 存放**答案區間預測**任務的相關模型與腳本。
*   `rawdata`: 存放從 Kaggle 下載的原始訓練、驗證與測試資料集。


## 自行訓練模型

專案中包含了完整的訓練與測試腳本，如果您想從頭開始訓練模型，請遵循以下步驟：


### Part 1: 段落篩選 (Multiple Choice)

#### 1.1 資料前處理

將原始資料轉換為適用於段落篩選任務的格式。

```bash
python ./convert_mc.py \
--context_name rawdata/context.json \
--train_name rawdata/train.json \
--valid_name rawdata/valid.json \
--test_name rawdata/test.json \
--output_dir ./newdata
```
> 執行後，將在 `newdata/` 目錄下生成 `mc_train.json`, `mc_valid.json` 和 `mc_test.json`。

#### 1.2 模型訓練

使用前處理後的資料來訓練段落篩選模型。

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

#### 1.3 模型推論

載入訓練好的模型，對測試資料進行預測，篩選出最相關的段落。

```bash
python ./inference_mc.py \
--model_name_or_path ./mc/chinese-roberta-wwm-ext \
--test_name ./newdata/mc_test.json \
--output_path ./newdata/mc_result.json
```

---

### Part 2: 答案區間預測 (Question Answering)

#### 2.1 資料前處理

將原始資料轉換為適用於答案抽取的格式。

```bash
python ./convert_qa.py \
--context_name rawdata/context.json \
--train_name rawdata/train.json \
--valid_name rawdata/valid.json \
--output_dir ./newdata
```
> 執行後，將在 `newdata/` 目錄下生成 `qa_train.json` 和 `qa_valid.json`。

#### 2.2 模型訓練

使用處理好的資料來訓練答案抽取模型。

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

#### 2.3 模型推論

結合前一階段的結果，從篩選出的段落中抽取出最終答案。

```bash
python ./inference_qa.py \
--model_name_or_path ./qa/chinese-roberta-wwm-ext \
--do_predict \
--test_file ./newdata/mc_result.json \
--max_seq_length 512 \
--output_dir ./qa/chinese-roberta-wwm-ext

# 將最終結果移至根目錄
mv ./qa/chinese-roberta-wwm-ext/test_submission.csv ./
```

## 成果展示

本專案在「答案完全匹配 (Exact Match)」指標上取得了優異的成績。下圖展示了不同模型在不同訓練週期 (Epoch) 下於驗證集上的表現：

![EM plot](./images/Exact%20Match%20Plot.png)