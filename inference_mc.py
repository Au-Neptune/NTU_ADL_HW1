import json, torch, argparse
from tqdm import tqdm
from itertools import chain
from transformers import AutoTokenizer, AutoModelForMultipleChoice

TEST_DATA = "data/test.json"

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default=None,
        help="The name of the test data.",
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./data/mc_result.json", 
        help="Where to store the final result."
    )
    args = parser.parse_args()
    return args

def preprocess_function(tokenizer, examples):
        question_header_name = "question"
        paragraph_names = [f"paragraph{i}" for i in range(4)]

        question = examples[question_header_name]
        sentences = [
            [[question, examples[paragraph]] for paragraph in paragraph_names]
        ]

        # Flatten out
        sentences = list(chain(*sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            sentences,
            return_tensors="pt", 
            max_length=512,
            padding='max_length',
            truncation=True,
        )

        return tokenized_examples

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)
    model.to('cuda')

    with open(args.test_name, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    log = []
    data = tqdm(test_data)
    for value in data:
        inputs = preprocess_function(tokenizer, value).to('cuda')
        labels = torch.tensor(0).unsqueeze(0).to('cuda')
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
        logits = outputs.logits.argmax().item()
        id = value["id"]
        ans = value[f"paragraph{logits}"]
        log.append(
            {
                "id": id,
                "question": value["question"],
                "context": ans,
                "answers": {
                    "answer_start": [0],
                    "text": [""]
                    },
            }
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()