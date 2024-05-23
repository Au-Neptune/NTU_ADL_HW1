import argparse, json, os

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument("--context_name", type=str, help="The name of the context data.")
    parser.add_argument("--train_name", type=str, help="The name of the train data.")
    parser.add_argument("--valid_name", type=str, help="The name of the valid data.")
    parser.add_argument("--output_dir", type=str, help="The name of the output directory.")
    args = parser.parse_args()
    return args

def convert_json_to_data(json_data):
    with open(json_data, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def convert_data_to_json(data, json_data):
    with open(json_data, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    args = parse_args()

    context_data = convert_json_to_data(args.context_name)
    train_data = convert_json_to_data(args.train_name)
    valid_data = convert_json_to_data(args.valid_name)

    for train in train_data:
        train["context"] = context_data[train["relevant"]]
        train["answer"]["answer_start"] = [train["answer"]["start"]]
        train["answer"]["text"] = [train["answer"]["text"]]
        del train["answer"]["start"]
        del train["relevant"]
        del train["paragraphs"]
        train["answers"] = train.pop("answer")

    for valid in valid_data:
        valid["context"] = context_data[valid["relevant"]]
        valid["answer"]["answer_start"] = [valid["answer"]["start"]]
        valid["answer"]["text"] = [valid["answer"]["text"]]
        del valid["answer"]["start"]
        del valid["relevant"]
        del valid["paragraphs"]
        valid["answers"] = valid.pop("answer")

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
        
    convert_data_to_json(train_data, os.path.join(args.output_dir, "qa_train.json"))
    convert_data_to_json(valid_data, os.path.join(args.output_dir, "qa_valid.json"))

if __name__ == "__main__":
    main()