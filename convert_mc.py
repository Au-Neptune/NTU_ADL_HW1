import argparse, json, os

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument("--context_name", type=str, help="The name of the context data.")
    parser.add_argument("--train_name", type=str, help="The name of the train data.")
    parser.add_argument("--valid_name", type=str, help="The name of the valid data.")
    parser.add_argument("--test_name", type=str, help="The name of the test data.")
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
    test_data = convert_json_to_data(args.test_name)

    for train in train_data:
        train["label"] = train["paragraphs"].index(train["relevant"])
        for i, value in enumerate(train["paragraphs"]):
            train[f"ending{i}"] = context_data[value]
        train['sent2'] = ""
        train["sent1"] = train.pop("question")
        del train["paragraphs"]
        del train["relevant"]

    for valid in valid_data:
        valid["label"] = valid["paragraphs"].index(valid["relevant"])
        for i, value in enumerate(valid["paragraphs"]):
            valid[f"ending{i}"] = context_data[value]
        valid['sent2'] = ""
        valid["sent1"] = valid.pop("question")
        del valid["paragraphs"]
        del valid["relevant"]

    for test in test_data:
        for i, value in enumerate(test["paragraphs"]):
            test[f"paragraph{i}"] = context_data[value]
        del test["paragraphs"]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    convert_data_to_json(train_data, os.path.join(args.output_dir, "mc_train.json"))
    convert_data_to_json(valid_data, os.path.join(args.output_dir, "mc_valid.json"))
    convert_data_to_json(test_data, os.path.join(args.output_dir, "mc_test.json"))

if __name__ == "__main__":
    main()