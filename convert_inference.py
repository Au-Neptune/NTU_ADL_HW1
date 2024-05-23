import argparse, json, os

def parse_args():
    parser = argparse.ArgumentParser(description="format data")
    parser.add_argument("--context_name", type=str, help="The name of the context data.")
    parser.add_argument("--test_name", type=str, help="The name of the test data.")
    parser.add_argument("--output_path", type=str, default="./tmp/test_converted.json" ,help="The output directory.")
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
    test_data = convert_json_to_data(args.test_name)

    for test in test_data:
        for i, value in enumerate(test["paragraphs"]):
            test[f"paragraph{i}"] = context_data[value]
        del test["paragraphs"]

    convert_data_to_json(test_data, args.output_path)

if __name__ == "__main__":
    main()