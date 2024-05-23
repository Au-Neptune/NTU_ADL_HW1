unzip final.zip
echo "[step log] unzip done"
mkdir "tmp"
echo "[step log] make tmp folder"
python ./convert_inference.py --context_name $1 --test_name $2 --output_path ./tmp/test_converted.json
echo "[step log] convert by context done"
python ./inference_mc.py --model_name_or_path ./multiple-choice/chinese-roberta-wwm-ext --test_name ./tmp/test_converted.json --output_path ./tmp/mc_result.json
echo "[step log] multiple choice done"
python ./inference_qa.py --model_name_or_path ./question-answering/chinese-roberta-wwm-ext --do_predict --test_file ./tmp/mc_result.json --max_seq_length 512 --output_dir ./question-answering/chinese-roberta-wwm-ext
echo "question answering done"
mv ./question-answering/chinese-roberta-wwm-ext/test_submission.csv $3
echo "[step log] move to $3"
echo "[step log] ALL DONE"