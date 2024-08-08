python ../..//tools/preprocess_data_gebert.py \
    --json-file xxx.jsonl \
    --json-key 'text' \
    --group-size 2048 \
    --tokenizer-model /nvme/xys/models/qwen \
    --tokenizer-type "HFTokenizer" \
    --output-prefix xxx \
    --dataset-impl mmap \
    --batch-size 2048 \
    --workers 16 \
    --chunk-size 1 \
    --log-interval 1 
