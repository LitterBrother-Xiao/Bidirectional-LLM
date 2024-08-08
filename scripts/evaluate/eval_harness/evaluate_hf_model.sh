lm-eval --model hf \
    --model_args pretrained=xxx,backend=seq2seq,trust_remote_code=True,max_length=1040 \
    --tasks xxx \
    --device cuda:6 \
    --batch_size 8
