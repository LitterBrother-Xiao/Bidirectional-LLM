model_size=0.11
num_layers=12
hidden_size=768
num_attn_heads=12
init_std=0.02

# BERT 1.3B
# model_size=1.3
# num_layers=24
# hidden_size=2048
# num_attn_heads=32
# init_std=0.013

inputfile=xxx/testing.jsonl
MODEL_PATH=xxx
TOKENIZER_PATH=xxx
for step in xxx
do 
    echo global_step${step} > ${MODEL_PATH}/latest

    
    outfile=xxx
    extra_outfile=xxx
    BERT_ARGS="
        --load ${MODEL_PATH} \
        --seq-length 2048 \
        --micro-batch-size 1 \
        --tokenizer-type HFTokenizer \
        --tokenizer-model ${TOKENIZER_PATH} \
        --tensor-model-parallel-size 1 \
        --num-layers ${num_layers} \
        --hidden-size ${hidden_size} \
        --num-attention-heads ${num_attn_heads} \
        --max-position-embeddings 2048 \
        --deepspeed \
        --fp16 \
        --no-load-rng \
        --swiglu \
        --use-rotary-position-embeddings \
        --rotary-percent 0.25 \
        --attention-softmax-in-fp32 \
        --no-load-rng \
        --outfile ${outfile} \
        --inputfile ${inputfile} \
        --extra-outfile ${extra_outfile} \
        --max-iter 10 \
        --length-beam 3 \
        --length-predict \
        --max-predict-length 2048 \
        --load-LP-module \
        --inftype position_beam_simple \
        --position-beam 3 "

    export CUDA_VISIBLE_DEVICES=0
    torchrun --master_port 23564 ../../../../evaluate_gebert_generation.py \
    $BERT_ARGS \

done