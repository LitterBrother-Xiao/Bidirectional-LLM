# export CUDA_LAUNCH_BLOCKING=1
export HF_DATASETS_OFFLINE="1"
num_layers=40
hidden_size=2560
num_attn_heads=20
init_std=0.013

export CUDA_VISIBLE_DEVICES=0
torchrun --master_port 18888 ../../../eval_harness_all.py \
    --task_list arc_easy \
    --load xxx \
    --seq-length 2048 \
    --micro-batch-size 1 \
    --tokenizer-model xxx \
    --tensor-model-parallel-size 1 \
    --num-layers ${num_layers} \
    --hidden-size ${hidden_size} \
    --num-attention-heads ${num_attn_heads} \
    --max-position-embeddings 2048 \
    --deepspeed \
    --bf16 \
    --no-load-rng \
    --no-load-optim \
    --no-load-lr-state \
    --swiglu \
    --use-rotary-position-embeddings \
    --rotary-percent 0.25 \
    --inftype "max_beam_based" \
    --beam 1
