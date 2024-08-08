python /nvme/xys/Megatron-DeepSpeed/convert_megatron_to_hf_ckp.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --load_path /nvme/xys/checkpoint/megatron_ckpt/pretraining_gebert/gebert_pretraining_large/iter_0028000  \
    --save_path /nvme/xys/checkpoint/hf_ckpt \
    --print-checkpoint-structure \