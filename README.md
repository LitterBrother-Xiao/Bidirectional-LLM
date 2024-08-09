### Environment
1. Install docker and nvidia-docker

2. Run the following command
```bash
  docker pull nvcr.io/nvidia/pytorch:23.12-py3
  docker run --gpus all --shm-size=128g --net=host -dit --rm --name megatron -v /your_dir:/your_dir -v /root/.ssh:/root/.ssh nvcr.io/nvidia/pytorch:23.12-py3
```
3. Install deepspeed
```bash
  pip install deepspeed
```
4. Install megatron-core packages
   download this codebase and get into your own path, then  
  ```bash
    pip install -e .
  ```

### Try to run quickly
1. prepare your pretrained data by running ```bash scripts/process/process_data.sh``` with setting several parameter.  
   We list the important parameters here.
   - `json-file`: set the path of your data file here, supporting parquet and jsonl files now.
   - `file-type`: set parquet or jsonl.
   - `json-key`: set the text key in your data file.
   - `tokenizer-model`: set the tokenizer path.
   - `tokenizer-type`: set your tokenizer type, set "HFTokenizer" if you adopt a specific tokenizer from huggingface, more information refers to `megatron/tokenizer/tokenizer.py`.
   - `output-prefix`: set your processed data file.
   - `group-size`: set your processed text length.

2. train the bi-directional LLM by running ```bash scripts/training/pretrain.sh``` with setting several parameters.  
   Most parameters are the same as those in official [`Megatron-LM`](https://github.com/NVIDIA/Megatron-LM) and [`Megatron-Deepspeed`](https://github.com/microsoft/Megatron-DeepSpeed) codebase, we list the additional parameters to support bi-directional training here.
   ##### training related: different ways to train the bi-directional LLM.
   - `has-sentence-split`: include this if you need split the sentence to perform as conditional training.
   - `has-attention-masking`: include this if you need set the specific attention mask to prevent the source sequence attending to the target sequence.
   - `masked-x-type`: set your masking type for the source sequence, more information refers to `megatron/data/gebert_dataset.py`.
   #### length related: NAR models usually need the know the target length during inference.
   - `length-predict`: include this if you need predict the target length.
   - `max-predict-length`: set the maximum predicted target length.
   - `length-factor`: set the length loss factor.
   - `load-LP-module`: include this if you need load the length prediction module from a pretrained model
   #### DPO related: we support DPO methods to optimize the decoding path preference
   - `dpo-training`: include this if you need DPO training.
   - `dpo-update-model-step`: set the steps to update the reference model, i.e., similar to iterative DPO methods.
   - `dpo-sampling-type`: set the way to sample DPO pairs.
   - `dpo-type`: set the DPO training type, more information refers to `pretrain_gebert.py`.

  3. evaluate your model:  
     (1) reasoning tasks with `lm-evaluation-harness` by running `bash scripts/evaluate/eval_harness/evaluate_all.sh`.  
     Most parameters are the same as those in official [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) codebase, we list our additional parameters here.
     - `has-attention-masking`: iinclude this if you set the specific attention mask during training. 
     - `inftype`: set the inference type, more information refers to `eval_harness_all.py`
       
     (2) language generation tasks with `Mask-Predict` decoding algorithm by running `bash scripts/evaluate/eval_generation/generation_scripts/eval_generation_finetune.sh`.  
     We list the important parameters here.
     - `has-attention-masking`: include this if you set the specific attention mask during training. 
     - `inftype`: set the inference type, more information refers to `evaluate_gebert_generation.py`.
     - `max-iter`: set your decoding steps.
     - `length-beam`: set the length beam number.
     - `position-beam`: set the position beam number if you adopt position beam search method.
     - `tokens-beam`: set the tokens beam number if you adopt tokens beam search method.

## *Note*: this project is in progress，feel free to contact us for further improvements.

