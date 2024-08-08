import zstandard
import sys
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))
from megatron.data import indexed_dataset

def pile_download(download_url, file_path, i):
    start = time.time()
    zstd_file_path = f"{file_path}{i:02}.jsonl.zst"
    download_path = f"{download_url}{i:02}.jsonl.zst"
    if not os.path.exists(zstd_file_path):
        os.system(f"wget -P {file_path} {download_path}")
        print(f"Finished downloading chunk {i} in {time.time() - start} sec")

def pile_decompress(download_url, file_path, i):
    zstd_file_path = f"{file_path}{i:02}.jsonl.zst"
    output_path = f"{file_path}{i:02}.jsonl"
    if not os.path.exists(output_path):
        if not os.path.exists(zstd_file_path):
            pile_download(download_url, file_path, i)
        start = time.time()
        with open(zstd_file_path, 'rb') as compressed:
            decomp = zstandard.ZstdDecompressor()
            with open(output_path, 'wb') as destination:
                decomp.copy_stream(compressed, destination)
        os.remove(zstd_file_path)
        print(f"Finished decompressing chunk {i} in {time.time() - start} sec")

def pile_preprocess(file_path, vocab_file, num_workers, i):
    json_file_path = f"{file_path}{i:02}.jsonl"
    output_prefix = f"{file_path}pile_bert_train_{i:02}"
    if not os.path.exists(f"{output_prefix}_text_sentence.idx"):
        start = time.time()
        cmd = f"python /nvme/xys/Megatron-DeepSpeed/tools/preprocess_data.py \
                --input {json_file_path} \
                --output-prefix {output_prefix} \
                --vocab-file {vocab_file} \
                --dataset-impl mmap \
                --tokenizer-type BertWordPieceLowerCase \
                --split-sentences \
                --workers {num_workers} "
        # It's possible to hit MemoryError during above cmd since the memory
        # usage is proportional to num_workers. In this case we delete the
        # incomplete output and user shall retry with smaller num_workers.
        # Our experience show that chunk 6, 7, 9, 17, 18, 20, 21, 24, 27
        # particularly have large memory usage.
        if os.system(cmd) == 0: # Success
            print("process_finished")
        else:
            print(f"Error: chunk {i} preprocessing got error, delete \
                    incomplete output. If MemoryError appeared, please retry \
                    with num_workers smaller than {num_workers}.")
            if os.path.exists(f"{output_prefix}_text_sentence.idx"):
                os.remove(f"{output_prefix}_text_sentence.idx")
            if os.path.exists(f"{output_prefix}_text_sentence.bin"):
                os.remove(f"{output_prefix}_text_sentence.bin")
        print(f"Finished preprocessing chunk {i} in {time.time() - start} sec")

def pile_merge(file_path):
    start = time.time()
    num_chunks = 30
    vocab_size = 30524
    for i in range(num_chunks):
        output_prefix = f"{file_path}pile_bert_train_{i:02}"
        assert os.path.exists(f"{output_prefix}_text_sentence.idx")
        assert os.path.exists(f"{output_prefix}_text_sentence.bin")
    builder = indexed_dataset.make_builder(
        f"{file_path}pile_bert_train_text_sentence.bin", impl="mmap",
        vocab_size=vocab_size)
    for i in range(num_chunks):
        chunk_file = f"{file_path}pile_bert_train_{i:02}_text_sentence"
        print(f"Merging file {chunk_file}")
        builder.merge_file_(chunk_file)
    print("Finalizing merged file ...")
    builder.finalize(f"{file_path}pile_bert_train_text_sentence.idx")
    print(f"Finished merging in {time.time() - start} sec")
    # After verifying the merged data with real training, you may want to
    # delete the data chunks.
    # for i in range(num_chunks):
    #     output_prefix = f"{file_path}pile_bert_train_{i:02}"
    #     os.remove(f"{output_prefix}_text_sentence.idx")
    #     os.remove(f"{output_prefix}_text_sentence.bin")


if __name__ == '__main__':
    file_path = "/nvme/xys/pre_trained_data/pile/unzip/"
    vocab_file = "/nvme/xys/DeepSpeedExamples-master/training/HelloDeepSpeed/hf_model/bert-large-uncased/vocab.txt"
    num_workers = 1
    for i in range(23,24):
        pile_preprocess(file_path, vocab_file, num_workers, i)