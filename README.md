# SVDR

This is a anonymous repo for NIPS submission

## Setup Environment
```
# install poetry first
# curl -sSL https://install.python-poetry.org | python3 -
poetry install
poetry shell
```

## Download Corpus

```
wget -O ./data/wiki21m.jsonl https://huggingface.co/datasets/svdr/wiki21m/resolve/main/wiki21m.jsonl
```

## Evaluation on Wiki21m

### 1. Build a Binary Token Index
To construct a binary token index for text corpus:
```bash
python -m inference.build_index.binary_token_index \
        --text_file=./data/wiki21m.jsonl \
        --save_file=./index/wiki21m_binary_token.npz \
        --batch_size=32 \
        --num_shift=999 \
        --max_len=256
```
Parameters:
- `--text_file`: Path to the corpus file to be indexed (`.jsonl` format).
- `--save_file`: Path where the index file will be saved (`.npz` format).
- `--batch_size`: Batch size for processing.
- `--num_shift`: Allows for shifting the vocabulary token IDs by a specified amount.
- `--max_len`: Maximum length for tokenization of the documents. 


### 2. Beta Search on Binary Token Index
```bash
python -m inference.search.beta_search \
        --checkpoint=svdr/svdr-nq \
        --query_file=./data/nq-test-questions.jsonl \
        --text_file=./data/wiki21m.jsonl \
        --index_file=./index/wiki21m_binary_token.npz \
        --save_file=./results/nq_search_result.json  \
        --device=cuda
```
Parameters:
- `--query_file`: Path to file containing questions, with each question as a separate line (`.jsonl` format). 
- `--qa_file`: Path to DPR-provided qa file (`.csv` format). Required if `--query_file` is not provided.
- `--text_file`: Path to the corpus file (`.jsonl` format).
- `--index_file`: Path to pre-computed index file (`.npz` format).
- `--save_file`: Path where the search results will be stored (`.json` format).
- `--batch_size`: Number of queries per batch.
- `--num_rerank`: Number of passages to re-rank.

### 3. Scoring on Wiki21m benchmark
```bash
python -m inference.score.eval_wiki21m \
        --text_file=./data/wiki21m.jsonl \
        --result_file=./results/nq_search_result.json \
        --qa_file=./data/nq-test.qa.csv
```
Parameters:
- `--text_file`: Path to the corpus file (`.jsonl` format).
- `--result_file`: Path to search results (`.json` format).
- `--qa_file`: Path to DPR-provided qa file (`.csv` format)


