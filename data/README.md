# Data and pretrained dependency provenance

## Movie reviews

The local 10,000-review subset was verified against [Stanford's IMDb v1 release](https://ai.stanford.edu/~amaas/data/sentiment/), introduced by Andrew Maas and collaborators in ACL 2011. All texts and binary labels matched. `review_source_manifest.csv` contains archive-member names, labels and normalized text SHA-256 hashes; `lexiscope prepare` reconstructs the exact normalized data from that manifest. All chosen members have matching official training-partition texts; some duplicates also appear in the official test partition. This cannot establish which partition was originally sampled, so results are described as a custom subset benchmark.

The original dataset's README is retained in `IMDB_NOTICE.txt`. Full review text is downloaded locally and excluded from Git; the project code's MIT license does not relicense the review corpus. The custom evaluation retains 9,983 distinct normalized reviews across three partitions. Fictional `example_reviews.csv` is for interface testing and is not part of training or benchmark scoring.

## Language-modeling corpus

`corpus.txt` is the supplied cleaned body of Andrew Carnegie's *James Watt* (1905), identified against [Project Gutenberg ebook 26131](https://www.gutenberg.org/ebooks/26131). A word-sequence comparison found that removed chapter headings/separators explain the differences from the selected source body. The retained preparation preserves the exact text used in the benchmark. The original Gutenberg terms are in `GUTENBERG_NOTICE.txt`; the underlying work is public domain in the United States. The corpus is an attributed source asset, not newly authored project prose.

## Embeddings and models

- [GloVe 6B-200d](https://nlp.stanford.edu/projects/glove/): 400,000 static vectors from Jeffrey Pennington, Richard Socher and Christopher Manning. Vector data uses Open Data Commons PDDL 1.0; the upstream code license is separate. Supply an existing `.txt` file or allow the official archive download.
- [BERT base uncased](https://huggingface.co/google-bert/bert-base-uncased): Apache 2.0; pinned revision `86b5e0934494bd15c9632b12f734a8a67f723594`.
- [GPT-2 tokenizer](https://huggingface.co/openai-community/gpt2): pinned revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Only tokenizer/configuration files are needed for the custom neural models; GPT-2 model weights are not used.
- [TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M): pinned revision `77f1b168e219585646439073245fe87e56b3023e`, from Ronen Eldan and Yuanzhi Li. The original model card does not specify license metadata. Its weights remain an external download. All 108 local weight tensors were verified equal to the official PyTorch release; extra tensors in that release are attention-mask buffers. Actual total parameter count is 3,745,984.

`source_metadata.json` records source identities and corpus preparation. Embedding and generation report metadata record the exact local dependency file hashes. Raw inputs, caches and pretrained/local model weights are ignored. Derived layer vectors, selected GloVe geometry, numerical predictions and generated samples are retained to support independent evidence checks.
