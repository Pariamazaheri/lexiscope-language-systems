# Experiment design and analytical limits

## Data and tokenization

The review corpus is the original local 10,000-review subset. Every normalized text and binary label was matched against the Stanford IMDb v1 archive. Some texts appear in both official partitions; source-member selection prefers an identical training-partition match. The project uses its own documented splits and does not claim a result on the full official IMDb test benchmark.

HTML removal, entity decoding and whitespace normalization precede SHA-256 deduplication. Seventeen repeated texts are removed before stratified sampling. Conflicting duplicate labels are rejected. The split manifest retains original row IDs, normalized text hashes, labels and partitions; model inputs exclude those identifiers. All vectorizers and classifiers fit exclusively on training reviews.

GPT-2 tokenization is shared by both neural architectures. EOS is used as the padding token, but padding is determined by the explicit attention mask, not token equality. The classification head selects the final valid position. Right padding prevents all-masked causal query positions; inference rejects invalid tokens instead of silently substituting another token.

The cleaned biography is a prepared version of Andrew Carnegie's 1905 *James Watt*. Its word order matches the identified Gutenberg body except for removed chapter titles and separators. The prepared text and original Gutenberg terms are retained. The 69,590-token source is separated into train/validation/test regions before block construction. Each block consumes 65 distinct positions: 64 inputs and 64 shifted targets. Blocks step by 65, making target-boundary positions disjoint as well. Training is capped at 32,768 tokens; validation and test each use up to 4,096 from their own source region. Sequential segments can differ in vocabulary, so these perplexities measure continuation-region generalization within one book, not broad language competence.

## Model and optimization controls

Both bodies have width 64, two layers and dropout 0.2. The Transformer has four attention heads, a feed-forward width of 256, learned positional embeddings and a Boolean future mask. Each layer is initialized independently. The LSTM is unidirectional; its selected valid hidden state cannot depend on trailing padding. Both bodies return batch × sequence × width tensors and accept replaceable task heads.

Language modeling uses AdamW at learning rate 0.001, batch size 16 and at most four epochs. Classification uses the same optimizer/rate, batch size 32 and at most six epochs. Weight decay is 0.01; gradient norm is clipped at 1.0. Real early stopping uses patience two and a 1e-5 improvement tolerance. The best validation-loss state is restored before test evaluation. Loss totals are weighted by target tokens for LM and reviews for classification, including the shorter final batch.

One Transformer LM is pretrained at seed 42. Each transfer classifier gets an independent deep copy of that best body. The paired scratch and transfer classifiers have identical head initialization and downstream shuffle/dropout seeds. Their architecture and training budgets match. Downstream seeds 42, 43 and 44 share a fixed data split and a fixed pretraining run. The original LM body is checked for immutability after downstream fitting.

The LSTM classifier is a single-seed reference. TF–IDF uses 20,000 features, unigrams/bigrams, minimum document frequency two, and logistic regression with C=2. It sees complete review text, whereas neural models see at most 128 tokens. Therefore the result is a comparison of practical pipeline configurations, not an isolated architecture experiment controlling text coverage. All configurations were fixed before final test scoring.

## Selection and uncertainty

Validation log loss chooses the operational inference candidate among seed-42 configurations. The 0.5 classification threshold is fixed; the final test set never chooses checkpoints, configurations or thresholds. Three other-seed Transformer contrasts quantify initialization/shuffle variation without choosing a favorable seed.

Paired bootstrapping resamples the same test-review IDs for both models 2,000 times. The stored intervals are percentile 95% intervals for accuracy difference conditional on the fitted pair and this sampled corpus. They are not intervals across independent data splits or pretraining runs. Two of three intervals include zero; the positive mean difference should not be presented as a universal transfer benefit.

## Representation and generation

GloVe neighbor queries normalize each positive/negative vector, combine them, then use cosine similarity against the full vocabulary, excluding query words. Analogy successes are examples, not an accuracy benchmark. The selected 50-term visualization uses normalized vectors, cosine distance, t-SNE perplexity eight and seeds 42/43/44. Five-neighbor category purity uses manually declared groups. Neighbor retention measures the fraction of original neighbors surviving projection. Categories were selected for illustration, so purity is not an unbiased population estimate.

BERT runs in evaluation mode without gradients. Character offsets select the intended word span, including all WordPiece components; states are pooled by their mean. Layer zero includes token, position and segment embeddings plus normalization. Matched-position controls make input-state equality interpretable. Cosine changes demonstrate context sensitivity but do not directly measure successful sense disambiguation. Similarity need not decrease monotonically through layers.

TinyStories uses verified pretrained weights: every local safetensors tensor equals the corresponding tensor in the pinned original PyTorch release. The 3,745,984 total parameters include embeddings; the model's “1M” name is not its full parameter count. Hugging Face decoding uses explicit EOS/pad IDs, attention masks and token budgets. Greedy token parity is verified for three single, unpadded prompts. This is not a general proof of parity for batched padding, processors or arbitrary library versions.

The custom sampler retains the token that crosses a nucleus threshold and validates temperature/filter bounds. Autoregressive generation stops at EOS, uses a local random generator and exposes a context limit. Custom LM generation uses a sliding 64-token window with reset positions; it is an illustrative continuation mechanism rather than full cached long-context inference. Sample quality is shown with actual text and simple repetition statistics; it is not inferred from likelihood alone. Beam search can find high-scoring sequences within its restricted search without guaranteeing semantic correctness or a global optimum.

## References

- [GloVe paper](https://aclanthology.org/D14-1162/) and [official vector release](https://nlp.stanford.edu/projects/glove/)
- [BERT paper](https://arxiv.org/abs/1810.04805) and [model card](https://huggingface.co/google-bert/bert-base-uncased)
- [TinyStories paper](https://arxiv.org/abs/2305.07759) and [published model](https://huggingface.co/roneneldan/TinyStories-1M)
- [IMDb source dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Hugging Face generation reference](https://huggingface.co/docs/transformers/main_classes/text_generation)
