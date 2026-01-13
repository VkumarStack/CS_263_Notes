# Large Language Models
## Language Representations
- **Word-Level Model**: Assumes a fixed vocabulary of tens of thousands of word, built from teh training set
  - Any novel words not seen at test time are mapped to a single `UNK` token
- **Sub-Word Model**: Break up words into sub-word units, enabling for the composition of potentially unknown words at test time
  - Common words end up part of a subword vocabulary while rare words are split into components
  - **Byte Pair Encoding (BPE)**: Learn a vocabulary of subword tokens, and each word is split into a sequence of known subwords at training and testing time
    - Start with a vocabulary containing only characters and an end-of-word symbol
    - Using a (large) corpus of text, find most common adjacent characters "a, b", and add "ab" as a subword
    - Replace instances of character pair with new subword, and repeat until desired vocabulary size
## Pretraining
- Language models can learn in a semi-supervised manner off of a large corpus of unannotated text by masking
- This **pretraining** step involves scaling a language model in this unsupervised fashion, where the model acquires a large amount of parametric knowledge (e.g. how the world works) from the sheer amount of data
  - **Post-training** involves generalizing a model to a down-stream task via additional finetuning
- **Bidirectional Models (e.g. BERT)**: Encoder models that are conditioned on both the past and future context - useful for tasks like classification
  - Goal is to generate strong representations of the input
  - One way to train for this is to *mask out* certain input words, and then try to predict those masked words
    - Usually, mask out $k = 15%$ of the words, and only add loss terms from words that are masked out
    - Specifically:
      - 80% of the time replace the input word with `[MASK]` token
      - 10% of the time replace the input word with a random word from the vocabulary
      - 10% of the time leave the input word unchanged
      - Doing this allows for more robustness to noise
  - Another sort of training involves learning relationships between sentences, where the model (BERT) predicts whether sentence B is actual sentence that proceeds sentence A, or just a random sentence
    - The goal here is to improve capabilities for capturing long-range dependencies
- **Unidirectional Models (e.g. GPT)**: Decoder models that are conditioned on just the past context - useful for text generation
- **Encoder-Decoders**: Encoder benefits from bidirectional context, and decoder is used to train whole model through language modeling
  - To train, replace different-length spans from input with unique placeholders, and decode out the spans that were removed
## In-Context Learning
- With **few-shot learning**, the ordering of the example prompts can actually significantly influence the result