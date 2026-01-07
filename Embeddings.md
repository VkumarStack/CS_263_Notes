# Embeddings
- **Distributional Hypothesis**: Words in similar contexts tend to have similar meanings
  - Basis for natural language processing
- **Embeddings**: Vector representations of the meanings of words, learned directly from word distributions
## Lexical Semantics
- Consider the various basic principles of word meaning (**lexical semantics**), all of which must be considered in an effective embedding representation
- **Lemma (Citation Form)**: Think of the 'general' form of a word. More specific forms are known as **wordforms**
  - e.g. Lemma - 'Mouse', Wordform - 'Mouse', 'Mice'
- Lemmas can have multiple *meanings* (**polysemous**) - each aspect of a meaning is a **word sense**
  - e.g. 'Mouse' (Rodent) versus 'Mouse' (Cursor Controller)
- If a sense of a word has a meaning identical to the sense of another word, the two words are **synonyms**
  - In truth, synonyms are not always identical - **principal of contrast**: difference in linguistic form is always associated with some difference in meaning
    - e.g. 'Water' is casual, '$H_2O$' is scientific
- Words can be **similar** without necessarily being synonyms
  - e.g. 'Cat' and 'Dog' are similar
- Aside from similarity, there is **relatedness**
  - e.g. 'Coffee' is not similar to 'Cup', but they are related (drink coffee in a cup)
  - One type of relatedness is belonging in the same **semantic field** - a set of words that cover a specific semantic domain
- **Connotation**: Aspect of a word's meaning related to the writer or reader's emotions
  - e.g. 'Fake' (negative) and 'Replica' (positive) mean the same thing, but have different connotations
  - Connotation based on positivty or negativity is known as **sentiment**
## Vector Semantics
- **Vector Semantics** try to define the meaning of a word by its *distribution* in language use - words with similar distributions likely have similar meanings
## Count-Based Embeddings
- The simplest vector embedding approach is utilizing a **word-context matrix**, which is a $|V| \times |V|$ matrix where each cell records the times the row (target) word and column (context) words co-occur in some corpus
  - Usually, a window of words is set to quantify this co-occurrence
- Each row in this type of matrix is a vector of dimensionality $|V|$
  - Most entries in the matrix are zero, so the representation is very *sparse* - this leads to inefficiency due to the sheer size and sparsity of the matrix
## Cosine for Measuring Similarity
- Similarity between two vectors can be measured using cosine similarity - dot product alone does not suffice since it biases long vectors (more frequent words) - so normalization is necessary
  - $\cos(\theta) = \frac{\bold{a} \cdot \bold{b}}{|\bold{a}||\bold{b}|}$
    - $= \frac{\sum_{i = 1}^N v_i w_i}{\sqrt{\sum_{i=1}^N v_i^2} \sqrt{\sum_{i=1}^N w_i^2}}$
## Word2Vec
- A better approach for NLP is leveraging **dense embeddings**, where word vectors are of dimension $d << |V|$
  - Dense vectors do a better job at capturing lexical and semantic information
- One classic method for computing dense embeddings is **skip-gram with negative sampling (SGNS)**, also known as **word2vec**
- The main idea is to:
  - Treat the target word and a neighboring context word as *positive examples*
  - Randomly sample other words in the lexicon to get *negative samples*
  - Use logistic regression to train a classifier to distinguish the two cases
  - Use the learned weights as embeddings
- Two embeddings are actually stored for each word: one for the word as a target (**W**), and another for the word considered as context (**C**)
  - These are $2|V| \times d$ parameters
- To learn, the algorithm takes as inpout a corpus of text and then performs self-supervised learning by going through each word and then using the nearby window words to construct the positive examples while randomly sampling noise words (any word that is *not* the positive example words)
  - e.g. "... lemon, a [tablespoon of **apricot** jam, a] pinch ..."
    - Positive examples: (apricot, tablespoon), (apricot, of), (apricot, jam), (apricot, a)
  - Negative examples are chosen according to their weighted unigram probability, though in practice the weight is set to $\alpha = 0.75$ to give rare words slightly higher probability
    - $P_{\alpha} = \frac{\text{count}(w)^{\alpha}}{\sum_{w'} \text{count}(w')^{\alpha}}$
  - Loss Function: $L(w, c_{pos}, c_{neg*}) = -[\log \sigma(c_{pos} \cdot w) + \sum_{i = 1}^k \log \sigma (-c_{neg_i} \cdot w)]$
    - *k* is the number of negatively sampled non-neighbor words
    - This is essentially *maximizing* the dot product of the word with the actual context word while *minimizing* the dot product of the dot products of the word with any negative words
  - Train with stochastic gradient descent (derivatives for $c_{pos}$, $c_{neg_i}$, and $w$ are pretty easy to find)
## Semantic Properties of Embeddings
- With vector semantic models, the size of the context window can actually influences the type of representation generated
  - Shorter context windows lead to more syntactic representations (since information is from immediate words) while larger context windows lead to more topical representations
- Vector semantic models can be used for **analogies** - though usually dense models are more efficient at this
  - e.g. king - man + woman ~ queen
  - Formally: $\bold{\hat{b}^*} = \argmin_x \text{distance}(\bold{x}, \bold{b} - \bold{a} + \bold{a^*})$
    - Make sure to ignore morphological variants, as these are typically closest with algorithms like word2vec (e.g. cherry:red :: potato:x should exclude 'potato' and 'potatoes')
- Embeddings can be used to study how meaning changes over time by computing multiple embedding spaces (each with texts from a particular time period)
## Bias and Embeddings
- Embedding models capture and amplify the biases of the texts that they were trained off of - e.g. gender and racial stereotypes
## Evaluating Embedding Models
- One common evaluation is through **similarity** performance, comparing the correlation between an algorithm's word similarity scores and the word similarity score ratings assigned by humans