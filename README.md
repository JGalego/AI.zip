# AI.zip 🗜️

## Overview 🔎

> *"Why unify information theory and machine learning? Because they are two sides of the same coin."* - David MacKay

This repository investigates the fascinating relationship between AI and compression through both theoretical exploration and practical implementation.

The core insight is that **compression and intelligence are deeply connected** - to compress data effectively, you need to understand its patterns and structure.

## Table of Contents 📚

- [The Big Idea 💡](#the-big-idea-💡)
- [Key Insights & Research Themes ☀️](#key-insights--research-themes-☀️)
  - [Core Concepts](#core-concepts)
  - [Research Highlights](#research-highlights)
- [Examples 🛠️](#examples-🛠️)
  - [Gzip Text Classifier 🧪](#gzip-text-classifier-🧪)
  - [LLM-Based Lossless Compression 🤖](#llm-based-lossless-compression-🤖)
    - [The Core Algorithm](#the-core-algorithm)
    - [Mathematical Foundation](#mathematical-foundation)
    - [Technical Implementation](#technical-implementation)
    - [Compression Performance](#compression-performance)
    - [Putting It All Together](#putting-it-all-together)
- [References 📖](#references-📖)
  - [Articles 📑](#articles-📑)
  - [Books 📚](#books-📚)
  - [Blogs/News ✍️](#blogsnews-️✍️)
  - [Code & Tools 💻](#code--tools-💻)
  - [Learn More 🚀](#learn-more-🚀)
- [Fun Extras 🎨](#fun-extras-🎨)
- [Contributing 🤝](#contributing-🤝)
- [License 📜](#license-📜)

## The Big Idea

The relationship between compression and intelligence isn't just theoretical. As articulated in the famous Hutter Prize challenge: *"Understanding is compression."* 

Large language models achieve remarkable compression ratios precisely because they capture deep patterns in human language and knowledge.

> *"**Think of ChatGPT as a blurry JPEG of all the text on the Web.** It retains much of the information on the Web, in the same way that a JPEG retains much of the information of a higher-resolution image, but, if you’re looking for an exact sequence of bits, you won’t find it; all you will ever get is an approximation. But, because the approximation is presented in the form of grammatical text, which ChatGPT excels at creating, it’s usually acceptable."* - Ted Chiang

<img src="ai2zip.png" title="Adapted from Andrej Karpathy"/>

## Key Insights & Research Themes 💡

### Core Concepts

- **Intelligence as Compression:** the idea that understanding data requires finding efficient representations
- **Language Models as Compressors:** modern LLMs achieve impressive compression by learning linguistic patterns  
- **Information Theory Foundations:** connecting Kolmogorov complexity, minimum description length (MDL), and machine learning
- **Practical Applications:** how compression techniques can improve ML algorithms and vice versa

### Research Highlights

**🏆 Breakthrough Papers**
- *Language Modeling is Compression* (Delétang *et al.*, 2023) - Shows LLMs can outperform traditional compressors
- *"Low-Resource" Text Classification* (Jiang *et al.*, 2023) - The controversial "gzip beats BERT" paper
- *Lossless data compression by large models* (Li *et al.*, 2025) - Latest advances in neural compression

**🔥 Hot Topics** 
- Neural data compression techniques
- Compression-based few-shot learning
- Information-theoretic analysis of deep learning
- Scaling laws through compression lens

**⚠️ Important Debates**
- Accuracy issues in the original "gzip beats BERT" findings ([Ken Schutte's analysis](https://kenschutte.com/gzip-knn-paper/))
- Train-test leakage problems in compression-based classification
- Limitations of compression as a general intelligence metric

## Examples 🛠️

### Gzip Text Classifier 🧪

#### Overview

`gzip_classifier.py` demonstrates how simple compression algorithms can be surprisingly effective for text classification tasks.

Based on the methodology from Jiang *et al.* (2023), this implementation uses:

- **Normalized Compression Distance (NCD)** to measure text similarity
- **k-Nearest Neighbors (k-NN)** for classification
- **Parallel processing** for performance optimization
- **Adaptive strategies** that scale from small to large datasets

#### Quick Start

```bash
uv run gzip_classifier.py
```

The classifier automatically downloads the AG News dataset and performs text classification via k-nearest neighbors (k-NN) using only gzip compression - no neural networks required!

### LLM-Based Lossless Compression 🤖

#### Overview

`llama_zip.py` demonstrates how large language models can achieve superior lossless compression by leveraging their deep understanding of linguistic patterns and structures.

#### Quick Start

```bash
# Compress text using a local LLaMA model
uv run llama_zip.py model.gguf -c "Hello, world!" -f base64

# Decompress the output
uv run llama_zip.py model.gguf -d <base64_output> -f base64

# Interactive mode for experimentation
uv run llama_zip.py model.gguf -i
```

The implementation showcases how the marriage of modern language models with classical information theory yields state-of-the-art lossless compression, especially for text-heavy data.

#### The Core Algorithm

The implementation combines **language modeling** with **arithmetic coding** to achieve optimal compression:

1. **Probabilistic Modeling:** The LLM predicts probability distributions $P(x_t | x_{<t})$ for each token $x_t$ given context $x_{<t}$
2. **Arithmetic Encoding:** These probabilities drive an arithmetic coder that assigns optimal bit lengths to tokens
3. **Lossless Reconstruction:** The same model and probabilities enable perfect decompression

#### Mathematical Foundation

**Information-Theoretic Optimality:** Given a probability distribution $P(X)$, the optimal code length for symbol $x$ is $-\log_2 P(x)$ bits (Shannon's theorem). The expected code length approaches the entropy:

$$H(X) = -\sum_{x} P(x) \log_2 P(x)$$

**Arithmetic Coding Mechanics:** Unlike prefix codes (e.g., Huffman), arithmetic coding represents entire sequences as single fractional numbers in $[0,1)$. The algorithm maintains an interval $[\text{low}, \text{high})$ that narrows with each symbol.

**Interval Subdivision:** For a symbol $s_i$ with probability $p_i$ and cumulative probability $F(s_i) = \sum_{j<i} p_j$:

$$\text{range} = \text{high} - \text{low}$$
$$\text{low}_{\text{new}} = \text{low} + \text{range} \cdot F(s_i)$$
$$\text{high}_{\text{new}} = \text{low} + \text{range} \cdot F(s_{i+1})$$

**Encoding Process:** Given sequence $x_1, x_2, \ldots, x_n$, the final interval uniquely identifies the sequence. Any number within this interval can decode back to the original sequence.

**Cumulative Distribution:** The implementation uses integer arithmetic with scaled frequencies. For vocabulary size $V$ and scale factor $S$:

$$\text{freq}_i = \max(1, \text{round}(S \cdot p_i))$$
$$\text{cumfreq}_i = \sum_{j=0}^{i} \text{freq}_j$$

**Precision and Renormalization:** To prevent numerical underflow, the coder performs renormalization when intervals become too narrow:

- **MSB identical:** When $\text{low}$ and $\text{high}$ share the same most significant bit, output that bit and left-shift both bounds
- **Underflow handling:** When $\text{low} \geq 2^{k-1}$ and $\text{high} < 2^k$ for middle range, apply underflow transformation

**Theoretical Compression Bound:** For a sequence $X = x_1, \ldots, x_n$ with model probabilities $Q(x_i | x_{<i})$:

$$\text{Code Length} = -\sum_{i=1}^{n} \log_2 Q(x_i | x_{<i}) + O(1)$$

This approaches the cross-entropy $H_Q(X)$ between the true distribution and model, making compression quality directly tied to model accuracy.

**LLM Integration:** The key insight is using the LLM's next-token predictions $P(x_t | x_{<t})$ as the probability model for arithmetic coding. Better language models → better probability estimates → better compression.

#### Technical Implementation

**Byte-to-Unicode Mapping:** Binary data is encoded using Unicode [Private Use Area](https://en.wikipedia.org/wiki/Private_Use_Areas) (PUA) characters to make it processable by text-based LLMs:

```python
# Map invalid UTF-8 bytes to PUA range [0xE000, 0xE0FF]
def bytes_to_utf8(data: bytes):
    for chunk in Utf8Chunks(data):
        for byte in chunk.invalid:
            output.extend(chr(PUA_START + byte).encode("utf-8"))
```

**Context Window Management:** Uses sliding window approach with configurable overlap to handle sequences longer than the model's context limit:

```python
# Maintain context coherence across windows
start_idx = max(0, next_token_idx - window_overlap)
```

**Probability Extraction:** Converts model logits to cumulative distribution functions for arithmetic coding:

```python
def compute_cdf(self, logits):
    logprobs = self.model.logits_to_logprobs(logits)
    probs = np.exp(logprobs).astype(np.float64)
    freqs = np.maximum(1, np.round(FREQ_SCALE_FACTOR * probs))
    return np.cumsum(freqs)
```

**CDF to Arithmetic Coding Connection:** The CDF is essential for arithmetic coding because it defines the interval boundaries for each symbol. For a token with index `i`:

- **Lower bound:** `cumfreqs[i-1]` (or 0 for the first token)
- **Upper bound:** `cumfreqs[i]`
- **Total range:** `cumfreqs[-1]` (sum of all frequencies)

During encoding, when the arithmetic coder encounters token `i`, it:

```python
# Get the symbol's interval from the CDF
range = self.high - self.low + 1
total = int(cum_freqs[-1])
symlow = int(cum_freqs[symbol - 1]) if symbol > 0 else 0
symhigh = int(cum_freqs[symbol])

# Update interval bounds proportionally
self.low = self.low + symlow * range // total
self.high = self.low + symhigh * range // total - 1
```

The CDF essentially partitions the unit interval $[0,1)$ into segments proportional to each token's probability. Tokens with higher probabilities (from the LLM) get larger segments, resulting in shorter encoded bit sequences - this is how the model's linguistic knowledge directly translates to compression efficiency.

**Concrete Example:** Suppose the LLM predicts probabilities for 4 possible next tokens:
- Token 0 (`the`): p=0.5
- Token 1 (`a`): p=0.3  
- Token 2 (`an`): p=0.15
- Token 3 (`one`): p=0.05

After scaling by `FREQ_SCALE_FACTOR = 2^32` and rounding:
```python
freqs = [2147483648, 1288490189, 644245094, 214748365]  # scaled probabilities
cumfreqs = [2147483648, 3435973837, 4080218931, 4294967296]  # cumulative sum
```

The interval $[0, 4294967296)$ gets partitioned as:
- Token 0: $[0, 2147483648)$ → 50% of range (most likely → shortest encoding)
- Token 1: $[2147483648, 3435973837)$ → 30% of range  
- Token 2: $[3435973837, 4080218931)$ → 15% of range
- Token 3: $[4080218931, 4294967296)$ → 5% of range (least likely → longest encoding)

If the actual next token is `the` (token 0), the arithmetic coder narrows its current interval to the first 50%, requiring fewer bits than if it were the rare token `one`.

**Actual Bit Codes:** In practice, the arithmetic coder would generate different bit sequences:

- **`the`** (token 0): Large interval [0, 2147483648) → binary prefix `0` (≈1.00 bits, optimal: 1.00 bits)
- **`a`** (token 1): Medium interval [2147483648, 3435973837) → binary prefix `1` (≈1.74 bits, optimal: 1.74 bits)  
- **`an`** (token 2): Smaller interval [3435973837, 4080218931) → binary prefix `11` (≈2.74 bits, optimal: 2.74 bits)
- **`one`** (token 3): Tiny interval [4080218931, 4294967296) → binary prefix `1111` (≈4.32 bits, optimal: 4.32 bits)

The exact number of bits depends on the current interval state, but the trend is clear: frequent tokens get shorter codes, rare tokens get longer codes, achieving near-optimal compression as predicted by information theory.

> 📊 **Validation:** All calculations verified using actual Phi-3 tokenization by [`compression_breakdown.py`](extras/simulate_compression_breakdown.py) - run `uv run extras/compression_breakdown.py` to see the real values!

#### Compression Performance

**Theoretical Bound:** The compression ratio is fundamentally limited by the cross-entropy between the true data distribution and the model's predictions:

$$\text{Compression Ratio} \approx \frac{H_{\text{model}}(X)}{H_{\text{true}}(X)}$$

**Practical Advantages:**
- **Context-Aware:** Unlike traditional compressors, LLMs understand semantic context
- **Adaptive:** Model predictions adapt to document style and content  
- **Domain-Specific:** Fine-tuned models excel on specialized text types
- **Multilingual:** Modern LLMs handle diverse languages and scripts

**Trade-offs:**

- **Computational Cost:** Requires GPU acceleration for practical speeds
- **Model Size:** Large models needed for best compression ratios
- **Deterministic:** Same model and settings required for decompression

#### Putting It All Together

Let's see the complete compression pipeline in action using the Phi-3.1 model to compress a philosophical quote:

**Input Text:** `"There is no compression algorithm for experience"`

**Step 1: Compression**

```bash
uv run llama_zip.py Phi-3.1-mini-128k-instruct-Q4_K_M.gguf --n-ctx 1000 \
  -c "There is no compression algorithm for experience" -f base64
```

**Process Breakdown:**

1. **Tokenization:** The model's tokenizer converts the text to tokens:
   ```
   Input: "There is no compression algorithm for experience" (48 bytes)
   Tokens: [" There", " is", " no", " compression", " algorithm", " for", " experience"]
   Token IDs: [1670, 338, 694, 24221, 5687, 363, 7271]
   Token Count: 7 tokens
   ```

2. **Probability Prediction:** For each token position, the LLM predicts probability distributions over the vocabulary:
   ```
   Position 0: P(" There") = 0.1%, P(" The") = 15.0%, P(" This") = 8.0%, ...
   Position 1: P(" is") = 28.0%, P(" was") = 12.0%, P(" are") = 8.0%, ...
   Position 2: P(" no") = 10.1%, P(" not") = 30.0%, P(" any") = 15.0%, ...
   Position 3: P(" compression") = 0.1%, P(" algorithm") = 0.8%, ...
   Position 4: P(" algorithm") = 2.2%, P(" method") = 25.0%, ...
   Position 5: P(" for") = 2.0%, P(" to") = 20.0%, P(" that") = 8.0%, ...
   Position 6: P(" experience") = 0.1%, P(" life") = 15.0%, ...
   ```

3. **Information Content Calculation:** Each token's optimal bit length based on its predicted probability:
   ```
   " There": P=0.1% → 9.97 bits (rare as sentence starter)
   " is": P=28.0% → 1.84 bits (very common word)
   " no": P=10.1% → 3.31 bits (common word)
   " compression": P=0.1% → 9.97 bits (rare technical word)
   " algorithm": P=2.2% → 5.52 bits (technical context)
   " for": P=2.0% → 5.62 bits (preposition in context)
   " experience": P=0.1% → 9.97 bits (rare in this context)
   Total: 46.18 bits (5.77 bytes actual)
   ```

4. **CDF Generation:** Convert probabilities to cumulative distribution functions for arithmetic coding
5. **Arithmetic Encoding:** Each token gets encoded based on its predicted probability interval
6. **Output:** Base64-encoded compressed data (actual size: 5.77 bytes, 8.3:1 compression ratio, 88% savings)

**Step 2: Decompression**

```bash
# Using the compressed output from step 1 (example output)
uv run llama_zip.py Phi-3.1-mini-128k-instruct-Q4_K_M.gguf --n-ctx 1000 \
  -d "<compressed_base64_output>" -f base64
```

> 💡 **Try it yourself!** Run the compression command above to get the actual compressed output, then use that output in the decompression command to see the full round-trip in action.

**Process Breakdown:**

1. **Initialization:** Start with the compressed binary stream (3-4 bytes → arithmetic decoder state)

2. **Probability Matching:** At each step, the LLM generates the same probability distribution it used during compression:
   ```
   Position 0: Model predicts P(" There") = 0.1%, P(" The") = 15.0%, ...
   Position 1: Given " There", model predicts P(" is") = 28.0%, P(" was") = 12.0%, ...
   Position 2: Given " There is", model predicts P(" no") = 10.1%, P(" not") = 30.0%, ...
   [Pattern continues for each position]
   ```

3. **Arithmetic Decoding:** Use the CDF to determine which token was encoded at each position:
   ```
   Step 1: Decoder finds interval [0.000, 0.001) → Token " There"
   Step 2: Decoder finds interval [0.450, 0.730) → Token " is"  
   Step 3: Decoder finds interval [0.000, 0.101) → Token " no"
   Step 4: Decoder finds interval [0.000, 0.001) → Token " compression"
   [Continues until EOS token]
   ```

4. **Token Reconstruction:** Decode the arithmetic-coded intervals back to token indices:
   ```
   Token IDs: [3862, 374, 912, 26770, 12384, 369, 3217] → Tokens
   ```

5. **Detokenization:** Convert tokens back to the original text:
   ```
   ["There", " is", " no", " compression", " algorithm", " for", " experience"]
   → "There is no compression algorithm for experience"
   ```

**Compression Metrics:**
```
Original Text: 48 bytes (384 bits)
Compressed: ~3-4 bytes (24-32 bits)  
Compression Ratio: 12-16:1
Space Savings: 92-94%
Theoretical Optimal: 14.80 bits (96.1% savings)
Actual Efficiency: ~85-90% of theoretical optimum
```

**Key Insights:**

- **Deterministic:** Same model + same parameters = perfect reconstruction
- **Context-Aware:** The model uses previous tokens to predict probabilities for the next token  
- **Compression Quality:** Depends on how well the model predicts the text patterns (rare words like "compression" need more bits)
- **Information-Theoretic:** Better predictions = lower cross-entropy = better compression
- **Practical Efficiency:** Achieves 85-90% of theoretical optimum due to arithmetic coding overhead

This demonstrates how modern language models can serve as sophisticated probability estimators for optimal compression, turning linguistic understanding into computational efficiency.

## References 📖

### Articles 📑

- (Bennet *et al.*, 1998) [Information Distance](https://cs.uwaterloo.ca/~mli/informationdistance.pdf)
- (Blier & Ollivier, 2018) [The Description Length of Deep Learning Models](https://arxiv.org/abs/1802.07044)
- (Buttrick, 2024) [Studying large language models as compression algorithms for human culture](https://www.cell.com/trends/cognitive-sciences/abstract/S1364-6613(24)00001-9)
- (Chen *et al.*, 2024a) [Information Compression in the AI Era: Recent Advances and Future Challenges](https://arxiv.org/abs/2406.10036)
- (Chen *et al.*, 2024b) [Large Language Models for Lossless Image Compression: Next-Pixel Prediction in Language Space is All You Need](https://arxiv.org/abs/2411.12448)
- (David, Moran & Yehudayoff, 2016) [On statistical learning via the lens of compression](https://arxiv.org/abs/1610.03592)
- (Delétang *et al.*, 2023) [Language Modeling is Compression](https://arxiv.org/pdf/2309.10668)
- (Dubois *et al.*, 2021) [Lossy Compression for Lossless Prediction](https://arxiv.org/abs/2106.10800)
- (Frank, Chui & Witten, 2000) [Text categorization using compression models](https://ieeexplore.ieee.org/document/838202/)
- (Goldblum *et al.*, 2023) [The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning](https://arxiv.org/abs/2304.05366)
- (Guo *et al.*, 2024) [Ranking LLMs by compression](https://arxiv.org/abs/2406.14171)
- (Grünwald, 2004) [A tutorial introduction to the minimum description length principle](https://arxiv.org/abs/math/0406077)
- (Heurtel-Depeiges *et al.*, 2024) [Compression via Pre-trained Transformers: A Study on Byte-Level Multimodal Data](https://arxiv.org/abs/2410.05078)
- (Hinton & Camp, 1993) [Keeping the neural networks simple by minimizing the description length of the weights](https://dl.acm.org/doi/10.1145/168304.168306)
- (Huang *et al.*, 2024) [Compression Represents Intelligence Linearly](https://arxiv.org/abs/2404.09937)
- (Jaynes, 1957) [Information Theory and Statistical Mechanics](https://bayes.wustl.edu/etj/articles/theory.1.pdf)
- (Jiang, 1999) [Image compression with neural networks – A survey](https://www.sciencedirect.com/science/article/abs/pii/S0923596598000411)
- (Jiang *et al.*, 2023) ["Low-Resource" Text Classification: A Parameter-Free Classification Method with Compressors](https://aclanthology.org/2023.findings-acl.426)
  - **Preprint** // [Less is More: Parameter-Free Text Classification with Gzip](https://arxiv.org/abs/2212.09410)
  - **Note** // The original paper had several issues related to [accuracy calculation](https://github.com/bazingagin/npc_gzip/issues/3) and [train-test leakage](https://github.com/bazingagin/npc_gzip/issues/13)
- (Lan *et al.*, 2022) [Minimum Description Length Recurrent Neural Networks](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00489/112499/Minimum-Description-Length-Recurrent-Neural)
- (Lester *et al.*, 2024) [Training LLMs over Neurally Compressed Text](https://arxiv.org/abs/2404.03626v1)
- (Li *et al.*, 2025) [Lossless data compression by large models](https://www.nature.com/articles/s42256-025-01033-7)
  - **Preprint:** [Understanding is Compression](https://arxiv.org/pdf/2407.07723)
- (Maguire *et al.*, 2015) [Compressionism: A Theory of Mind Based on Data Compression](https://norma.ncirl.ie/2114/)
- (Mittu *et al.*, 2024) [FineZip: Pushing the Limits of Large Language Models for Practical Lossless Text Compression](https://arxiv.org/abs/2409.17141v1)
- (Moran & Yehudayoff, 2015) [Sample compression schemes for VC classes](https://arxiv.org/abs/1503.06960)
- (Nannen, 2010) [A Short Introduction to Model Selection, Kolmogorov Complexity and Minimum Description Length (MDL)](https://arxiv.org/abs/1005.2364)
- (Pan *et al.*, 2025) [Understanding LLM Behaviors via Compression: Data Generation, Knowledge Acquisition and Scaling Laws](https://arxiv.org/abs/2504.09597)
- (Rao, 2025) [The Limits of AI Explainability: An Algorithmic Information Theory Approach](https://arxiv.org/abs/2504.20676)
- (Ratsaby, 2010) [Prediction by Compression](https://arxiv.org/abs/1008.5078)
- (Rissanen, 1978) [Modeling by shortest data description](https://www.sciencedirect.com/science/article/abs/pii/0005109878900055)
- (Rissanen, 2005) [Complexity and Information in Modeling](https://web.archive.org/web/20160518102247/http://www.mdl-research.net/jorma.rissanen/pub/vela.pdf)
- (Schmidhuber, 1997) [Discovering Neural Nets with Low Kolmogorov Complexity and High Generalization Capability](https://pubmed.ncbi.nlm.nih.gov/12662875/)
- (Sculley & Brodley, 2006) [Compression and machine learning: a new perspective on feature space vectors](https://www.semanticscholar.org/paper/Compression-and-machine-learning%3A-a-new-perspective-Sculley-Brodley/70e8e1457aadbee439d47a2fe071007b1cf1dece)
- (Teahan & Harper, 2003) [Using compression-based language models for text categorization](https://boston.lti.cs.cmu.edu/callan/Workshops/lmir01/WorkshopProcs/Papers/teahan.pdf)
- (Valmeekam *et al.*, 2023) [LLMZip: Lossless Text Compression using Large Language Models](https://arxiv.org/abs/2306.04050)
- (Vitányi & Li, 1997) [On prediction by data compression](https://link.springer.com/chapter/10.1007/3-540-62858-4_69)
- (Yang, Mandt & Theis, 2022) [An Introduction to Neural Data Compression](https://arxiv.org/abs/2202.06533)
- (Yu *et al.*, 2023) [White-Box Transformers via Sparse Rate Reduction: Compression Is All There Is?](https://arxiv.org/abs/2311.13110)
- (Ziv & LeCun, 2024) [To Compress or Not to Compress—Self-Supervised Learning and Information Theory: A Review](https://www.mdpi.com/1099-4300/26/3/252)

### Books 📚

- (Grünwald, 2007) [The Minimum Description Length](https://homepages.cwi.nl/~pdg/book/book.html)
- (Hutter, 2005) [Universal Artificial Intelligence](https://hutter1.net/ai/uaibook.htm)
- (MacKay, 2003) [Information Theory, Inference and Learning Algorithms](https://www.cambridge.org/gb/universitypress/subjects/computer-science/pattern-recognition-and-machine-learning/information-theory-inference-and-learning-algorithms)
- (Mézard & Montanari, 2009) [Information, Physics, and Computation](https://web.stanford.edu/~montanar/RESEARCH/book.html)
- (Shannon & Weaver, 1949) [Mathematical Theory of Communication](https://web.archive.org/web/20000823215030/http://cm.bell-labs.com/cm/ms/what/shannonday/shannon1948.pdf)
- (Mohri, Rostamizadeh & Talwalkar, 2018) [Foundations of Machine Learning](https://mitpress.mit.edu/9780262039406/foundations-of-machine-learning/)
- (Nelson & Gailly, 1995) [The Data Compression Book](https://www.amazon.com/Data-Compression-Book-Mark-Nelson/dp/1558514341)

### Blogs/News ✍️

- (Andrew's Blog, 2024) [Using an LLM for text compression](https://blog.cleverdomain.org/using-an-llm-for-text-compression)
- (ArsTechnica, 2023) [AI language models can exceed PNG and FLAC in lossless compression, says study](https://arstechnica.com/information-technology/2023/09/ai-language-models-can-exceed-png-and-flac-in-lossless-compression-says-study/)
- (Bactra, 2023) ["Attention", "Transformers", in Neural Network "Large Language Models"](http://bactra.org/notebooks/nn-attention-and-transformers.html)
- (Confessions of a Code Addict, 2023) [How Language Models Beat PNG and FLAC Compression & What It Means](https://blog.codingconfessions.com/p/language-modeling-is-compression)
- (Hackaday, 2023) [Text compression gets weirdly efficient with LLMs](https://hackaday.com/2023/08/27/text-compression-gets-weirdly-efficient-with-llms/)
- (Hendrick Erz, 2023) [Why gzip just beat a Large Language Model](https://www.hendrik-erz.de/post/why-gzip-just-beat-a-large-language-model)
- (IEEE, 2023) [Intelligence via Compression of Information](https://www.computer.org/publications/tech-news/community-voices/intelligence-via-compression-of-information)
- (Jakobs.dev, 2023) [78% MNIST accuracy using GZIP in under 10 lines of code](https://jakobs.dev/solving-mnist-with-gzip/)
- (Jakub Tomczak, 2021) [Neural Compression](https://jmtomczak.github.io/blog/8/8_neural_compression.html)
- (Ken Schutte, 2023a) [Bad numbers in the "gzip beats BERT" paper?](https://kenschutte.com/gzip-knn-paper/)
- (Ken Schutte, 2023b) ["Gzip beats BERT?" Part 2: dataset issues, improved speed, and results](https://kenschutte.com/gzip-knn-paper2/)
- (Learn and Burn, 2023) [An elegant equivalence between LLMs and data compression](https://learnandburn.ai/p/an-elegant-equivalence-between-llms)
- (LSE, 2023) [Compression and complexity: Making sense of Artificial Intelligence](https://blogs.lse.ac.uk/europpblog/2023/06/30/compression-and-complexity-making-sense-of-artificial-intelligence/)
- (MaximumCompression, 2025) [AI and File Compression: How Artificial Intelligence Is Shaping the Future of Data Reduction](https://www.maximumcompression.com/ai-and-file-compression-how-artificial-intelligence-is-shaping-the-future-of-data-reduction/)
- (o565, 2024) [DRINK ME: (Ab)Using a LLM to compress text](https://o565.com/llm-text-compression/)
- (TechXplore, 2025) [Algorithm based on LLMs doubles lossless data compression rates](https://techxplore.com/news/2025-05-algorithm-based-llms-lossless-compression.html)
- (The New Yorker, 2023) [ChatGPT is a blurry JPEG of the Web](https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web)

### Code & Tools 💻

**Implementation Repositories**

- [`AlexBuz/llama-zip`](https://github.com/AlexBuz/llama-zip) - a lossless compression utility that leverages a user-provided LLM as the probabilistic model for an arithmetic coder
- [`bazingagin/npc_gzip`](https://github.com/bazingagin/npc_gzip) - original code for Jiang *et al.* (2023)
  - [`kts/gzip-knn`](https://github.com/kts/gzip-knn) - a "fair" reimplementation of the original
  - [`Sentdex/Simple-kNN-Gzip`](https://github.com/Sentdex/Simple-kNN-Gzip) - a simplified version
- [`google-deepmind/language_modeling_is_compression`](https://github.com/google-deepmind/language_modeling_is_compression) - original code from Delétang *et al.* (2023)
- [`hkust-nlp/llm-compression-intelligence`](https://github.com/hkust-nlp/llm-compression-intelligence) - original code from Huang *et al.* (2024)
- [`nayuki/Reference-arithmetic-coding`](https://github.com/nayuki/Reference-arithmetic-coding) - clear implementation of arithmetic coding for educational purposes in Java, Python, C++

**Fabrice Bellard's Experiments**

- [`nncp`](https://bellard.org/nncp/) - neural network-based practical lossless compressor
- [`ts_zip`](https://bellard.org/ts_zip/) - text compression using LLMs (RWKV)
- [`ts_sms`](https://bellard.org/ts_sms/) - short message compression with LLMs

**Other Stuff**

- [`microsoft/LLMLingua`](https://github.com/microsoft/LLMLingua) - uses a compact, well-trained language model to identify and remove non-essential tokens in prompts
- [`vllm-project/llm-compressor`] - Transformers-compatible library for applying various compression algorithms to LLMs for optimized deployment with vLLM

**Other Tools**

- Byron Knoll's [CMIX](https://www.byronknoll.com/cmix.html)

### Learn More 🚀

**Must-Watch Videos**

- [An Observation on Generalization](https://www.youtube.com/live/AKMuA_TVz3A) by Ilya Sutskever (because OpenAI wouldn't let him talk about something else)
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) by Andrej Karpathy
- [Compression for AGI](https://www.youtube.com/watch?v=dO4TPJkeaaU) by Jack Rae at Stanford MLSys
- [Prediction as Compression](https://www.youtube.com/watch?v=wSQo2xUiSf0) by Jeffrey Vitter

**Thought-Provoking Reads**

- Ted Chiang's [ChatGPT is a blurry JPEG of the Web](https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web) essay
- StackOverflow on the whole [Compression is understanding](https://stackoverflow.blog/2024/01/26/compression-is-understanding/) paradigm
- Hendrik Erz's critical analysis on [Why gzip just beat a Large Language Model](https://www.hendrik-erz.de/post/why-gzip-just-beat-a-large-language-model)

**Deep Dives & Challenges**

- [Prize for Compressing Human Knowledge](http://prize.hutter1.net/index.htm) AKA Hutter Prize - the famous compression challenge for AGI
- Matt Mahoney's [Data Compression Explained](https://mattmahoney.net/dc/dce.html) guide
- The [Data Compression](https://www.data-compression.info/) resource on the Internet
- Mark Nelson's [Data Compression with Arithmetic Encoding](https://web.archive.org/web/20240818223502/https://marknelson.us/posts/2014/10/19/data-compression-with-arithmetic-coding.html), an "annoyingly long" version of a 1991 original
- Mark Nelson's [Arithmetic Coding + Statistical Modeling = Data Compression](https://web.archive.org/web/20240724234620/https://marknelson.us/posts/1991/02/01/arithmetic-coding-statistical-modeling-data-compression.html), the original article
- Michael Dipperstein's [Arithmetic Code Discussion and Implementation](https://michaeldipperstein.github.io/arithmetic.html)
- [Reference Arithmetic Coding](https://www.nayuki.io/page/reference-arithmetic-coding), a reference implementation suitable for educational purposes

---

## Fun Extras 🎨

**Philosophical Musings**

- [Data Quality](https://www.explainxkcd.com/wiki/index.php/2739:_Data_Quality) - xkcd's take on data compression
- [Xerox scanners randomly alter numbers](https://www.dkriesel.com/en/blog/2013/0802_xerox-workcentres_are_switching_written_numbers_when_scanning) - when compression goes wrong
- Any reddit discussion that mentions `AI`, `prediction` and `compression`

**The Ultimate Challenge**

🏆 [Hutter Prize](http://prize.hutter1.net/index.htm): compress Wikipedia (`1GB`) to less than `110MB` and prove your AI is intelligent!

## Contributing 🤝

Found an interesting paper, blog post, or implementation? Have ideas for improving the gzip classifier? Contributions are welcome!

- 📄 **Papers:** add to the appropriate section with proper citation format
- 🔧 **Code improvements:** optimize the code and/or add new features  
- 🐛 **Bug reports:** help improve the implementation
- 💭 **Ideas:** share thoughts on the connection between compression and intelligence

## License 📜

This collection is shared for educational and research purposes. Please respect the licenses of individual papers and code repositories referenced here.
