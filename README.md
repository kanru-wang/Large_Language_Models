# Large Language Models

<img src="image/image012.png" width="800"/>

## LLM Usage

#### Providing examples inside the context window is called in-context learning

<img src="image/image002.png" width="600"/>

#### Chain-of-Thought Prompting (One Shot or Few Shot) can help LLMs reason

<img src="image/image015.png" width="650"/>

#### Program-Aided Language models (PAL) (One Shot)

The LLM doesn't really have to decide to run the Python code, it just has to write the script which the orchestrator then passes to the external Python interpreter to run.

<img src="image/image016.png" width="600"/>

#### ReAct framework

- There is an API to query external data.
- The structure of the prompt: (1) Instructions, (2) ReAct example, (3) Question you want to ask.
- The completion contains multiple sets of Thought-Action-Observation trio.
  - Thoughts are what the LLM generates to reason about the current situation
  - The three allowed Actions (a limit imposed by the Instruction) are as follows. Notice that in the Instruction, the Action is formatted using square brackets so that the model will format its completions in the same way.
    - Search, which retrieves external data (a paragraph) about a particular topic
    - Lookup, which searches the next sentence containing the keyword in the current paragraph
    - Finish, which is the conclusion the model reaches
  - Observations are the new information gained from the external search and brought into the context.

<img src="image/image013.jpg" width="500"/>

## Model Architectures and Pre-training Objectives

<img src="image/image031.png" width="650"/>

#### Encoder-only

- Encoder-only models (a.k.a. Autoencoding models) (e.g. BERT and RoBERTa) are pre-trained using masked language modeling. 
- Tokens in the input sequence are randomly masked, and the training objective is to predict the masked tokens in order to reconstruct the original sentence. 
- The bi-directional representations of the input sequence allows the model to have an understanding of the full context of a token, and not just of the words that come before. 
- Encoder-only models are suited to tasks that benefit from bi-directional contexts. E.g. sentence classification or NER.

#### Decoder-only

- Decoder-only models (a.k.a. Autoregressive models) (e.g. GPT and BLOOM) are pre-trained using causal language modeling. 
- The training objective is to predict the next token based on the previous sequence of tokens. The model masks the input sequence, and can only see the input tokens leading up to the token in question. The model then iterates over the input sequence one by one to predict the following token. In contrast to the encoder architecture, this means that the context is unidirectional. 
- Decoder-only models are used for text generation, although large decoder-only models show strong zero-shot inference abilities, and can perform a range of tasks well.

#### Encoder-decoder

- Encoder-decoder models (a.k.a. Sequence-to-Sequence models) (e.g. T5 and BART) have pre-training objective that vary from model to model.
- For example, T5 pre-trains the encoder using span corruption, which masks random sequences of input tokens, and replaces them with a unique Sentinel token, shown here as x. Sentinel tokens are special tokens that do not correspond to any actual word. The decoder is then tasked with reconstructing the masked token sequences auto-regressively. The output is the Sentinel token followed by the predicted tokens.
- Use sequence-to-sequence models for translation, summarization, and question-answering. They are useful when you have a body of texts as both input and output.

<img src="image/image043.png" width="500"/>

## GPU Memory Limit


1B parameters takes about 4GB (32-bit full precision) GPU RAM. However to train a model with 1B parameters, need about 20 times the amount of GPU RAM that the model weights alone take up, i.e. 4 x 20 = 80GB.

#### Quantization: INT8

<img src="image/image008.png" width="600"/>

#### Quantization: FP16

<img src="image/image042.png" width="600"/>

#### Quantization: BFLOAT16

<img src="image/image004.png" width="600"/>

## Data Parallelism vs Model Sharding

(May be similar to https://leimao.github.io/blog/Data-Parallelism-vs-Model-Paralelism/)

### Distributed Data Parallel (DDP)

DDP requires that the model weights, and additional parameters, gradients, and optimizer states to be fit onto a single GPU. If they are too large, should use model sharding.

<img src="image/image007.png" width="600"/>

### Model Sharding

#### Zero Redundancy Optimizer (ZeRO)

- Reduces memory by distributing (sharding) the model parameters, gradients, and optimizer states across GPUs.
- There are 3 stages of ZeRO, each requires less memory.

<img src="image/image046.png" width="650"/>

