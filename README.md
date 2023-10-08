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

#### Fully Sharded Data Parallel (FSDP)

- FSDP allows models that are too big to fit on a single chip.
- When using FSDP, distribute the data across multiple GPUs, but shard the model parameters, gradients, and optimizer states across the GPU nodes using one of the 3 ZeRO strategies.
- FSDP requires collecting model states (required for processing each batch) from all of the GPUs before the forward and backward pass. Each GPU requests data from the other GPUs on-demand, to materialize the sharded data into unsharded data for the duration of the operation. After the operation, release the unsharded non-local data back to the other GPUs as original sharded data. Can also choose to keep it for future operations, for example during backward pass, but this requires more GPU RAM again (a typical performance vs memory trade-off).
- In the final step after the backward pass, FSDP synchronizes the gradients across the GPUs in the same way they were for DDP.

<img src="image/image014.png" width="650"/>

<br>

<img src="image/image010.png" width="550"/>

- For a 70 billion parameter model, the ideal training dataset contains 1.4 trillion tokens (20 times the number of parameters).
- A compute optimal Chinchilla model outperforms non compute optimal models such as GPT-3 on a large range of downstream evaluation tasks.

<img src="image/image011.png" width="400"/>

## Fine-tuning (Instruction Fine-tuning)

- For LLM, fine-tuning means instruction fine tuning. Often, only 500-1000 examples are needed to fine-tune a single task.
- However, a fine-tuned model may forget how to do other tasks, called Catastrophic Forgetting. To deal with Catastrophic Forgetting, can either
  1. Ignore the problem because the use cases are limited
  2. Fine-tune on multiple tasks at the same time (may need 50,000 to 100,000 examples in the training set)
  3. Consider Parameter Efficient Fine-tuning (PEFT).
- A sample prompt training dataset (e.g. SAMsum) can be used to fine-tune (e.g. generate the FLAN-T5 from the pretrained T5. Here is a template for fine-tuning:

    <img src="image/image018.jpg" width="500"/>

## Parameter Efficient Fine-Tuning (PEFT)

<img src="image/image025.png" width="500"/>

Benefits of PEFT
- Avoid catastrophic forgetting
- Only need to train small number of weights compared to the original LLM
- Only need to store different versions of PEFT weights (instead of different versions of LLM) for each new task

### Low-Rank Adaptation of Large Language Models (LoRA)

The optimum rank is in a range of 4 to 32.

<img src="image/image026.png" width="600"/>

<img src="image/image027.png" width="600"/>

Applying LoRA to just the self-attention layers is enough to fine-tune for a task and achieve performance gains. Can also use LoRA on other components like the feed-forward layers.

<img src="image/image028.png" width="500"/>
