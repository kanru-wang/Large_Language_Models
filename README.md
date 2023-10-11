# Large Language Models

- [LLM Usage]()
- [Model Architectures and Pre-training Objectives]()
- [GPU Memory Limit]()
- [Data Parallelism vs Model Sharding]()
  - [Distributed Data Parallel (DDP)]()
  - [Model Sharding]()
- [Optimal Model Size and Training Data Size Balance]()
- [Fine-tuning (Instruction Fine-tuning)]()
- [Parameter Efficient Fine-Tuning (PEFT)]()
  - [Low-Rank Adaptation of Large Language Models (LoRA)]()
  - [Prompt Tuning with Trainable Soft Prompts]()
- [LLM Evaluation Metrics]()
- [Reinforcement Learning from Human Feedback (RLHF)]()
  - [Prepare labeled data for training]()
  - [Proximal policy optimization (PPO)]()
  - [Reward Hacking]()
- [Constitutional AI]()

<img src="image/image012.png" width="800"/>

## LLM Usage

#### Providing examples inside the context window is called in-context learning

<img src="image/image002.png" width="600"/>

#### Chain-of-Thought Prompting (One Shot or Few Shot) can help LLMs reason

<img src="image/image015.png" width="650"/>

#### Program-Aided Language models (PAL) (One Shot)

The LLM doesn't really have to decide to run the Python code, it just has to write the script which the orchestrator then passes to the external Python interpreter to run.

<img src="image/image016.png" width="550"/>

#### ReAct Framework

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

<img src="image/image031.png" width="600"/>

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

<img src="image/image008.png" width="550"/>

#### Quantization: FP16

<img src="image/image042.png" width="550"/>

#### Quantization: BFLOAT16

<img src="image/image004.png" width="550"/>

## Data Parallelism vs Model Sharding

(May be similar to https://leimao.github.io/blog/Data-Parallelism-vs-Model-Paralelism/)

### Distributed Data Parallel (DDP)

DDP requires that the model weights, and additional parameters, gradients, and optimizer states to be fit onto a single GPU. If they are too large, should use model sharding.

<img src="image/image007.png" width="550"/>

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

## Optimal Model Size and Training Data Size Balance

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

<img src="image/image025.png" width="450"/>

Benefits of PEFT
- Avoid catastrophic forgetting
- Only need to train small number of weights compared to the original LLM
- Only need to store different versions of PEFT weights at the size of MBs (instead of different versions of LLM at the size of GBs) for each task (e.g. QA, summarizing, completing...)

### Low-Rank Adaptation of Large Language Models (LoRA)

The optimum rank is in a range of 4 to 32.

<img src="image/image026.png" width="600"/>

<img src="image/image027.png" width="600"/>

Applying LoRA to just the self-attention layers is enough to fine-tune for a task and achieve performance gains. Can also use LoRA on other components like the feed-forward layers.

<img src="image/image028.png" width="500"/>

<img src="image/image029.png" width="500"/>

### Prompt Tuning with Trainable Soft Prompts

- A set of trainable tokens that are added to a prompt and whose values are updated during additional training to improve performance on specific tasks.
- Switch out Soft Prompt at inference time to change task.
- When the original LLM is large enough, Prompt Tuning can be as effective as full Fine-tuning.

<img src="image/image032.png" width="500"/>

## LLM Evaluation Metrics

<img src="image/image020.png" width="400"/>

<img src="image/image019.png" width="500"/>

<img src="image/image021.png" width="500"/>

<img src="image/image022.png" width="500"/>

A few examples Rouge-1 scores would fail:

- All words are present, but in a wrong order. Partially mitigated by Rouge-n
- “It is cold outside” and “It is not cold outside” are similar. Partially mitigated by Rouge-n
- “It is cold outside” and “Cold cold cold cold” would result in a Rouge-1 precision of 1.0. Mitigated by a clipping function that limits the number of unigram matches to the max count for that unigram within the reference sentence.
 
Can only use Rouge scores to compare the capabilities of models if the scores were determined for the same task. For example, summarization. Rouge scores for different tasks are not comparable to one another.

<img src="image/image023.png" width="400"/>

## Reinforcement Learning from Human Feedback (RLHF)

<img src="image/image033.png" width="500"/>

The LLM weights are updated iteratively to maximize the Reward, enabling the model to generate non-toxic completions.

Use an additional model, known as the Reward Model, to classify the outputs of the LLM and evaluate the degree of alignment with human preferences. Specifically, start with a smaller number of human evaluated reward (pairwise comparison data) to train the Reward Model by traditional supervised learning methods (e.g. BERT). Once trained, use the Reward Model to assign a reward value (to the output of the LLM) which is used to update the weights off the LLM.

<img src="image/image035.png" width="450"/>

### Prepare labeled data for training

Convert rankings into pairwise training data for the reward model. While thumbs-up thumbs-down feedback is often easier to gather than ranking feedback, ranked feedback can generate more training data, i.e. three prompt completion pairs from each human ranking.

<img src="image/image034.png" width="600"/>

### Proximal policy optimization (PPO)

- Reinforcement learning algorithm takes the output of the Reward Model and uses it to update the LLM model weights so that the reward score increases over time.
- Phase 1: use LLM to complete the given prompts
- Phase 2: update LLM against the Reward Model
- The PPO objective updates the model weights through back propagation over several steps. Once the model weights are updated, PPO starts a new cycle. For the next iteration, the LLM is replaced with the updated LLM, and a new PPO cycle starts. After many iterations, the human-aligned LLM is obtained.

<img src="image/image041.png" width="400"/>

#### Value Function and Value Loss

- The expected reward of a completion is estimated through a separate head of the LLM called the value function.
- The value function estimates the expected total reward for a given State S. In other words, as the LLM generates each token of a completion, estimate the total future reward based on the current sequence of tokens.
- The goal is to minimize the value loss that is the difference between the actual future total reward (e.g. 1.87), and its approximation to the value function (e.g. 1.23). The value loss makes estimates for future rewards more accurate.
- The value function is then used in Advantage Estimation in Phase 2.

<img src="image/image036.png" width="350"/>

#### Policy Loss

- π(a_t | S_t) is the probability of the next token a_t given the current prompt S_t.
- π is the model’s probability distribution over tokens.
- The action a_t is the next token, and the state S_t is the completed prompt up to the token t.
- A-hat_t is the estimated advantage term of a given choice of action.
  - The advantage term estimates how much better or worse the current action (current token A_t) is compared to all possible actions (all the possible tokens) at that state.
  - We look at the expected future rewards of a completion following the new token, and we estimate how advantageous this completion is compared to the rest.
  - There is a recursive formula to estimate this quantity based on the value function.
  - A positive advantage means that the suggested token is better than the average. Therefore, increasing the probability of the current token seems like a good strategy that leads to higher rewards.
- The advantage estimates are valid only when the old and new policies are close to each other. These extra errors terms are guardrails defining a trust region in proximity to the LLM.

<img src="image/image037.png" width="650"/>

<img src="image/image044.png" width="650"/>

#### Entropy Loss

While the policy loss moves the model towards alignment goal, entropy allows the model to maintain creativity. If you kept the entropy low, the model may always complete the prompt in the same way.

### Reward Hacking

- Reward hacking is when the agent learns to cheat the system by favoring actions that maximize the reward received even if those actions don't align well with the original objective.
- In the context of LLMs, reward hacking can manifest as the addition of words or phrases to completions that result in high scores for the metric being aligned, but that reduce the overall quality of the language.
- For example, the completion “… is the most awesome, most incredible thing ever” is certainly non-toxic, but it is not useful.
- During training, each prompt is passed to both the reference LLM and the RL updated LLM.
- KL divergence is a statistical measure of how different two probability distributions are.
- Added KL divergence to the reward calculation, which penalizes the RL updated LLM if it shifts too far from the reference LLM (generating completions that are too different).
- Can use PEFT, which reuses the same underlying LLM for both the reference LLM and the RL updated LLM. The benefit is the reduced memory footprint.

<img src="image/image038.png" width="550"/>

<img src="image/image045.png" width="600"/>

## Constitutional AI

<img src="image/image039.png" width="700"/>

- In the first stage,
  1. Generate harmful responses, which is called Red Teaming.
  2. Ask the model to critique its own harmful responses according to the constitutional principles and revise them to comply with those rules.
  3. Fine-tune the model using the pairs of red team prompts and the revised constitutional responses. The goal is to create a fine-tuned LLM that has learned to generate constitutional responses.
- In the second stage,
  1. Is similar to RLHF, except that instead of human feedback, we now use feedback generated by a model. This is called RLAIF (Reinforcement Learning from AI Feedback).
  2. Use the fine-tuned model from the previous step to generate a set of responses to prompts. The result is a model generated preference (according to the constitutional principles) dataset for training a Reward Model.
  3. With the Reward Model, can now fine-tune the fine-tuned model further using a reinforcement learning algorithm like PPO.

<img src="image/image040.png" width="550"/>
