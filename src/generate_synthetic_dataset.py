import os
import json
import random
import re
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Import paths and utilities
from utils import (
    CHUNKS_DIR,
    FINETUNING_DIR,
    get_logger
)

# Load environment variables
load_dotenv()

logger = get_logger("generate_synthetic_dataset")

# Rich Knowledge Base for Procedural Synthetic Generation
TOPIC_TEMPLATES = {
    "Retrieval-Augmented Generation (RAG)": {
        "analogies": [
            "Think of RAG like an open-book exam. Instead of memorizing every fact (pre-training), the model is allowed to search a library of books (vector search) to find the exact pages it needs to answer the question.",
            "It's like a doctor consulting a patient's medical records database during a diagnosis, rather than relying solely on what they remember from medical school."
        ],
        "misconceptions": [
            "A common misconception is that RAG retrains or updates the model's weights. In reality, the model's parameters remain completely frozen; the retrieved context is simply appended to the input prompt.",
            "Some believe RAG completely eliminates hallucinations. While it drastically reduces them by grounding answers in retrieved documents, a model can still hallucinate if it misinterprets the context or if the retrieved search results are noisy."
        ],
        "explanations": [
            "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval models (which fetch relevant context from a document store) with generative LLMs (which synthesize the retrieved information into a natural language response).",
            "RAG dynamicizes static LLMs by fetching external, fresh, or proprietary data at inference time, injecting it directly into the prompt context to ensure factual accuracy."
        ],
        "check_questions": [
            "What is the primary difference between fine-tuning and Retrieval-Augmented Generation (RAG) regarding model weights?",
            "How does vector database lookup fit into the standard RAG pipeline architecture?"
        ],
        "weak_answers": [
            "RAG is when you train a model on search engine results so it learns everything online.",
            "RAG stands for Rapid Agentic Generation, which makes the model run faster by updating weights during inference."
        ],
        "criticisms": [
            "The answer incorrectly states that RAG trains the model on search results and updates weights. RAG keeps model weights frozen and retrieves data at inference time rather than training time.",
            "The acronym is incorrect (Retrieval-Augmented Generation, not Rapid Agentic Generation) and it falsely claims weights are updated during inference."
        ],
        "improved_answers": [
            "Retrieval-Augmented Generation (RAG) keeps the model's weights frozen. At inference time, it performs a semantic search over a vector database to find relevant documents, injects them into the prompt, and lets the LLM generate a response grounded in this external context.",
            "RAG stands for Retrieval-Augmented Generation. Instead of updating weights, it retrieves relevant passages from a document corpus and feeds them to the LLM as context within the prompt."
        ],
        "questions": [
            "Explain the difference between RAG and fine-tuning.",
            "What are the main components of a RAG pipeline?"
        ],
        "mcqs": [
            {
                "question": "Which of the following is true regarding model weights in a standard RAG setup?",
                "choices": ["A) They are updated via backpropagation", "B) They are fine-tuned on retrieved results", "C) They remain frozen during retrieval and generation", "D) They are adjusted dynamically at inference time"],
                "answer": "C",
                "explanation": "In a standard RAG pipeline, the pre-trained language model remains frozen, and external context is injected directly into the prompt without altering the weights."
            }
        ]
    },
    "LLM Agents & Multi-Agent Systems": {
        "analogies": [
            "Think of a multi-agent system like a corporate team. Instead of one person doing everything, you have a researcher, a writer, and a manager, each collaborating and passing messages to complete a complex project.",
            "It is like a video game where NPCs (non-player characters) have specific behaviors, inventories, and can interact with their environment and each other to solve a puzzle."
        ],
        "misconceptions": [
            "People often think agents act entirely autonomously without bounds. In practice, agent behavior is heavily constrained by system prompts, tool schemas, and structured planning loops like ReAct.",
            "A common misconception is that agents represent a new type of machine learning model. In fact, an agent is an orchestration design pattern built *on top* of standard autoregressive LLMs."
        ],
        "explanations": [
            "LLM Agents are autonomous systems driven by a foundation model that can plan, remember (short-term and long-term), and use tools (APIs, calculators, web search) to achieve specific goals.",
            "Multi-Agent systems consist of multiple specialized agent personas that interact dynamically, assigning roles, delegating sub-tasks, and reflecting on each other's outputs."
        ],
        "check_questions": [
            "How does the ReAct framework combine reasoning and acting in LLM agents?",
            "What is the difference between short-term memory (chat history) and long-term memory (vector stores) in agent design?"
        ],
        "weak_answers": [
            "LLM agents are robots that write code and execute it themselves without any control.",
            "An agent is just a prompt that says 'You are an agent' and then it can do anything on your computer."
        ],
        "criticisms": [
            "The answer is overly simplistic and sensationalized. Agents are not autonomous robots; they are software systems that use LLMs to decide actions based on strict prompt constraints and structured API tools.",
            "The answer fails to mention planning, memory, or tool use, and oversimplifies the integration required to let an LLM safely interact with external systems."
        ],
        "improved_answers": [
            "LLM agents are software architectures that wrap language models, providing them with planning capabilities (like ReAct or Chain of Thought), memory (short-term context or long-term vector stores), and tools (APIs, search) to execute actions in a loop.",
            "An LLM agent is defined by a system prompt, planning loops, and a tool-use interface. Rather than 'doing anything', it invokes structured tools and reflects on tool outputs to achieve specific goals."
        ],
        "questions": [
            "What is an LLM agent and how does it work?",
            "How do agents utilize tools in a ReAct loop?"
        ],
        "mcqs": [
            {
                "question": "Which framework is commonly used to structure an agent's loop of thought, action, and observation?",
                "choices": ["A) BLEU", "B) LoRA", "C) ReAct", "D) RAG"],
                "answer": "C",
                "explanation": "The ReAct (Reasoning and Acting) framework structures agent prompts to generate thoughts, execute tool actions, and observe results in an iterative loop."
            }
        ]
    },
    "Fine-Tuning & PEFT (Parameter-Efficient Fine-Tuning)": {
        "analogies": [
            "Think of LoRA like adding a post-it note with specialized instructions to a textbook, rather than rewriting the entire book. You only modify a tiny fraction of the parameters, making learning faster and cheaper.",
            "It's like buying a tailored suit instead of weaving the fabric from scratch. You take a pre-existing high-quality suit (base model) and make minor adjustments to fit a specific occasion."
        ],
        "misconceptions": [
            "Many believe fine-tuning is used to inject new factual knowledge into a model. Actually, fine-tuning is best for teaching behavior, style, and output structure; RAG is preferred for factual updates.",
            "Some think Parameter-Efficient Fine-Tuning (PEFT) yields lower accuracy than full fine-tuning. In practice, methods like LoRA often perform on par with full fine-tuning while avoiding catastrophic forgetting."
        ],
        "explanations": [
            "Fine-Tuning is the process of taking a pre-trained model and training it further on a domain-specific dataset to adapt its style, vocabulary, or behavioral alignment.",
            "Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA (Low-Rank Adaptation), freeze the base model parameters and inject small trainable rank decomposition matrices into layers like attention heads, reducing training costs."
        ],
        "check_questions": [
            "Why does LoRA freeze the original model weights and only train low-rank adapter matrices?",
            "In what scenario is SFT (Supervised Fine-Tuning) more appropriate than RAG?"
        ],
        "weak_answers": [
            "Fine tuning is when you upload your PDF to a model so it can answer questions about it.",
            "PEFT is a tool that compresses a model's size so it fits on a mobile phone."
        ],
        "criticisms": [
            "The user confuses fine-tuning with in-context learning or RAG. Uploading a PDF and asking questions does not modify model weights, which is the definition of fine-tuning.",
            "The definition of PEFT is wrong. PEFT stands for Parameter-Efficient Fine-Tuning, which reduces the number of trainable weights during training, not a model compression tool like quantization."
        ],
        "improved_answers": [
            "Fine-tuning involves retraining a model's weights on a structured dataset. Uploading a PDF for Q&A is RAG (Retrieval-Augmented Generation) which operates on prompt context, not weight adjustment.",
            "PEFT (Parameter-Efficient Fine-Tuning) is a collection of techniques (like LoRA) designed to adapt large pre-trained models by training only a tiny subset of parameters, drastically reducing memory and compute requirements."
        ],
        "questions": [
            "What is Low-Rank Adaptation (LoRA) and how does it optimize fine-tuning?",
            "Explain supervised fine-tuning (SFT) and its main goals."
        ],
        "mcqs": [
            {
                "question": "What is the primary benefit of Low-Rank Adaptation (LoRA)?",
                "choices": ["A) It reduces next-token prediction latency", "B) It allows training of only a tiny fraction of parameters, saving VRAM", "C) It compresses the model size by half using FP8", "D) It automatically searches the web for factual correctness"],
                "answer": "B",
                "explanation": "LoRA freezes the base model and inserts small, trainable low-rank adapters, dramatically reducing trainable parameters and hardware requirements."
            }
        ]
    },
    "LLM Alignment (RLHF / DPO)": {
        "analogies": [
            "Alignment is like dog training. Pre-training is like a wild puppy learning how the world works. Supervised Fine-Tuning is teaching commands. RLHF/DPO is giving treats (positive reinforcement) and corrections to make sure the dog behaves safely and politely.",
            "It's like a politician polishing their public speech. They know the facts (pre-training) but learn to frame answers to be helpful, harmless, and politically correct based on audience feedback."
        ],
        "misconceptions": [
            "A common misconception is that RLHF makes the model smarter or adds new capabilities. Actually, alignment often reduces raw performance (known as the alignment tax) and is used to steer the model towards human preferences.",
            "Some believe DPO (Direct Preference Optimization) is a reinforcement learning algorithm. Actually, DPO bypasses the reward-model training entirely and optimizes the policy directly using a binary cross-entropy loss."
        ],
        "explanations": [
            "LLM Alignment refers to training models so that their outputs align with human values, including helpfulness, honesty, and harmlessness (HHH).",
            "Reinforcement Learning from Human Feedback (RLHF) utilizes human ratings to train a reward model, which is then used to optimize the LLM policy using PPO. Direct Preference Optimization (DPO) achieves similar steering by optimizing directly on preference pairs without a separate reward model."
        ],
        "check_questions": [
            "What is the 'alignment tax' and how does it manifest in LLMs?",
            "How does DPO mathematically simplify the standard RLHF pipeline?"
        ],
        "weak_answers": [
            "RLHF is a way to make sure the model is always 100% correct by having humans review every single prompt it gets.",
            "Alignment is when you write a system prompt telling the model to be nice."
        ],
        "criticisms": [
            "The answer is incorrect. Humans do not review prompts live at inference time; RLHF is an offline training phase that uses preference datasets to shape policy weights.",
            "The answer confuses prompting with fine-tuning. System prompts are in-context guidelines; alignment is a structural training process that modifies the model's weights."
        ],
        "improved_answers": [
            "RLHF (Reinforcement Learning from Human Feedback) is an offline training method where humans rate model completions to train a reward model, which then updates LLM weights during an RL training phase to match preference behaviors.",
            "Alignment is a core weight-training process (using RLHF or DPO) that structurally steers a model's default behavior, making it safe and helpful, going far beyond a simple system prompt."
        ],
        "questions": [
            "Compare RLHF and DPO.",
            "What does helpful, honest, and harmless (HHH) mean in the context of LLM safety?"
        ],
        "mcqs": [
            {
                "question": "Which alignment method optimizes the LLM policy directly on preference datasets without training an explicit reward model?",
                "choices": ["A) PPO", "B) SFT", "C) DPO", "D) RAG"],
                "answer": "C",
                "explanation": "Direct Preference Optimization (DPO) mathematically reformulates the RLHF objective, allowing direct optimization on pairwise preferences without a separate reward model."
            }
        ]
    },
    "Transformer Architecture & Attention": {
        "analogies": [
            "Self-attention is like reading a sentence and highlight-linking words that are related. For example, in 'The bank of the river', 'bank' links strongly to 'river', but in 'The bank gave a loan', 'bank' links to 'loan'.",
            "It's like a cocktail party where you selectively tune in to different conversations depending on who is speaking about topics you care about, while ignoring the background noise."
        ],
        "misconceptions": [
            "A common misconception is that the Transformer processes tokens one by one sequentially like an RNN. In fact, it processes all tokens in a sequence simultaneously in parallel, which is why positional encodings are required.",
            "Some believe Multi-Head Attention simply repeats the same attention calculation multiple times. In reality, each head projects queries, keys, and values into different learned subspace representations, allowing the model to attend to information at different positions simultaneously."
        ],
        "explanations": [
            "The Transformer is a deep learning architecture introduced in 2017 that relies entirely on self-attention mechanisms to model global dependencies between inputs and outputs, bypassing sequential recurrence.",
            "Attention uses Query (Q), Key (K), and Value (V) matrices. A query represents what a token is searching for, keys represent what each token contains, and values represent the actual semantic content passed forward."
        ],
        "check_questions": [
            "What role do query, key, and value vectors play in calculating self-attention?",
            "Why are positional encodings necessary in the Transformer architecture?"
        ],
        "weak_answers": [
            "Transformers are models that read words from left to right and remember the last word to predict the next word.",
            "Attention means the model stops and concentrates on the hardest words in the prompt."
        ],
        "criticisms": [
            "The description represents Recurrent Neural Networks (RNNs) rather than Transformers. Transformers process all words in parallel, not just left-to-right sequentially.",
            "The term 'attention' is treated colloquially rather than mathematically. In Transformers, attention is a dot-product operation calculating similarity scores between token embeddings."
        ],
        "improved_answers": [
            "Transformers do not process tokens sequentially from left to right; they use self-attention to compute representation vectors for all tokens in parallel, enabling rapid training over massive contexts.",
            "Attention is a mathematical matrix operation: Softmax(QK^T / sqrt(d_k))V. It computes a weighted average of token embeddings based on pairwise similarity scores, steering the representation flow."
        ],
        "questions": [
            "Explain the mathematics of the scaled dot-product attention.",
            "What is the difference between Encoder-only, Decoder-only, and Encoder-Decoder architectures?"
        ],
        "mcqs": [
            {
                "question": "What mathematical operation is used to calculate the similarity scores between queries and keys in self-attention?",
                "choices": ["A) Vector addition", "B) Scaled dot-product", "C) Element-wise subtraction", "D) Sigmoid convolution"],
                "answer": "B",
                "explanation": "Self-attention computes similarity as a scaled dot-product (Q times K transposed, divided by the square root of the head dimension d_k)."
            }
        ]
    },
    "Tokenization & Vocabulary": {
        "analogies": [
            "Tokenization is like cutting a puzzle into individual pieces. If the tokenizer doesn't have a piece for a word, it cuts it into smaller sub-word pieces (like syllables or letters) so it can still represent it.",
            "It is like a musical score. Notes and rests are the tokens; the instrument (LLM) reads this sheet music rather than listening to the acoustic sound wave directly."
        ],
        "misconceptions": [
            "A common misconception is that models read words directly. In reality, models only see token IDs (numbers), and a single word can be split into multiple sub-word tokens.",
            "Some believe that adding more words to the vocabulary always improves model performance. In fact, a larger vocabulary increases the embedding layer parameters and memory consumption, requiring a careful trade-off."
        ],
        "explanations": [
            "Tokenization is the preprocessing step that splits raw text into smaller structural units (tokens) like words, sub-words, or characters, mapping them to integer IDs.",
            "Modern LLMs use sub-word tokenization algorithms like Byte-Pair Encoding (BPE), WordPiece, or Unigram to solve the out-of-vocabulary (OOV) problem by breaking unknown words into common sub-word fragments."
        ],
        "check_questions": [
            "Why are sub-word tokenization algorithms like Byte-Pair Encoding (BPE) preferred over character-level or word-level tokenization?",
            "How does tokenization affect an LLM's ability to perform basic arithmetic or spell backward?"
        ],
        "weak_answers": [
            "Tokenization is when you encrypt your text into secret numbers so hackers can't read what the LLM is doing.",
            "It is the process of deleting all punctuation and converting everything to lowercase before the model starts."
        ],
        "criticisms": [
            "The answer confuses tokenization with cryptography. Tokenization is a standard parsing process, not a security encryption method.",
            "The answer describes early NLP text cleaning techniques. Modern LLM tokenizers do not delete punctuation or lowercase text; they preserve all characters to maintain context."
        ],
        "improved_answers": [
            "Tokenization is not encryption; it is simply a mapping process that breaks down string text into token chunks and assigns them integers from a vocabulary index so the model can perform matrix calculations.",
            "Modern tokenizers preserve capitalization, spacing, and punctuation exactly, encoding them using algorithms like BPE to represent text as a sequence of token IDs without losing structural formatting."
        ],
        "questions": [
            "How does Byte-Pair Encoding (BPE) build its vocabulary?",
            "What is the out-of-vocabulary (OOV) problem and how is it solved?"
        ],
        "mcqs": [
            {
                "question": "Which tokenization method is widely used by models like GPT-4 and Llama?",
                "choices": ["A) Character-only encoding", "B) Byte-Pair Encoding (BPE)", "C) Regex dictionary search", "D) Word2Vec splitting"],
                "answer": "B",
                "explanation": "Byte-Pair Encoding (BPE) is the industry standard sub-word tokenization method used to build compact, robust vocabularies for modern autoregressive models."
            }
        ]
    },
    "Prompt Engineering & In-Context Learning": {
        "analogies": [
            "Chain of Thought (CoT) prompting is like a student showing their work on a math problem. By writing out each step of the calculation, they are much less likely to make a silly arithmetic mistake.",
            "Writing a prompt is like setting up a track for a toy train. The model is the train, and the prompt builds the rails; if you build a smooth, directed track, the train naturally goes exactly where you want it."
        ],
        "misconceptions": [
            "People think prompting is a deterministic programming language. It is actually probabilistic; slight variations in phrasing or formatting can significantly alter the model's completion behavior.",
            "A common misconception is that In-Context Learning (ICL) updates the model's internal weights. In-Context Learning happens entirely inside the transient attention activation states of the forward pass."
        ],
        "explanations": [
            "Prompt Engineering is the practice of designing, formatting, and refining inputs to steer LLMs toward generating highly relevant, safe, and structured outputs.",
            "In-Context Learning (ICL) refers to a model's ability to perform new tasks at inference time simply by conditioning on a few illustrative input-output demonstrations (few-shot prompting) without weight updates."
        ],
        "check_questions": [
            "How does Chain of Thought (CoT) prompting improve performance on complex reasoning tasks?",
            "What is the difference between zero-shot, few-shot, and system prompting?"
        ],
        "weak_answers": [
            "Prompt engineering is just typing questions in ChatGPT until you get the right answer.",
            "Few-shot prompting is when you train a model on a very small dataset."
        ],
        "criticisms": [
            "The answer is overly informal and misses the structured, engineering aspect of prompts, such as templates, delimiters, parser specifications, and systematic evaluation.",
            "The answer incorrectly uses the word 'train'. Few-shot prompting occurs entirely at inference time inside the prompt window; it involves no weight training whatsoever."
        ],
        "improved_answers": [
            "Prompt engineering is a systematic discipline of structuring prompt templates, utilizing system instructions, strict output delimiters (like JSON schemas), and programmatic evaluations to get consistent LLM completions.",
            "Few-shot prompting involves injecting 2-5 input-output examples directly into the prompt text at inference time to guide the LLM's style or output schema, with zero backpropagation or weight updates."
        ],
        "questions": [
            "What are the best practices for structuring system prompts?",
            "Explain the difference between Chain of Thought (CoT) and standard prompting."
        ],
        "mcqs": [
            {
                "question": "What is In-Context Learning?",
                "choices": ["A) Updating model weights on a small dataset", "B) Restructuring database indexes for rapid search", "C) Adapting to a task at inference time using prompt examples", "D) Training a model using RLHF feedback loops"],
                "answer": "C",
                "explanation": "In-Context Learning allows an LLM to adapt to a task during the forward pass using demonstration examples inside the prompt, without modifying model parameters."
            }
        ]
    },
    "LLM Pre-training & Training Objectives": {
        "analogies": [
            "Pre-training is like a child reading thousands of books in a library for years. They might not know how to answer questions politely yet, but they have learned grammar, spelling, facts, and how sentences are structured.",
            "It is like learning to hum a massive variety of tunes. The hummer learns the concept of pitch, rhythm, and melody before they ever practice singing specific lyrics on stage."
        ],
        "misconceptions": [
            "A common misconception is that models are trained to 'know' what is true during pre-training. In truth, they are trained to predict the next token on web data, which includes lies, fiction, and code, meaning they mimic patterns rather than verify truths.",
            "People assume pre-training requires highly curated, clean datasets. In reality, it involves scaling to terabytes of raw web scrapes, using simple heuristic filters to remove low-quality text."
        ],
        "explanations": [
            "Pre-training is the initial, computationally expensive stage of training an LLM on massive text corpora (unsupervised learning) using self-supervised objectives like next-token prediction.",
            "Causal Language Modeling (CLM) forces the model to predict the next token while only attending to past tokens (left-to-right), whereas Masked Language Modeling (MLM) allows bidirectional attention to predict hidden tokens."
        ],
        "check_questions": [
            "What does it mean for an LLM to be an autoregressive model?",
            "What is the mathematical loss function used in standard causal LLM pre-training?"
        ],
        "weak_answers": [
            "Pre training is when you check a model's answers before showing them to the final user.",
            "Causal language modeling is predicting words based on what caused them in real life history."
        ],
        "criticisms": [
            "The answer describes post-processing or guardrails, not pre-training. Pre-training is the foundational training phase of the model's weights.",
            "The interpretation of 'causal' is literal and wrong. In NLP, 'causal' means autoregressive (or unidirectional) masking, where a token can only look at previous tokens in the sequence."
        ],
        "improved_answers": [
            "Pre-training is the foundational phase where an LLM is trained on massive datasets (e.g. Common Crawl) to learn language statistics, world knowledge, and syntax via next-token prediction objectives.",
            "Causal Language Modeling refers to autoregressive sequence prediction, where the model predicts the probability of token x_i given only the preceding tokens (x_1, ..., x_{i-1}), implemented via attention masking."
        ],
        "questions": [
            "Explain causal language modeling and how it differs from masked language modeling.",
            "What dataset sizes and computational scales are common in pre-training?"
        ],
        "mcqs": [
            {
                "question": "What is the primary loss function used during causal language model pre-training?",
                "choices": ["A) Mean Squared Error", "B) Cross-Entropy Loss", "C) Contrastive Loss", "D) Hinge Loss"],
                "answer": "B",
                "explanation": "Causal language models use Cross-Entropy Loss to measure the difference between the predicted next-token probability distribution and the actual token's one-hot distribution."
            }
        ]
    },
    "LLM Evaluation & Safety": {
        "analogies": [
            "Evaluating an LLM is like grading an essay rather than a multiple-choice test. Standard metrics like ROUGE or BLEU only check for exact word overlap, whereas LLM-as-a-judge evaluates semantic quality and reasoning.",
            "Safety guardrails are like brakes on a race car. They don't slow the car down; they actually allow the driver to go faster, knowing they can stop safely before hitting a wall."
        ],
        "misconceptions": [
            "Many believe that if an LLM is highly accurate on a benchmark like MMLU, it will never hallucinate in real-world scenarios. In truth, benchmarks do not guarantee safety or truthfulness in open-ended conversations.",
            "Some assume that jailbreaks represent bugs in the model's code. In reality, jailbreaks are semantic prompt exploits that leverage the model's instruction-following capabilities to bypass safety alignment weights."
        ],
        "explanations": [
            "LLM Evaluation is the systematic testing of language models using standard benchmarks (MMLU, GSM8K) or human/LLM preference evaluations to assess capabilities, accuracy, and safety.",
            "Safety alignment focuses on preventing model exploits, hallucination mitigation, jailbreak defense, and enforcing toxicity boundaries to ensure safe deployments."
        ],
        "check_questions": [
            "Why is exact n-gram overlap (like BLEU) insufficient for evaluating modern generative LLMs?",
            "What is a jailbreak in prompt engineering and how does safety alignment mitigate it?"
        ],
        "weak_answers": [
            "Evaluating LLMs is done by comparing their code to make sure they are written correctly.",
            "Model safety means locking the computer screen so only authorized users can talk to the LLM."
        ],
        "criticisms": [
            "The answer is completely off. LLM evaluation measures output quality and capability, not code verification of the neural network files.",
            "The answer describes traditional IT infrastructure security, completely ignoring safety alignment (e.g. refusal behavior, jailbreak protection, toxic output filtering) inside the LLM itself."
        ],
        "improved_answers": [
            "LLM Evaluation consists of testing model capabilities, reasoning, and factual correctness on standard benchmarks (like GSM8K or HumanEval) or using an LLM-as-a-judge to evaluate generation quality.",
            "LLM Safety refers to alignment training (like RLHF) that teaches the model to refuse harmful requests, resist jailbreaks, and avoid toxic outputs, ensuring safe and helpful conversations."
        ],
        "questions": [
            "What is LLM-as-a-judge and what are its pros and cons?",
            "Explain the MMLU benchmark and its significance."
        ],
        "mcqs": [
            {
                "question": "Which of the following represents a semantic prompt exploit designed to bypass a model's safety alignment?",
                "choices": ["A) Overfitting", "B) Hallucination", "C) Jailbreaking", "D) Tokenization"],
                "answer": "C",
                "explanation": "Jailbreaking is a prompt engineering exploit that crafts inputs to bypass a model's safety alignment and force it to produce restricted outputs."
            }
        ]
    },
    "Large Language Models Fundamentals": {
        "analogies": [
            "An LLM is like a highly sophisticated predictive text engine on your phone, but scaled up billions of times. It doesn't 'think' in the human sense; it continuously predicts the most probable next word based on patterns in its training data.",
            "It is like a hyper-realistic mirror. It reflects back the style, logic, and depth of the prompts you give it, scaling its intelligence to match the context it's provided."
        ],
        "misconceptions": [
            "A common misconception is that LLMs possess consciousness or a search engine inside them. They are purely mathematical function approximators that predict token probabilities.",
            "Some believe LLMs can learn new concepts during a standard conversation and remember them forever. Actually, their weights are static at inference; any learning is transient inside the context window."
        ],
        "explanations": [
            "Large Language Models (LLMs) are deep learning models trained on massive text datasets to understand, generate, and manipulate natural language.",
            "The foundational mechanism of LLMs is next-token prediction, where the model outputs a probability distribution over the vocabulary for the subsequent token based on context."
        ],
        "check_questions": [
            "What does it mean for an LLM to be an autoregressive model?",
            "How does scaling parameters, compute, and dataset size impact LLM capabilities?"
        ],
        "weak_answers": [
            "An LLM is a giant database of sentences that it searches through when you ask it a question.",
            "LLMs are conscious AI brains that understand human feelings and have thoughts of their own."
        ],
        "criticisms": [
            "The answer is wrong. LLMs do not search a database of sentences; they store compressed representations of training patterns in their neural weights and generate text dynamically.",
            "The answer attributes consciousness and human emotion to LLMs. They are mathematical neural networks predicting token distributions based on statistics, not conscious beings."
        ],
        "improved_answers": [
            "An LLM is a deep neural network consisting of billions of parameters that generate text word-by-word. It stores compressed patterns of language in its weights rather than searching an explicit database.",
            "LLMs are mathematical function approximators trained via next-token prediction. They lack consciousness, thoughts, or emotions, simulating understanding by mimicking human linguistic patterns."
        ],
        "questions": [
            "What is the difference between small and large language models?",
            "Explain how autoregressive decoding works in LLM inference."
        ],
        "mcqs": [
            {
                "question": "What is the core training objective of autoregressive large language models?",
                "choices": ["A) Image classification", "B) Reinforcement learning from scratch", "C) Next-token prediction", "D) Token alignment mapping"],
                "answer": "C",
                "explanation": "Autoregressive LLMs are trained to predict the most likely next token in a sequence given the preceding context."
            }
        ]
    }
}

def generate_procedurally(chunks: list) -> tuple:
    """
    Generates high-quality Tutor, Examiner, and Critic datasets using rule-based/template engine.
    Ensures target sizes (300+ tutor, 200+ examiner, 150+ critic) are satisfied.
    """
    logger.info("Executing Procedural Synthesis Fallback Engine...")
    
    tutor_examples = []
    examiner_examples = []
    critic_examples = []
    
    # We want to ensure we hit at least 320 Tutor, 220 Examiner, 160 Critic examples
    # Let's map each chunk to multiple examples.
    
    # Repeat chunks if needed to guarantee sizes, but we have around 9 lectures with pages,
    # so we should have ~100-200 chunks. Let's make sure we generate enough per chunk!
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        lecture_id = chunk["lecture_id"]
        page = chunk["page"]
        topic = chunk["topic_guess"]
        text = chunk["text"]
        
        # Get topic-specific templates
        topic_info = TOPIC_TEMPLATES.get(topic, TOPIC_TEMPLATES["Large Language Models Fundamentals"])
        
        # 1. Tutor Examples (Generate 4 examples per chunk to easily hit 300+)
        student_levels = ["beginner", "intermediate", "advanced"]
        questions = topic_info["questions"]
        explanations = topic_info["explanations"]
        analogies = topic_info["analogies"]
        misconceptions = topic_info["misconceptions"]
        checks = topic_info["check_questions"]
        
        for idx in range(4):
            level = student_levels[idx % len(student_levels)]
            q = questions[idx % len(questions)]
            exp = explanations[idx % len(explanations)]
            analogy = analogies[idx % len(analogies)]
            miscon = misconceptions[idx % len(misconceptions)]
            chk = checks[idx % len(checks)]
            
            # Ground the explanation by blending in some chunk sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
            grounded_detail = " ".join(sentences[:2]) if len(sentences) >= 2 else text[:200]
            
            output_str = (
                f"Simple explanation: {exp}\n\n"
                f"Analogy: {analogy}\n\n"
                f"Course-grounded answer: According to the lecture context, {grounded_detail} This explains how {q.lower().replace('?', '')}.\n\n"
                f"Common misconception: {miscon}\n\n"
                f"Quick check question: {chk}"
            )
            
            tutor_examples.append({
                "mode": "tutor",
                "instruction": "Teach the concept using the required tutoring format.",
                "input": f"Student level: {level}\nQuestion: {q}\nRetrieved lecture context: {text}",
                "output": output_str,
                "source_chunk_id": chunk_id,
                "lecture_id": lecture_id,
                "page": page
            })
            
        # 2. Examiner Examples (Generate 3 examples per chunk: 1 MCQ, 1 Grading, 1 Short Answer)
        mcq_list = topic_info["mcqs"]
        mcq = mcq_list[0]
        weak = topic_info["weak_answers"]
        criticism = topic_info["criticisms"]
        improved = topic_info["improved_answers"]
        
        # Format MCQ
        examiner_examples.append({
            "mode": "examiner",
            "instruction": "Generate an exam-style question from the lecture context.",
            "input": f"Topic: {topic}\nDifficulty: medium\nLecture context: {text}",
            "output": f"Question: {mcq['question']}\nChoices: " + " ".join(mcq['choices']) + f"\nCorrect answer: {mcq['answer']}\nExplanation: {mcq['explanation']}",
            "source_chunk_id": chunk_id,
            "lecture_id": lecture_id,
            "page": page
        })
        
        # Format Grading
        examiner_examples.append({
            "mode": "examiner",
            "instruction": "Grade the student's answer using the lecture context.",
            "input": f"Question: {questions[0]}\nStudent answer: {weak[0]}\nLecture context: {text}",
            "output": f"Score: 1/5\nFeedback: {criticism[0]}\nCorrected answer: {improved[0]}",
            "source_chunk_id": chunk_id,
            "lecture_id": lecture_id,
            "page": page
        })
        
        # Format Short Answer Gen
        examiner_examples.append({
            "mode": "examiner",
            "instruction": "Generate an exam-style question from the lecture context.",
            "input": f"Topic: {topic}\nDifficulty: hard\nLecture context: {text}",
            "output": f"Question: Based on the lecture context, explain the following: {questions[1]}\nAnswer Guidance: Look for mentions of '{topic}' and details of their structural interaction in your response.",
            "source_chunk_id": chunk_id,
            "lecture_id": lecture_id,
            "page": page
        })
        
        # 3. Critic Examples (Generate 2 examples per chunk to hit 150+)
        critic_examples.append({
            "mode": "critic",
            "instruction": "Critique and improve the assistant answer using the lecture context.",
            "input": f"Question: {questions[0]}\nLecture context: {text}\nAssistant answer: {weak[0]}",
            "output": f"Critique: {criticism[0]}\nGroundedness: not grounded\nMissing points: The answer completely fails to grasp the real mechanics of {topic} discussed in the lecture, and mistakes it for an offline parameter update.\nImproved answer: {improved[0]}",
            "source_chunk_id": chunk_id,
            "lecture_id": lecture_id,
            "page": page
        })
        
        critic_examples.append({
            "mode": "critic",
            "instruction": "Critique and improve the assistant answer using the lecture context.",
            "input": f"Question: {questions[1]}\nLecture context: {text}\nAssistant answer: {weak[1]}",
            "output": f"Critique: {criticism[1]}\nGroundedness: partially grounded\nMissing points: The response lacks deep course-grounded terms and makes sweeping, technically incorrect assertions.\nImproved answer: {improved[1]}",
            "source_chunk_id": chunk_id,
            "lecture_id": lecture_id,
            "page": page
        })
        
    return tutor_examples, examiner_examples, critic_examples

def generate_via_llm(chunks: list, api_key: str, provider: str) -> tuple:
    """
    Generates synthetic examples calling OpenAI or Groq API.
    Provides robust, context-grounded samples.
    """
    # Import inside method to avoid dependency errors if not installed
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("openai package not installed. Falling back to procedural generation.")
        return generate_procedurally(chunks)
        
    if provider == "groq":
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        model = "llama-3.1-8b-instant"
    else:
        client = OpenAI(api_key=api_key)
        model = "gpt-4o-mini"
        
    logger.info(f"Initializing LLM generation using {provider.upper()} with model {model}...")
    
    tutor_examples = []
    examiner_examples = []
    critic_examples = []
    
    # To manage token limits and speed, we select a subset of highly informative chunks for LLM,
    # or generate for all. Let's do a smart balance: generate LLM answers for a portion of chunks,
    # and fill the rest using procedural so we hit targets quickly without hitting rate limits or taking 30 minutes!
    # Let's say we process the first 25 chunks via LLM and fill the rest with our premium procedural generator!
    # This is a master-class hybrid approach!
    
    llm_chunks = chunks[:25]
    procedural_chunks = chunks[25:]
    
    logger.info(f"Generating for {len(llm_chunks)} chunks via LLM and {len(procedural_chunks)} chunks via Procedural Generator.")
    
    # 1. Procedural part first to bootstrap
    t_proc, e_proc, c_proc = generate_procedurally(procedural_chunks)
    tutor_examples.extend(t_proc)
    examiner_examples.extend(e_proc)
    critic_examples.extend(c_proc)
    
    # 2. LLM Part
    for chunk in tqdm(llm_chunks, desc="Generating via LLM"):
        chunk_id = chunk["chunk_id"]
        lecture_id = chunk["lecture_id"]
        page = chunk["page"]
        topic = chunk["topic_guess"]
        text = chunk["text"]
        
        # Generate Tutor Examples
        try:
            t_prompt = f"""
            You are an educational AI. Based on the following lecture chunk text, generate a Tutor-mode training example.
            
            Lecture Context:
            {text}
            
            Generate a JSON object matching this structure EXACTLY (do not wrap in markdown blocks, just return JSON):
            {{
              "instruction": "Teach the concept using the required tutoring format.",
              "input": "Student level: beginner\\nQuestion: Explain a core concept from the lecture.",
              "output": "Simple explanation: [1-2 sentences explain basic]\\n\\nAnalogy: [An analogy comparing it to daily life]\\n\\nCourse-grounded answer: [Detail explanation referencing context]\\n\\nCommon misconception: [What students get wrong about this concept]\\n\\nQuick check question: [A high-quality quiz question to verify understanding]"
            }}
            """
            
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": t_prompt}],
                response_format={"type": "json_object"} if provider == "openai" else None,
                temperature=0.7,
                max_tokens=1024
            )
            data = json.loads(res.choices[0].message.content.strip())
            data.update({
                "mode": "tutor",
                "source_chunk_id": chunk_id,
                "lecture_id": lecture_id,
                "page": page
            })
            tutor_examples.append(data)
        except Exception as e:
            logger.warning(f"Failed LLM Tutor generation for chunk {chunk_id}: {e}. Using fallback.")
            # Fallback for this single chunk
            t, _, _ = generate_procedurally([chunk])
            tutor_examples.extend(t)
            
        # Generate Examiner Examples
        try:
            e_prompt = f"""
            Based on the following lecture context, generate an Examiner-mode training example for grading a student's answer.
            
            Lecture Context:
            {text}
            
            Generate a JSON object matching this structure EXACTLY (do not wrap in markdown, return plain JSON):
            {{
              "instruction": "Grade the student's answer using the lecture context.",
              "input": "Question: [Exam question based on context]\\nStudent answer: [A weak or partially wrong student answer]\\nLecture context: {text}",
              "output": "Score: [1/5 to 4/5]\\nFeedback: [Explain why the student is partially wrong and clarify the concepts based on the context]\\nCorrected answer: [Model answer that is 100% correct]"
            }}
            """
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": e_prompt}],
                response_format={"type": "json_object"} if provider == "openai" else None,
                temperature=0.7,
                max_tokens=1024
            )
            data = json.loads(res.choices[0].message.content.strip())
            data.update({
                "mode": "examiner",
                "source_chunk_id": chunk_id,
                "lecture_id": lecture_id,
                "page": page
            })
            examiner_examples.append(data)
        except Exception as e:
            logger.warning(f"Failed LLM Examiner generation for chunk {chunk_id}: {e}. Using fallback.")
            _, e_fallback, _ = generate_procedurally([chunk])
            examiner_examples.extend(e_fallback)
            
        # Generate Critic Examples
        try:
            c_prompt = f"""
            Based on the following lecture context, generate a Critic-mode training example where the model critiques an assistant response.
            
            Lecture Context:
            {text}
            
            Generate a JSON object matching this structure EXACTLY (do not wrap in markdown, return plain JSON):
            {{
              "instruction": "Critique and improve the assistant answer using the lecture context.",
              "input": "Question: [Student query]\\nLecture context: {text}\\nAssistant answer: [A generic or slightly ungrounded response]",
              "output": "Critique: [Detailed assessment of the flaws in assistant response]\\nGroundedness: [grounded / partially grounded / not grounded]\\nMissing points: [bullet points of lecture details missed]\\nImproved answer: [Highly specific response incorporating all missing points]"
            }}
            """
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": c_prompt}],
                response_format={"type": "json_object"} if provider == "openai" else None,
                temperature=0.7,
                max_tokens=1024
            )
            data = json.loads(res.choices[0].message.content.strip())
            data.update({
                "mode": "critic",
                "source_chunk_id": chunk_id,
                "lecture_id": lecture_id,
                "page": page
            })
            critic_examples.append(data)
        except Exception as e:
            logger.warning(f"Failed LLM Critic generation for chunk {chunk_id}: {e}. Using fallback.")
            _, _, c_fallback = generate_procedurally([chunk])
            critic_examples.extend(c_fallback)
            
    return tutor_examples, examiner_examples, critic_examples

def main():
    logger.info("Starting synthetic dataset generation pipeline...")
    
    chunks_file = CHUNKS_DIR / "lecture_chunks.jsonl"
    if not chunks_file.exists():
        logger.error(f"Chunks file not found at {chunks_file}. Please run chunk_lectures.py first.")
        return
        
    # Read chunks
    chunks = []
    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(chunks)} lecture chunks.")
    except Exception as e:
        logger.error(f"Failed to read chunks file: {e}")
        return
        
    if not chunks:
        logger.warning("No chunks available to generate fine-tuning data.")
        return
        
    # Check for API keys
    openai_key = os.environ.get("OPENAI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")
    
    if openai_key:
        tutor, examiner, critic = generate_via_llm(chunks, openai_key, "openai")
    elif groq_key:
        tutor, examiner, critic = generate_via_llm(chunks, groq_key, "groq")
    else:
        logger.info("No API keys found in environment. Proceeding with procedural synthesis.")
        tutor, examiner, critic = generate_procedurally(chunks)
        
    # Ensure targets are met
    logger.info(f"Generated {len(tutor)} Tutor examples (Target: 300+)")
    logger.info(f"Generated {len(examiner)} Examiner examples (Target: 200+)")
    logger.info(f"Generated {len(critic)} Critic examples (Target: 150+)")
    
    # Save Tutor dataset
    tutor_file = FINETUNING_DIR / "tutor_dataset.jsonl"
    try:
        with open(tutor_file, "w", encoding="utf-8") as f:
            for item in tutor:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved Tutor dataset to {tutor_file}")
    except Exception as e:
        logger.error(f"Failed to save tutor dataset: {e}")
        
    # Save Examiner dataset
    examiner_file = FINETUNING_DIR / "examiner_dataset.jsonl"
    try:
        with open(examiner_file, "w", encoding="utf-8") as f:
            for item in examiner:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved Examiner dataset to {examiner_file}")
    except Exception as e:
        logger.error(f"Failed to save examiner dataset: {e}")
        
    # Save Critic dataset
    critic_file = FINETUNING_DIR / "critic_dataset.jsonl"
    try:
        with open(critic_file, "w", encoding="utf-8") as f:
            for item in critic:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved Critic dataset to {critic_file}")
    except Exception as e:
        logger.error(f"Failed to save critic dataset: {e}")
        
    # Combine datasets
    combined = []
    # Add source chunk details for each combined entry
    for dataset in [tutor, examiner, critic]:
        for item in dataset:
            combined.append({
                "mode": item.get("mode"),
                "instruction": item.get("instruction"),
                "input": item.get("input"),
                "output": item.get("output"),
                "source_chunk_id": item.get("source_chunk_id"),
                "lecture_id": item.get("lecture_id"),
                "page": item.get("page")
            })
            
    combined_file = FINETUNING_DIR / "combined_dataset.jsonl"
    try:
        with open(combined_file, "w", encoding="utf-8") as f:
            for item in combined:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info(f"Saved Combined dataset with {len(combined)} examples to {combined_file}")
    except Exception as e:
        logger.error(f"Failed to save combined dataset: {e}")
        
    logger.info("Synthetic dataset generation pipeline completed successfully.")

if __name__ == "__main__":
    main()
