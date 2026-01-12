---
date: '2026-01-11'
draft: false
title: 'Transformer Part I: the details that actually matter'
---

Transformer is arguably the most influential AI innovation in the past decade, serving as the foundation as our modern day LLM models. It also revolutionize some adjacent fields such as computer vision, recommender system etc. In this note, we are going to revisit this revolutionary technology with a focus on discussing a set of details that matter. 

# Transformer Recap

Transformer was first proposed as a encoder-decoder model to solve the machine translation problem. Take English to Chinese language translation as our example, encoder is basically encoding text tokens in a way such that both individual token meaning and joint dependency can be captured. The output is the tensor representation of English sentences. The decoder is responsible for text generation and learning relationship between English and Chines. The text generation process is basically next token prediction which is achieved by masking unseen tokens so attention is only paid to preceding tokens in the sentence, namely causal masking. The English and Chineses relationship is learned by cross attention where query is Chinese tokens, which pays more attention to tokens from encoded English output with high attention weights. 

In this note, we are only interested the decoder part as the decoder-only model is actually being adopted by most succesful LLMs such as GPT. Most building blocks and technical details are similar between encoder and decoder.

# The decoder-only transformer overview

Let's think about what are some most important elements going from tokens to next token generation.
1. Token and its embedding: we need tokenization to covert sentences into list of index (the maximum index is the token vocab size), and we need embedding lookup table to lookup embedding for each index. 
2. Token and their relative position: we need to consider the relative position of those tokens as putting words together in different order have different meanings. 
3. Depedency learning: we need to know the joint relationship between tokens, that is, for the current token, which tokens in the sentence looks more relevant. 
4. Diversified dependencies: depencies have different patterns: subject -> verb: the dog runs, verb -> direct:  object: i buy a car, phonatactic pattern etc.

For each of the above 4 elements, transformer has corresponding supporting components. From input to output: 
1. sequence of tokens and their embeddings: for training, we chop text corpus into sequences, and let's take asequence of tokens as our input unit. The sequence of tokens will generate an embedding matrix by looking up token embedding table.
2. position encoding: position encoding will encode the relative postion from 0 to L-1 where L is the context window/sentence length. The resulting position embedding will be added to sequence token embeddings which will serve as the input for self attention module next. 
3. masked self attention: the input for self attention is still individual token sequence without talking to each other. The most important role of self attention is to force the current token talks to earlier tokens in the context and transform current token to make it context-aware. This is the heart of the transformer and we will delve into it later.
4. multi-head attention: having only one attention head can only learn one type of dependency, that's why it's dessirable to have multiple heads for learning different dependent patterns. 

In addition to the above components, the transformer also implemented add & norm where add is using x + f(x) type of residual connection to enhance optiomization for deep networks, particularly addressing gradient vanishing problem. The norm here refers to layer norm which normalize token-wise embeddings, we will explain later why this matters. The multi-head attention blocks are stacked sequentially to increase the depth of the network. Overall design is like below.

![Transformer attention](transformer.png)

# Technical deep dive

Now we dive into some key technical aspects of transformers to understand the intuition behind them.

## Self Attention 

I remember the first time i read the "Attention is all you need" paper years back, i was really puzzled about what Q,K,V really means. It is easier if we first start pondering what's needed in the output. So, the input now is embedding vectors for each individual tokens plus some positional embeddings, which is still token-wise information without communicating with each other. What's really desired is to let token to communicate and transform itself in such a way that more relevant tokens will play a bigger role in the final output. Mathematically, we need to consider current_token_emb_output = sum over all tokens { similarity(current_token, token) * token }. So the target we are transformation is our Query, the token for which is used for similarity calculation is Key, and the token value that's being averaged over is Value. The naming makes a lot of sense because Query is trying to search for keys that are most relevant and the corresponding values are convolutionized by the affinity between query and key. 

At the individual token level, $ output = \sum_{i=1}^L sim(q, k_i) v_i $ where $L$ is the length of the sequence and $q, k_i, v_i$ are $d$ dimensional vectors. A natural way to calculate sim is dot product $ q^T k_i = \sum_{j=1}^d q_{j} k_{ji}$, assuming each element of $q$ and $k$ are mutually i.i.d Normal(0,1), the variance of the dot product is $d$, to make the dot product standardized normal, we need to rescale it therefore $sim = q^Tk_i/\sqrt{d}$. However, the $sim$ function is only standardized across $d$ token embedding entries, not really normalized across $L$ tokens, namely, we need to rescale $sim(q, k_i)$ such that $\sum_{i=1}^L sim(q, k_i) = 1$. The rescaling is important to bound the variance the final output. A typical way to standardize $sim$ is to use softmax, which is differentiable compared to absolute values. Now, let's switch to matrix notation for compactness, $Q = [q_1^T, ..., q_L^T]^T$, similar for $K, V$. The final output $(L, d)$ matrix is 
$output = Softmax(QK^T/\sqrt{d})V$.   

Maybe one detail worth mentioning here is going from the input embedding matrix $x$ to $q, k, v$, we add learnable projection matrix $W^q, W^k, W^v$ to make $q, k ,v$ data driven and learnable, namely, $q, k, v = xW^q, xW^k, xW^v$. The dimension for matrix $W$ usually is set to $(d, d/H)$ where $H$ is the number of attention heads. The reason is for consistency because after self attetion transformation from individual head, we'll concatenate output from each head and $d/H * H = d$ will make the output has the same dimension as token embedding dimension.

## Masking

Let's say if the sentence "The biggest animal on earth is the blue whale, with a heart the size of a small car" is in our training data, now if someone ask transformer "the biggest animal on earth is", we need to sample from the tokens and aim to ensure $P(the \quad blue \quad whale| the \quad biggest \quad animal \quad on \quad earth \quad is)$ has an extremely high probablity. So you see, the setenece after "the blue whale" like "with a heart..." does not really matter during token generation. Therefore, if our goal is to generate token with maximum likelihood given preceding tokens in the sentence, we have to ensure the current token only pay attention to earlier tokens. This requires revising the attention calculation.

Let's say our query $q_t$ is at postion $t$ in the sequence, then $q_t = \sum_{i=1}^t sim(q_t, k_i) v_i$ not sum over all L tokens. The matrix form is shown in the picture above with the lower triangular matrix form. 

To make sure the masking is consistent, we also need to shift the input by 1 position, say if the target sentence is Target = "The biggest animal on earth is the blue whale", Input =  "<\START> The biggest animal on earth is the blue" if our token is just word. The reason for doing so is because the first token "The" cannot pay attention to itself so we need to create a special token <\START> to act as its preceding token. This is logical because generating "The" is based on empty context.  


## Add & Norm

The Add & Norm refers to residual connection and layer norm. Let's first look at the residual connection: $x + f(x)$, the idea is simple, for the transformation, the output is not $f(x)$, instead its $y = x + f(x)$. This equivalently is using $f(x) = y - x$ to learn the residual between output and input. The residual connection is applied to both attention head and FFN. Why this is important ? Let's consider a deep neural networks $y = f_1 (f_2 ..,(f_M (x,w)))$, then $\frac{dy}{dw} = \partial f_1 * {...}  *\partial f_{M}$, and if lots of the partial derivatives are between 0 and 1, the gradient of w will vanish as a result, the deeper the network, the more likely it will happen. Now if we switch to residual connection $y = f_1 (. + f_2(..., x + f_M(x, w)))$, 
$\frac{dy}{dw} = (1 + \partial f_1) * {...}  * (1+\partial f_{M})$ which won't suffer from gradient vanishing thanks to adding the input back to the transformation. 

For the layer norm, besides generally stablizing the network and reducing the training time. One important reason why it's so crucial for transformation is because it forces the normalization chain is consistently maintained, as aformentioned we apply $1/\sqrt{d}$ scaling factor to dot product to standardize variance, which is based on the ***ASSUMPTION*** that each entry in $q$ and $k$ vector are normalized to standard normal distribution. The make the asusmption hold, we need to apply layer norm to first standardize the embedding vector for each token in the sequence so that $x = (x_1, ..., x_L)$ matrix where $x_i$ is d dimensional, $x_{ij} = \frac{x_{ij} - mean(x_i)}{std(x_i)}$, here we ignored the tunable parameters in layernorm for illustration purpose. Then $q, k , v = x W^q, x W^k, x W^v $ are standardized as a result.    


## Model Complexity 


## Optimization opportunities




# Implementation & Experments















