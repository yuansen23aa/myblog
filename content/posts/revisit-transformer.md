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











