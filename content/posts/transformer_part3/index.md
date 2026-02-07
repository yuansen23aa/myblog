---
date: '2026-02-06'
draft: false
title: 'Transformer Part III: Infra Consideration'
---

In Part I & II, we covered basics of the transformer archecture and its implementation. The focus of Part III is infra aspects of transformer. Particularly, we will discuss about:
1. How KV Cache improves efficiency
2. How FlashAttention improves memory and efficiency
3. Basic idea behind vLLM and PagedAttention
4. DPP & FSDP 
5. Quantization.
6. Sparsity & Low Rank. 


# How KV Cache improves efficiency

Let's first read the autoregressive generation implementation from
Part II to see where the efficiency deficit is from. The third from
the last line where we add next_id to x_id_matrix is to extend the sequence
to include the generated token for the next round inference. But the problem is 
the next round inference will consider the entire x_id_matrix including the tokens already inferenced in the prev round. 

Keep in mind the decoder-only model is causal transformer, which says the current token's state (Q, K, V, X) only depends on earlier position tokens. This suggests once 
the token's state are computed, we can the its state for later usage because their value won't change as we run inference for next tokens. 

````python
   def generate(self, x_id_matrix, max_new_tokens, top_k):
        B, L = x_id_matrix.shape
        out = x_id_matrix.clone()
        for _ in range(max_new_tokens):
            if x_id_matrix.shape[1] > self.block_size:
                x_id_matrix = x_id_matrix[:, -self.block_size:]   
            logits, _ = self(x_id_matrix)
            # focus only on the last time step
            logits = logits[:, -1, :] # (B, vocab_size)
            top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
            top_k_probs = F.softmax(top_k_logits, dim=-1) # (B, vocab_size)
            # sample from the distribution
            sampled = torch.multinomial(top_k_probs, num_samples=1) # (B, 1)
            next_id = torch.gather(top_k_indices, -1, sampled)  # (B, 1)
            # append sampled index to the running sequence
            # THIS IS WASTING THE COMPUTE
            x_id_matrix= torch.cat((x_id_matrix, next_id), dim=1) # (B, L+1)
            out = torch.cat((out, next_id), dim=1)
        return out

````


Instead, the KV cached version will look like below (incomplete implementation).
The idea is that, the forward pass of the attention block will return kv_cache values,
For the prefilling stage where the model inference is performed on the user prompt, a full forward pass will be executed and K, V for the prompt tokens will be cached. Then imagine if we generate the next_token, we only need to do forward/self(next_token, kv_cached) because K,V don't need to be recomputed and dot(q_next, [k_cached, k_next]) [v_cached, v_next] will be executed. Then k_cached = [k_cached, k_next]. This is simply just trading off efficiency with memory.  

````python
def generate(self, x_id_matrix, max_new_tokens, top_k):
    """
    KV-cached autoregressive generation.

    Assumptions:
      - self(idx, kv_cache=None, use_cache=False) -> (logits, kv_cache)
        where logits is (B, T, vocab) and kv_cache is a per-layer structure
        that stores past K/V and can be fed back in.
      - If your forward has a different signature, adapt the two calls below:
          (1) prefill on the prompt
          (2) decode one token at a time with kv_cache
    """
    B, L0 = x_id_matrix.shape
    device = x_id_matrix.device

    out = x_id_matrix.clone()

    # ---- 1) Prefill: run the (possibly truncated) prompt once, build kv_cache ----
    if x_id_matrix.shape[1] > self.block_size:
        x_ctx = x_id_matrix[:, -self.block_size:]
    else:
        x_ctx = x_id_matrix

    # logits: (B, T, vocab), kv_cache: filled for T tokens
    logits, kv_cache = self(x_ctx, kv_cache=None, use_cache=True)

    # ---- 2) Decode: only feed the last generated token each step, reuse kv_cache ----
    for _ in range(max_new_tokens):
        # focus only on the last time step
        last_logits = logits[:, -1, :]  # (B, vocab)

        # top-k sampling
        top_k_logits, top_k_indices = torch.topk(last_logits, k=top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        sampled = torch.multinomial(top_k_probs, num_samples=1)           # (B, 1)
        next_id = torch.gather(top_k_indices, -1, sampled)                # (B, 1)

        # append to output sequence (full history for the caller)
        out = torch.cat((out, next_id), dim=1)

        # ---- Feed ONLY the new token to the model, with kv_cache ----
        logits, kv_cache = self(next_id, kv_cache=kv_cache, use_cache=True)
    return out
````

