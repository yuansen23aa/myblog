---
date: '2026-01-15'
draft: false
title: 'Transformer Part II: The implementation & fun experiments'
---

In this episode, we will go into some details of the causal transfomer implementation. Some toy experiment results are shown to analyze transformer, in an attempt to understand what drives the performance. We will not go through every single line of implementation. The code used for illustration can be found: https://github.com/yuansen23aa/GPT-learning/blob/main/basic_gpt.ipynb, which largely follows Anrej Karpathy's nanoGPT implementation with some modifications. So Let's dig in.

We use those terms interchangeably: block size = sequence length, causal attention = masked attention, causal transformation = decoder-only transformer. 

# Overall structure

Let's peel the onion and see layer by layer of the overall structure.  

- GPT  = Embbeding Lookup -> **Attention Blocks** -> LM

- Attention Blocks = [**Multihead Self Attention** -> Feed Forward Network]

- Multihead Self Attention = Concat All Single **Masked Self Attention** Head

Following this structure, we need to implement the following classes: GPT, Attention Blocks, Multihead Attention, Masked Self Attention, Feed Forward Network.

- **GPT class**:
    - **Forward**: map  X id matrix (batch size, block size) into token embedding tensor (batch size, block size, emb size), create position embedding (block size, emb size) and broadcast it to (batch size, block size, emb size) and add position and token embedding together as the input for attention blocks. The attention blocks spit out (batch size, block size, emb size) tensor as LM head input, LM will map emb size to vocab size as logits vector for all batch size * block size tokens. 
  - **Generation**: autoregressive token generation based on input context. 

- **Attention blocks**: since we first layernorm input x for multihead attention, then run residual connection x + mha(x). Similarly, we do layernorm for FFN and residual connection sequetially. 

- **Multihead attention**: the main role is concatenation

- **Masked self attention**: implement Q, K, V and masked attention head. 

- **Feed Forward Network**: projecting multihead attention output to 4*emb size hidden layer with GeLu, then projecting to the output with size emb size. 


# Bug prone components

Instead of checking the entire implementation, we highlight the parts that are essential but error prone. 


## Shift Input Position by 1
````python

class dataloaderlite:
    def __init__(self, data, block_size, batch_size, device, shuffle=True, tag=None):
      ...

    ...
    def __next__(self):
       ...   
        x = torch.stack([self.data[i:i+self.L] for i in idx])
        y = torch.stack([self.data[i+1:i+self.L+1] for i in idx])    
        return x.to(self.device), y.to(self.device)

````
The last three lines of the dataloader we implemented are crucial because it defines y as 1 poisition ahead of x so that the model is properly defined. Additionally, it registers the input and target to the device (cpu or cuda) used.


## Pay attention to the pytorch status/mode

````python
@torch.no_grad()
def loss_estimation(model, grad_norm=False):
    output = {}
    model.eval()
    ....
    model.train()
    return output

````

Whe decorator @torch.no_grad() disables gradient calculation via Not creating computation graph to speed up the inference. When eval started, we enter eval mode so that functionalities such as dropout can be called correctly. Finally, remember to switch back to train mode when eval is done.

## Input reshaping for cross entropy loss

````python
class GPT(nn.Module):
    ...
    def forward(self, x_id_matrix, y_id_matrix=None):
        ...
        else:
            targets = y_id_matrix.view(Batch_size * Block_size)
            logits = logits.view(Batch_size * Block_size, Vocab_size)
````

The reshaping flatterns both targets and logits so inputs can be accepted by the cross entropy loss. Bascially, you can think of Batches of Blocks as a single long block by stitching batches one after another. Given the memory is laid out by row-first internally, the view() will treat the last dimension of logits unchanged (vocab size) and will collaps the first two dimensions.  

## Autoregressive Generation 
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
            x_id_matrix= torch.cat((x_id_matrix, next_id), dim=1) # (B, L+1)
            out = torch.cat((out, next_id), dim=1)
        return out

````

## Masked Attention

````python
class MaskedAttention(nn.Module):
    def __init__(self, num_heads, embed_size, block_size):
        super().__init__()
        self.head_size = embed_size // num_heads
        self.q_proj = nn.Linear(embed_size, self.head_size, bias=False)
        self.k_proj = nn.Linear(embed_size, self.head_size, bias=False)
        self.v_proj = nn.Linear(embed_size, self.head_size, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q, k ,v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        kt = k.transpose(-1,-2)
        att = q @ kt/ (self.head_size**0.5)
        mask = torch.tril(torch.ones(B, L, L, device=device, requires_grad=False))
        att = att.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(att, dim=-1)
        weights = self.dropout(weights)
        att_output = weights @ v
        return att_output 
````