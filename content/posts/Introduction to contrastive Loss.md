## Contrastive Loss

Contrastive Loss is a widely used objective in **metric learning** and **contrastive learning**.  
Its goal is to learn an embedding space where **similar samples are close together**, while **dissimilar samples are far apart**.

The loss operates on **pairs of samples**:
- **Positive pairs**: two samples that should be considered similar
- **Negative pairs**: two samples that should be considered different

Given a pair of embeddings and a binary label, contrastive loss:
- penalizes **large distances** between positive pairs
- penalizes **small distances** between negative pairs (up to a margin)

This encourages the model to learn representations that are discriminative and geometry-aware.

---

### A Common Formulation

A typical contrastive loss can be written as:

\[
L = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2
\]

where:
- \(d\) is the distance between two embeddings
- \(y = 1\) for a positive pair, \(y = 0\) for a negative pair
- \(m\) is a margin hyperparameter

---

### Why It Matters

Contrastive Loss is especially useful when:
- explicit labels are scarce
- similarity relationships matter more than class boundaries

It is commonly used in:
- representation learning
- retrieval and recommendation systems
- self-supervised learning (e.g., SimCLR, CLIP)
- Siamese and dual-encoder models

By shaping the embedding space directly, contrastive loss often leads to representations that transfer well to downstream tasks.
