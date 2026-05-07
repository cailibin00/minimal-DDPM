# Overall API Design Pattern
## Model Level
- Core Idea:
A model architecture itself **cannot determine** whether it predicts v, score, or noise.
Therefore, during the model's forward inference phase, an **external signal** is required to determine how to perform integration.
However, this signal **must not be encoded** as an initialization parameter of the model itself.

Thus, an **external wrapper** is needed to implement this logic. The wrapper must:
1. Contain the model itself, and the model must be capable of computing the loss (this is MSE loss, which is straightforward and also my personal design preference).
2. Support multiple prediction types.
3. Support forward sampling (compute using the model's output).

Therefore:
- **GaussianDiffusion**:
  1. Contains the model
  2. Supports model inference
  3. Supports various sampling methods
  4. Computes the loss
  5. generate data (input datasets , according to the mode)


---

## Data Generation
Let's first examine the DDPM process of diffusion:
Given the real data $p_0$, we generate noise $\epsilon$, and given a timestep $t$, apply it to $x_t$.

Thus, we have:
1. **Input**: $x_t$, $t$
2. **Label**: $\epsilon$

Actually:
$x_t = \text{function}(x_0, \epsilon)$
So the input contains the information of $x_0$ that the model is trained to learn.

However , this process happens in GaussianDiffusion but not here.
Because GaussianDiffusion has the generate mode , such as $epsilon%$ ,v  prediction , so we need to generate data in the model level.

---

## Trainer Level
- Core Idea:
We **do not need to compute the loss** here.
The trainer only needs to handle training configurations such as the optimizer.
Additionally, it must be compatible with data loaders.

One design principle here:
**Ensure the dataset and model modes are matched.** If they do not match, training cannot proceed.