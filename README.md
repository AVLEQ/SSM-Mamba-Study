# SSM-Mamba-Study

## Week 1: Foundations of State Space Models (SSMs)
The goal of this week is to understand why we are moving away from Transformers for long sequences and how SSMs bridge the gap between continuous math and discrete code.

### Key Concepts Learned:
* **The Transformer Problem:** Quadratic complexity $O(L^2)$ means they get too slow as data gets longer.
* **SSM Motivation:** Linear complexity $O(L)$ allows for processing massive sequences (like audio or long books).
* **The Continuous System:** * $h'(t) = Ah(t) + Bx(t)$ (How the hidden state changes)
  * $y(t) = Ch(t)$ (How we get an output)

### Practical Work:
* Visualized **Discretization** using Python to see how the step size ($\Delta$) affects signal quality.# SSM-Mamba-Study
