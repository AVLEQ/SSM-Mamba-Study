# SSM-Mamba-Study: From Attention to Selective State Spaces

This repository documents my research and implementation journey into **State Space Models (SSMs)** and the **Mamba architecture**. The goal is to bridge the gap between traditional Transformers and the next generation of linear-time sequence models.

---

##  Quick Links
* **[Interactive Research Notebook (Week 1)](https://colab.research.google.com/drive/1WYiMhxXD_9q0G3BFkMeZFPzZ2RTqsFaO?usp=sharing)** *
* **Seminal Paper:** [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)

---

##  Week 1: Foundations & Motivation

###  Objective
Understand the mathematical limitations of Transformers ($O(L^2)$ complexity) and explore the foundational S4 model as a solution for long-context sequences.

###  Key Concepts Explored

#### 1. The Transformer Bottleneck
Visualized the "Quadratic Wall." As sequence length increases, the attention mechanism's memory and compute requirements grow exponentially, making long-form audio or genomics processing nearly impossible on standard hardware.

#### 2. Discretization: Bridging Math and Code
Learned how to transform continuous-time differential equations into discrete steps that a computer can process using the **Bilinear Transform**.
* **Formula:** $h_t = \bar{A}h_{t-1} + \bar{B}x_t$

#### 3. The HiPPO Matrix (The "Secret Sauce")
Implemented the **High-Order Polynomial Projection Operators (HiPPO)**. This structured matrix allows the model to compress history into Legendre polynomials, solving the "vanishing memory" problem inherent in standard RNNs.



#### 4. The Mamba Breakthrough: Selectivity
Explored the transition from **S4 (Time-Invariant)** to **Mamba (Selective)**. 
* **Selection Mechanism:** Making parameters $(\Delta, B, C)$ dependent on the input $x$ allows the model to selectively remember or ignore information based on content.
* **Hardware Awareness:** Understanding how custom CUDA kernels keep the hidden state in **SRAM** to bypass the "Memory Wall" between HBM and the GPU processor.



---

##  Implementation Progress
- [x] Benchmark Transformer vs. SSM scaling.
- [x] Visualize Discretization effects on signal fidelity.
- [x] Implement and visualize the HiPPO Matrix.
- [x] Simulate the Recurrent "Summary Notebook" update loop.
- [x] Visualize the Selective Delta ($\Delta$) mechanism.

---

##  Study Resources
* **The Annotated S4:** [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/annotated-s4/)
* **Mamba Repository:** [SSMphony AI Voice Generation](https://github.com/SSM11011/SSMphony-AI-voice-generation-using-Mamba-SSMs)
* **Video Deep Dive:** [Mamba: Linear-Time Sequence Modeling](https://www.youtube.com/watch?v=9dSkvxS2EB0)

---

## 🔭 Next Steps (Week 2)
Moving from theoretical visualization to a **Base Level Implementation** of the S4/Mamba layer in JAX/PyTorch.
