# SSM-Mamba-Study: From Attention to Selective State Spaces

This repository documents my research and implementation journey into State Space Models (SSM) and the Mamba architecture. The goal is to bridge the gap between traditional Transformers and the next generation of linear-time sequence models.

---

## Quick Links

* **[Interactive Research Notebook (Week 1)](https://colab.research.google.com/drive/1WYiMhxXD_9q0G3BFkMeZFPzZ2RTqsFaO?usp=sharing)**
* **[S4 Implementation and Midterm Report (Week 2)](https://colab.research.google.com/drive/1SNYH7mojMq6TIYx_T4buu-6II_TPQxvB?usp=sharing)**
* **Seminal Paper:** [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)

---

## Week 1: Foundations and Motivation

### Objective

Understand the mathematical limitations of Transformers ( complexity) and explore the foundational S4 model as a solution for long-context sequences.

### Key Concepts Explored

#### 1. The Transformer Bottleneck

I visualized the "Quadratic Wall." As sequence length increases, the attention mechanism's memory and compute requirements grow exponentially, making long-form audio or genomics processing nearly impossible on standard hardware.

#### 2. Discretization: Bridging Math and Code

I learned how to transform continuous-time differential equations into discrete steps that a computer can process using the Bilinear Transform.

* Formula: 

#### 3. The HiPPO Matrix

I implemented the High-Order Polynomial Projection Operators (HiPPO). This structured matrix allows the model to compress history into Legendre polynomials, solving the "vanishing memory" problem inherent in standard RNNs.

#### 4. The Mamba Breakthrough: Selectivity

I explored the transition from S4 (Time-Invariant) to Mamba (Selective).

* Selection Mechanism: Making parameters (Delta, B, C) dependent on the input x allows the model to selectively remember or ignore information based on content.
* Hardware Awareness: Understanding how custom CUDA kernels keep the hidden state in SRAM to bypass the "Memory Wall" between HBM and the GPU processor.

---

## Week 2: S4 Base Implementation (Midterm)

### Objective

Construct a functional Structured State Space Model (S4) in JAX/Flax capable of processing high-dimensional sequences.

### Technical Achievements

#### 1. Numerical Stability and Solvers

I identified and resolved numerical instability issues (NaNs) by replacing standard matrix inversion with more robust linear solvers (jnp.linalg.solve). I also implemented diagonal regularization to avoid mathematical poles in the complex plane.

#### 2. Frequency Domain Duality

I implemented a frequency-domain generating function and applied the SSM kernel via Fast Fourier Transforms (FFT). This allowed me to achieve  training complexity, bypassing the limitations of sequential recurrence.

#### 3. Global Average Pooling

I integrated Global Average Pooling into the classification head. This modification proved essential for capturing spatial features across the long sequences required for image classification tasks.

---

## Implementation Progress

* [x] Benchmark Transformer vs. SSM scaling.
* [x] Implement and visualize the HiPPO Matrix.
* [x] Build functional S4 Layer in JAX/Flax.
* [x] Solve numerical stability issues (NaNs) in discretization.
* [x] Validate model on Sequential MNIST (L=784).
* [x] Validate model on Fashion-MNIST (L=784).
* [x] Validate model on CIFAR-10 (L=3072).

---

## Study Resources

* **The Annotated S4:** [ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/annotated-s4/)
* **Mamba Repository:** [SSMphony AI Voice Generation](https://github.com/SSM11011/SSMphony-AI-voice-generation-using-Mamba-SSMs)
* **Video Deep Dive:** [Mamba: Linear-Time Sequence Modeling](https://www.youtube.com/watch?v=9dSkvxS2EB0)# SSM-Mamba-Study: From Attention to Selective State Spaces

This repository documents my research and implementation journey into **State Space Models (SSMs)** and the **Mamba architecture**. The goal is to bridge the gap between traditional Transformers and the next generation of linear-time sequence models.

---

##  Quick Links
* **[Interactive Research Notebook (Week 1)](https://colab.research.google.com/drive/1WYiMhxXD_9q0G3BFkMeZFPzZ2RTqsFaO?usp=sharing)** 
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

