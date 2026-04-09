# Making Pixels from Scratch
### A plain-English guide to Generative AI in Computer Vision

> *GANs, VAEs, and Diffusion Models — explained like you're five.*

**Computer Vision & Generative AI — Mini TAF · S8 · April 2026**

---

## Table of Contents

1. [What is Generative AI in Computer Vision?](#1-what-is-generative-ai-in-computer-vision)
2. [Why Use Generative AI?](#2-why-use-generative-ai)
3. [The Three Main Architectures](#3-the-three-main-architectures)
   - [GAN — The Counterfeiter vs. The Detective](#31-gan--the-counterfeiter-vs-the-detective)
   - [VAE — The Artist Who Paints from Memory](#32-vae--the-artist-who-paints-from-memory)
   - [Diffusion Models — The Sculptor of Chaos](#33-diffusion-models--the-sculptor-of-chaos)
4. [Side-by-Side Comparison](#4-side-by-side-comparison)

---

## 1. What is Generative AI in Computer Vision?

Most AI systems you've heard of are **discriminative** — they look at an image and tell you what's in it. Generative AI flips the script: it *creates* brand-new images from nothing (or from a text description).

Traditional CV models are like art critics. They look at a painting and say "that's a cat." Generative AI models are like the artist — they can paint a cat (or an astronaut riding a horse, or a medieval city at night) entirely on their own.

> 💡 **Simple analogy:** Imagine teaching a child to recognise dogs vs. cats — that's classic CV. Now imagine the child can *draw* any dog or cat you describe, in any style, that has never existed before. That's Generative AI in Computer Vision.

Technically, a generative model learns the underlying statistical distribution of a dataset of images. Once it understands *how images are structured*, it can sample from that distribution to create new, realistic ones. The model isn't memorising pictures — it's learning the rules of how pixels relate to each other, and using those rules to invent.

---

## 2. Why Use Generative AI?

The ability to generate realistic images unlocks a wide range of applications that were impossible or impractical with classical CV approaches.

| Use case | Description |
|---|---|
| **Data augmentation** | Generate synthetic training samples when real data is scarce or expensive |
| **Image editing** | Inpainting, style transfer, super-resolution, colourisation |
| **Creative tools** | Text-to-image generation, concept art, product design, fashion |
| **Medical imaging** | Synthesise rare pathology images to train diagnostic models |

In short: Generative AI lets us create what doesn't exist yet — and that matters enormously in both research and industry.

---

## 3. The Three Main Architectures

There are three dominant approaches to generative modelling in computer vision. Each has a different philosophy about *how* to teach a machine to invent images.

| | GAN | VAE | Diffusion |
|---|---|---|---|
| **Symbol** | ⚔ | ◎ | ❄ |
| **One line** | Two networks compete | Encode → sample → decode | Reverse a noising process |

---

### 3.1 GAN — The Counterfeiter vs. The Detective

> *Idea: put two networks in direct competition and let them train each other.*

A GAN is composed of two neural networks playing a minimax game:

- The **Generator** tries to produce fake images that look real, starting from random noise.
- The **Discriminator** tries to tell real images apart from the Generator's fakes.

They train simultaneously — the Generator improves by fooling the Discriminator, while the Discriminator improves by catching fakes.

> 💡 **ELI5 analogy:** A forger (Generator) tries to paint fake Picassos. An art detective (Discriminator) checks every canvas and says "real or fake?". The forger studies every rejection and gets better and better. Eventually, the forgeries become indistinguishable from the originals.

**Flow:**
```
Random noise → Generator → Fake image → Discriminator → Real / Fake?
```

**Advantages:**
- Extremely fast inference (single forward pass)
- Very sharp, photorealistic image quality
- Widely used in style transfer, face synthesis, super-resolution

**Limits:**
- Training is unstable — risk of **mode collapse** (Generator produces only a few image types)
- Requires careful hyperparameter tuning
- Evaluation is subjective and difficult

---

### 3.2 VAE — The Artist Who Paints from Memory

> *Idea: learn a compact, structured representation of images and sample from it to generate new ones.*

A VAE has two parts:

- An **Encoder** that compresses an image into a small vector of numbers (the *latent space*)
- A **Decoder** that reconstructs an image from that vector

The key trick: the latent space is constrained to be **continuous and normally distributed**, so you can sample random points from it and decode them into plausible new images. The model is trained by minimising both reconstruction error and a regularisation term (KL divergence).

> 💡 **ELI5 analogy:** Imagine you compress your memory of every dog you've ever seen into a single mental "concept of dogs." You can now close your eyes, pick a random point in that concept, and draw a new dog that never existed — but looks completely believable.

**Flow:**
```
Image x → Encoder → Latent (μ, σ) → sample z → Decoder → Reconstructed x̂
```

**Advantages:**
- Stable and principled training
- Smooth latent space supports **interpolation** between images
- Well-suited for anomaly detection and data augmentation

**Limits:**
- Reconstructed images tend to be slightly blurry
- Image quality generally lower than GANs or Diffusion Models

---

### 3.3 Diffusion Models — The Sculptor of Chaos

> *Idea: learn to reverse a gradual noising process, one step at a time, until a clean image emerges from static.*

Diffusion models work in two phases:

1. **Forward process:** real images have Gaussian noise added step-by-step (over hundreds or thousands of steps) until they become pure random static.
2. **Reverse process:** a neural network (typically a U-Net) is trained to predict and remove the noise added at each step.

At generation time, the model starts from pure noise and iteratively denoises it into a coherent image. This is the technology behind **Stable Diffusion**, **DALL-E**, and **Midjourney**.

> 💡 **ELI5 analogy:** Imagine taking a photograph and shredding it into confetti, one tiny cut at a time. A diffusion model watches this destruction carefully, memorising every step. At generation time, it starts with a pile of random confetti and reconstructs the photo in reverse — one piece at a time — until a coherent picture emerges.

**Flow:**
```
Real image x₀ → +noise → +more noise → ··· → Pure noise x_T
Generation    ← denoise ← denoise    ← ··· ← start here
```

**Advantages:**
- State-of-the-art image quality and diversity
- Highly controllable — conditioned on text, sketches, or other signals
- Stable and well-understood training

**Limits:**
- Generation is **slow** — requires many sequential denoising steps (20–1000)
- Higher computational cost than GANs at inference time

---

## 4. Side-by-Side Comparison

| Criterion | GAN | VAE | Diffusion |
|---|---|---|---|
| **Core idea** | Adversarial Generator vs. Discriminator | Encode → latent → decode | Reverse a noising process |
| **Image quality** | Very sharp, photorealistic | Good but slightly blurry | State-of-the-art |
| **Training stability** | Unstable (mode collapse risk) | Stable and reliable | Stable but slow |
| **Output diversity** | Can lack variety | Good | Excellent |
| **Generation speed** | Very fast (single pass) | Fast | Slow (many steps) |
| **Controllability** | Moderate | Good (latent interpolation) | Excellent (text, sketch, guidance) |
| **Best for** | Super-res, face synthesis, style | Augmentation, anomaly detection | Text-to-image, inpainting, video |
| **Famous examples** | StyleGAN, CycleGAN, Pix2Pix | β-VAE, VQ-VAE | Stable Diffusion, DALL-E, Midjourney |

> **One-sentence takeaway:**
> - **GAN** — fastest and sharpest, but fragile to train.
> - **VAE** — most mathematically principled, with smooth latent spaces, but images can look soft.
> - **Diffusion** — most powerful and controllable today, at the cost of generation speed.

---

## Conclusion

Generative AI in computer vision has gone from an academic curiosity to a technology reshaping entire industries — film, medicine, design, and beyond.

Understanding GAN, VAE, and Diffusion Models at a conceptual level is the foundation. The next step is building them — and that starts now.

---

*Computer Vision & Generative AI · S8 · April 2026*
