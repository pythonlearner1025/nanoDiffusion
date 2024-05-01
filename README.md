# nanoDiffusion 

*NOTE: this repo is incomplete* 

A nanoscale implementation of diffusion models, with experimental code to explore reasoning in text diffusion models.

```textgen.py``` is a minimal implementation of Apple's [ml-planner](https://github.com/apple/ml-planner?tab=readme-ov-file)

```imagen.py``` is a minimal implementation of [minDDPM](https://github.com/aleksandrinvictor/minDDPM)  

# how to run

First, install dependencies: 

```pip install -r requirements.txt```

To train on FashionMNIST, run 

```python imagen.py```

To train on text datasets, run 

```python textgen.py --dataset tripadvisor``` 

The default accelerator device is ```cuda```. If you don't have a CUDA gpu, set device to ```cpu```

*NOTE: tripadvisor is the only text dataset supported rn*

# diffusion models

Diffusion models (referring to Denoising Diffusion Probabilistic Models) are effective at modeling complex data distributions, allowing us to draw samples from it. For example, once we train a diffusion model over a dataset of many RGB cat images, we can then generate new cat images by sampling from it. 

Diffusion models work by adding noise to the input sample T times autoregressively in the *forward process* or the *diffusion process*, and then removing the added noise T times in the *reverse process* or the *denoising process*. We can think of the entire process as traversing a Markov Chain back and forth, where in the forward process we go from our clean sample to gaussian noise, and in the reverse process we go from gaussian noise back to our clean sample. 

We only care about learning the reverse process: ultimately, we want a generative model that "generates" new samples by iteratively denoising random noise until we get samples that appear to have originated from the training dataset.

# reasoning in text diffusion models

I adopt ml-planner's VAE approach for text diffusion, where the diffusion process happens not over raw text samples but over compressed latents of text. ```Bert-Base``` is used as the encoding model to compress text into latents, and ```GPT2-Decoder``` is used as the decoding model to autoregressively decode latents back into text.

Diffusion of Thought (DoT) puts an auto-regressive spin onto diffusion models by concatenating diffused outputs R1 with conditioning text C to form C' = [C, R1]. The output of C' is then R2, and C'' = [C, R1, R2], etc. The R's can be interpreted as reasoning "thoughts" the model chains, much like how thoughts are chained in Chain-of-Thought reasoning in LLMs. 

**Incomplete ideas on DoT**
- chain of thought reasoning is equivalent to learning good state-action trajectories, which is already implicit in diffusion model's markovian formulation
- in DoT, conditioning text is what guides generated text. analogous to prompt in llms. however, unlike in llms the "prompt" conditions each markov transition of the entire generated text in the reverse process. for example, different "snapshots" of text in varying stages of being denoised can be sampled and added to the conditioning text, allowing later steps to correct errors made in earlier steps. 
- modeling reasoning explicitly as a markov decision process will be helpful both for understandability + ability to build upon it. or maybe i just didn't learn the bitter lesson?  

See the [Diffusion of Thought](https://github.com/HKUNLP/diffusion-of-thoughts) repo for more. They were able to show interesting results like diffusion models self-correcting errors they made when solving math problems from GSM8K.   

# maybe
- [ ] add comments
- [ ] add DoT support 

# references
- [ml-planner](https://arxiv.org/abs/2306.02531)
- [Diffusion-of-Thought](https://arxiv.org/abs/2402.07754)
- [DDIM](https://arxiv.org/abs/2010.02502) 
- [DDPM](https://arxiv.org/abs/2006.11239)



