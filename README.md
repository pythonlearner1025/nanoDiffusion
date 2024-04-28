# nanoDiffusion 

A nanoscale implementation of diffusion models, with experimental code to explore reasoning in text diffusion models.

Diffusion models (referring to Denoising Diffusion Probabilistic Models) are effective at modeling complex data distributions, allowing us to draw samples from it. For example, once we train a diffusion model over a dataset of many RGB cat images, we can then generate new cat images by sampling from it. 

Diffusion models work by adding noise to the input sample T times autoregressively in the *forward process* or the *diffusion process*, and then removing the added noise T times in the *reverse process* or the *denoising process*. We can think of the entire process as traversing a Markov Chain back and forth, where in the forward process we go from our clean sample to gaussian noise, and in the reverse process we go from gaussian noise back to our clean sample. 

We only care about learning the reverse process: ultimately, we want a generative model that "generates" new samples by iteratively denoising random noise until we get samples that appear to have originated from the training dataset.

# reasoning in text diffusion models

I adopt ml-planner's VAE approach for text diffusion, where the diffusion process happens not over raw text samples but over compressed latents of text. ```Bert-Base``` is used as the encoding model to compress text into latents, and ```GPT2-Decoder``` is used as the decoding model to autoregressively decode latents back into text.

# how to run

To train on FashionMNIST, run ```python image.py```

To train over text datasets, pick a dataset then run ```python text.py --dataset YOUR_CHOICE```

```json
CHOICES = {
    "hotel_reviews" : "Trip Advisor Hotel Reviews Dataset. Good for testing summarization",
    "commonsense" : "WinoGrad dataset. Good for testing commonsense reasoning",
    "math" : "GSM8K dataset. Good for testing math" 
}
```

# References
- ml-planner
- diffusion of thought
- DDIM https://arxiv.org/pdf/2010.02502
- DDPM https://arxiv.org/pdf/2006.11239



