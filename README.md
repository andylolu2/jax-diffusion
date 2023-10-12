# JAX Diffusion

Unofficial implementation of 
[Denoising Diffusion Probabilistic Models (DDPM)][1] in JAX and Flax.

[Denoising Diffusion Implicit Models (DDIM)][2] sampling is used as well.

[1]: https://arxiv.org/abs/2006.11239
[2]: https://arxiv.org/abs/2010.02502

## MNIST

| Real      | Generated |
| --------- | --------- |
| ![img][3] | ![img][4] |

[3]: https://user-images.githubusercontent.com/66584117/184720519-1cfaa9ba-9e8d-4dd4-bf3a-f0bc3e15e7c5.png
[4]: https://user-images.githubusercontent.com/66584117/184720531-85dfe572-5ddd-4432-98b4-7ee11434067d.png

### Training details

Model has 5.46M parameters, trained on Colab (T4) for 100K steps with batch size 128
in 8.5 hours.

Full hyperparameters can be found in 
[configs/mnist.py](jax_diffusion/configs/mnist.py).

## Fashion MNIST

| Real      | Generated |
| --------- | --------- |
| ![img][5] | ![img][6] |

[5]: https://user-images.githubusercontent.com/66584117/185258572-a51e78aa-8296-471e-b5e7-4049f541134b.png
[6]: https://github.com/andylolu2/jax-diffusion/assets/66584117/1e10294c-0be0-4c99-b53f-2480c9035ff4

### Training details

Model has 9.70M parameters, trained on Kaggle (TPUv3-8) for 40K steps with batch size 128 in 2.5 hours.

Full hyperparameters can be found in 
[configs/fashion_mnist.py](jax_diffusion/configs/fashion_mnist.py).

## Celeb A

### Results

| Real      | Generated |
| --------- | --------- |
| ![img][7] | ![img][8] |

[7]: https://user-images.githubusercontent.com/66584117/187550442-95287154-6598-4e2e-89f2-567c53230cc9.png
[8]: https://user-images.githubusercontent.com/66584117/187550318-f92f7778-2b3e-4167-a14b-c6ba0a90c772.png

### Training details

Due to compute constraints, the model is only trained for 64 x 64 images.

Model has 72.70M parameters, trained on Kaggle (P100) for 60K steps with batch size 64 
in 22 hours.

Full hyperparameters can be found in [configs/celeb_a64.py](jax_diffusion/configs/celeb_a64.py).
