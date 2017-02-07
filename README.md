# DCGAN-Keras
The correct way to implement GAN in Keras.

## How this happened
DCGAN have been implemented in a lot of frameworks. However, existing Keras and Tensorflow implementations are SLOW due to duplicated computation.

Basically we want to do two things in one forward-backward pass:

1. update Wd w.r.t. D_loss
2. update Wg w.r.t. G_loss

This kind of update(different parameters w.r.t. different loss) however is not possible in Keras.

> but possible in Torch - check soumith/dcgan.

So the dumb solution was to create two model, one updates Wd after its forward-backward pass, another updates Wg after its forward-backward pass. All those DCGAN on GitHub are almost all implemented this way.

## better solution

Implement your D and G networks in Keras, then write your parameter update code in raw Tensorflow.

I wrote a detailed description to the problem: <https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html>

## It works

![](https://ctmakro.github.io/site/on_learning/gan_cifar_32.png)
