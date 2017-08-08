# DCGAN-Keras
The CORRECT and PERFORMANT way to implement GAN in Keras.

This is the Tensorflow version, here's another guy who wrote the theano version: https://github.com/bstriner/keras-adversarial

# In short

1. create your D and G network as usual in Keras

2. call `gan_feed = gan(G,D)`

3. feed your data manually:
    ```py
    # sample from cifar
    j = i % int(length/batch_size)
    minibatch = shuffled_cifar[j*batch_size:(j+1)*batch_size]
    z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))

    # train for one step
    losses = gan_feed(sess,minibatch,z_input)
    ```

## How this happened
DCGAN have been implemented in a lot of frameworks. However, existing Keras and Tensorflow implementations are SLOW due to duplicated computation.

Basically we want to do two things in one forward-backward pass:

1. update Wd w.r.t. D_loss
2. update Wg w.r.t. G_loss

This kind of update(different parameters w.r.t. different loss) however is not possible in Keras.

> but possible in Torch - check soumith/dcgan.

So the dumb solution was to create two model, one updates Wd after its forward-backward pass, another updates Wg after its forward-backward pass. All those DCGAN on GitHub are almost all implemented this way.

I wrote a detailed description to the problem: <https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html>

## It works

![](https://ctmakro.github.io/site/on_learning/gan_cifar_32.png)

## License

someone suggest I put a license here.

PUBLIC DOMAIN

USE THIS CODE HOWEVER WHATEVER WHEREVER WHENEVER.
