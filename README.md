# DCGAN-Keras
The CORRECT and PERFORMANT way to implement GAN in Keras.

This is the Tensorflow version, here's another guy who wrote the theano version: https://github.com/bstriner/keras-adversarial

## How this happened
DCGAN have been implemented in a lot of frameworks. However, existing Keras and Tensorflow implementations are SLOW due to duplicated computation.

Basically we want to do two things in one forward-backward pass:

1. update Wd w.r.t. D_loss
2. update Wg w.r.t. G_loss

This kind of update(different parameters w.r.t. different loss) however is not possible in Keras.

> but possible in Torch - check soumith/dcgan.

So the dumb solution was to create two model, one updates Wd after its forward-backward pass, another updates Wg after its forward-backward pass. All those DCGAN on GitHub are almost all implemented this way.

## better solution

1. Create D and G network in Keras, as usual;

2. Write your parameter update operations by hand:

    ```py
    # noise: the input z
    noise = Input(shape=(zed,))
    # real_image input
    real_image = Input(shape=(32,32,3))

    # dm and gm are your generative and discriminative network.
    generated = gm(noise)

    # dm should produce a score between (0,1) remember?
    gscore = dm(generated)
    rscore = dm(real_image)

    def log_eps(i):
        return K.log(i+1e-11)

    # calculate the losses

    # single side label smoothing: replace 1.0 with 0.9 for real input
    dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
    gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-4,.2 # otherwise won't converge.
    optimizer = Adam(lr,beta1=b1)

    # update Wd w.r.t. D_loss
    grad_loss_wd = optimizer.compute_gradients(dloss, dm.trainable_weights)
    update_wd = optimizer.apply_gradients(grad_loss_wd)

    # update Wg w.r.t G_loss
    grad_loss_wg = optimizer.compute_gradients(gloss, gm.trainable_weights)
    update_wg = optimizer.apply_gradients(grad_loss_wg)

    ```

3. instead of using `model.fit()`, run the parameter update by hand:

    ```py
    train_step = [update_wd, update_wg, other_parameter_updates]
    sess.run(train_step,feed_dict={noise,real_image.......})
    ```

I wrote a detailed description to the problem: <https://ctmakro.github.io/site/on_learning/fast_gan_in_keras.html>

## It works

![](https://ctmakro.github.io/site/on_learning/gan_cifar_32.png)
