# pip install jax[cuda12]
import jax
import jax.numpy as jnp
# dm-haiku 神经网络库
import haiku as hk
# optax 优化器 损失函数
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm


class CNN(hk.Module):
    def __call__(self, x):
        x = hk.Conv2D(output_channels=32, kernel_shape=3, stride=1, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='VALID')(x)

        x = hk.Conv2D(output_channels=64, kernel_shape=3, stride=1, padding='SAME')(x)
        x = jax.nn.relu(x)
        x = hk.MaxPool(window_shape=(2, 2), strides=(2, 2), padding='VALID')(x)

        x = hk.Flatten()(x)
        x = hk.Linear(128)(x)  # 输出的维度
        x = jax.nn.relu(x)
        x = hk.Linear(10)(x)
        return x


def CNN_forward(x):
    cnn = CNN()
    return cnn(x)


CNN_model = hk.transform(CNN_forward)


def load_dataset(split, batch_size=256):
    ds = tfds.load('mnist', split=split, as_supervised=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds = ds.map(lambda img, label: (tf.cast(img, tf.float32) / 255.0, label))
    return ds

@jax.jit
def loss_fn(params, key, x, y):
    predict = CNN_model.apply(params, key, x)
    target = jax.nn.one_hot(y, num_classes=10)
    loss = optax.softmax_cross_entropy(predict, target).mean()
    return loss

@jax.jit
def update(params, key, x, y):
    loss, grad = jax.value_and_grad(loss_fn)(params, key, x, y)
    return grad

#动态不可注释
def apply_update(optimizer, opt_state, params, grad):
    update, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, update)
    return new_params, opt_state


def run():
    batch_size = 256
    epochs = 10
    train_data = load_dataset("train", batch_size)
    test_data = load_dataset("test", batch_size)

    key = jax.random.PRNGKey(0)
    update_key, test_key = jax.random.split(key, 2)
    input_size = jnp.ones([batch_size, 28, 28, 1])  # 样本输入
    params = CNN_model.init(key, input_size)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    for epoch in range(epochs):
        for x, y in tqdm(train_data.as_numpy_iterator()):
            # x, y = jax.device_put(x), jax.device_put(y)
            grad = update(params, update_key, x, y)
            params, opt_state = apply_update(optimizer, opt_state, params, grad)
        correct, total = 0, 0
        for x_batch, y_batch in test_data.as_numpy_iterator():
            # x_batch = jax.device_put(x_batch)
            preds = CNN_model.apply(params, test_key, x_batch)
            preds = jnp.argmax(preds, axis=-1)
            correct += jnp.sum(preds==y_batch)
            total += len(y_batch)
        accuracy = correct / total
        print(f"Epoch {epoch + 1}, Accuracy:{accuracy:.4f}")


if __name__ == '__main__':
    run()
