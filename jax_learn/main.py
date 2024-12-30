import jax
import jax.numpy as jnp

# numpy
x = jnp.array([1.0, 20.0])
print(x.mean())
print(jnp.mean(x))


# autograd
# 求导
def func(x):
    return x ** 2 + 3


grad_func = jax.grad(func)
print(grad_func(2.0))


# 求偏导
def g(x, y):
    return x * y + +jnp.sin(x)


grad_g = jax.grad(g, argnums=0)  # dg/dx
print(grad_g(1.0, 2.0))


# 常用 jax中不可忽略求导
# jax.value_and_grad
# loss,(arg1_grad,arg2_grad) = jax.value_and_grad(f,arunums=(0,1))(arg1,arg2)


# 并行计算
# vmap
# 向量化映射 替代for循环
def f(x):
    return x ** 2


vmp_f = jax.vmap(f)  # 默认第0维进行向量化处理
x = jnp.array([1.2, 2.2])
print(vmp_f(x))


def g(x, y):
    return x + y + 5


vmp_g = jax.vmap(g, in_axes=(0, None))  # 0向量化
x = jnp.array([1.2, 2.2])
y = 5.0
print(vmp_g(x, y))

# keys 1000样本 obs(256,64)(batch_size,state) ->act:(256,8)
# acts:(256,1000,8) out_axes=1
# action = jax.vmap(agent.get_action,in_axes=(0,None,None),out_axes=1)(keys,policy_params,obs)

# pmap 多gpu计算
"""
def f(x):
    return x ** 2
parallel_f = jax.pmap(f)
x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
# 设备控制 不支持windows
gpu_device = jax.devices('gpu')[0]
x = jax.arrary([1.0, 2.0, 3.0])
gpu_x = jax.devce_put(x, device=gpu_device)
cpu_x = jax.device_get(gpu_x)

with jax.default_device(gpu_device):
    pass
"""

# 复杂if,for
"""
def body_fn(carry, input):
    x = carry
    t, noise, = input
    noise_pred = jax.vmap(lambda x: model(t, x))(x)
    model_mean, model_std = p_mean_std(t, x, noise_pred)
    carry = model_mean + (t > 0) * model_std * noise
    return carry, None
x, _ = jax.lax.scan(body_fn, x, (t, noise))
"""


def f(carry, x):
    return carry + x, carry + x


carry, result = jax.lax.scan(f, 0, jnp.array([1, 2, 3, 4]))
print(carry, result)

#if
def p(x):
    return jax.lax.cond(x > 0, lambda x: x ** 2 + 1, lambda x: x ** 2, x)
print(p(-2))

# 随机性控制
key = jax.random.PRNGKey(0)
action_key,q_key = jax.random.split(key,2)

# 外部函数引用
# entropy = jax.pure_callback(func,jax.ShapeDtypeStruct((,jnp.float32),actions))

# 加速
# jax.jit
"""
@jax.jit
def func():
    pass
func_jit = jax.jit(func)
"""