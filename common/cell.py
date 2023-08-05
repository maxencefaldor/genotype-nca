import jax
import jax.numpy as jnp
import flax.linen as nn


@jax.jit
def to_rgba(x):
	return x[..., :4]

@jax.jit
def to_alpha(x):
	return jnp.clip(x[..., 3:4], a_min=0.0, a_max=1.0)

@jax.jit
def to_rgb(x):
	# assume rgb premultiplied by alpha
	rgb, a = x[..., :3], to_alpha(x)
	return 1.0-a+rgb

@jax.jit
def get_living_mask(x):
	alpha = x[..., 3:4]
	return nn.max_pool(alpha, window_shape=(3, 3), strides=(1, 1), padding="SAME") > 0.1

def make_circle_masks(random_key, n, h, w):
	x = jnp.linspace(-1.0, 1.0, w)[None, None, :]
	y = jnp.linspace(-1.0, 1.0, h)[None, :, None]
	random_key_1, random_key_2 = jax.random.split(random_key)
	center = jax.random.uniform(random_key_1, shape=(2, n, 1, 1), minval=-0.5, maxval=0.5)
	r = jax.random.uniform(random_key_2, shape=(n, 1, 1), minval=0.1, maxval=0.4)
	x, y = (x-center[0])/r, (y-center[1])/r
	mask = x*x+y*y < 1.0
	return mask
