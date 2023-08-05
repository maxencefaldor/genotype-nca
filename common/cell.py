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
