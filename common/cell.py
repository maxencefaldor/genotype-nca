from functools import partial

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
	alpha = to_alpha(x)
	return nn.max_pool(alpha, window_shape=(3, 3), strides=(1, 1), padding="SAME") > 0.1

@partial(jax.jit, static_argnames=("height", "width",))
def make_ellipse_mask(center, height, width, r1, r2):
    x = 0.5 + jnp.linspace(0, width-1, width)[None, :]
    y = 0.5 + jnp.linspace(0, height-1, height)[:, None]
    x, y = (x-center[0])/r1, (y-center[1])/r2
    mask = x*x + y*y < 1.0
    return mask

@partial(jax.jit, static_argnames=("height", "width",))
def make_circle_masks(random_key, height, width):
	random_key_1, random_key_2 = jax.random.split(random_key)
	center = (1 + jax.random.uniform(random_key_1, shape=(2,), minval=-0.5, maxval=0.5)) * width/2
	r = jax.random.uniform(random_key_2, shape=(), minval=0.1, maxval=0.4) * width/2
	return make_ellipse_mask(center, height, width, r, r)
