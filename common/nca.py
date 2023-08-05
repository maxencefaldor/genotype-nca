from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import traverse_util

from . import cell


class NCA(nn.Module):
	cell_state_size: int
	fire_rate: float

	@nn.compact
	def __call__(self, random_key, x, step_size=1.0):
		pre_life_mask = cell.get_living_mask(x)

		# Perceive with depthwise convolution
		y = nn.Conv(features=3*x.shape[-1], kernel_size=(3, 3), padding="SAME", feature_group_count=x.shape[-1], use_bias=False, name="perceive")(x)

		# Update
		dx = nn.relu(nn.Conv(features=128, kernel_size=(1, 1))(y))
		dx = nn.Conv(features=self.cell_state_size, kernel_size=(1, 1), kernel_init=nn.initializers.zeros)(dx) * step_size
		update_mask = jax.random.uniform(random_key, shape=(*x.shape[:-1], 1), minval=0., maxval=1.) <= self.fire_rate
		x += dx * update_mask

		post_life_mask = cell.get_living_mask(x)
		life_mask = pre_life_mask & post_life_mask
		return x * life_mask

	@partial(jax.jit, static_argnames=("self",))
	def _get_kernel(self, angle=0.):
		identify = jnp.array([0., 1., 0.])
		identify = jnp.outer(identify, identify)
		dx = jnp.outer(jnp.array([1., 2., 1.]), jnp.array([-1., 0., 1.])) / 8.0  # Sobel filter
		dy = dx.T
		c, s = jnp.cos(angle), jnp.sin(angle)
		kernel = jnp.stack([identify, c*dx-s*dy, s*dx+c*dy], axis=-1)[:, :, None, :]
		kernel = jnp.tile(kernel, (1, 1, 1, 16))
		return kernel

	def set_kernel(self, params, angle=0.):
		kernel = self._get_kernel(angle)
		params["params"]["perceive"]["kernel"] = kernel
		return params

	def get_perceive_mask(self, params):
		flat_params = traverse_util.flatten_dict(params, sep="/")
		flat_params = dict.fromkeys(flat_params, False)

		for key in flat_params:
			if "perceive" in key:
				flat_params[key] = True
		return traverse_util.unflatten_dict(flat_params, sep="/")
