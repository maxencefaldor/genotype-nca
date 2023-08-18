from functools import partial

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import traverse_util

from . import cell


class NCA(nn.Module):
	cell_state_size: int
	n_perceive_free: int
	update_size: int
	fire_rate: float

	@nn.compact
	def __call__(self, random_key, x, genotype, step_size=1.0):
		pre_life_mask = cell.get_living_mask(x)

		# Perceive with depthwise convolution
		y = nn.Conv(features=3*self.cell_state_size, kernel_size=(3, 3), padding="SAME", feature_group_count=self.cell_state_size, use_bias=False, name="perceive_frozen")(x)
		if self.n_perceive_free > 0:
			y_free = nn.Conv(features=self.n_perceive_free*self.cell_state_size, kernel_size=(3, 3), padding="SAME", feature_group_count=self.cell_state_size, use_bias=False, name="perceive_free")(x)
			y = jnp.concatenate([y, y_free], axis=-1)

		# Add genotype
		genotype = jnp.repeat(genotype[..., None, :], repeats=x.shape[-3], axis=-2)
		genotype = jnp.repeat(genotype[..., None, :], repeats=x.shape[-2], axis=-2)
		y = jnp.concatenate([y, genotype], axis=-1)

		# Update
		dx = nn.relu(nn.Conv(features=self.update_size, kernel_size=(1, 1))(y))
		dx = nn.Conv(features=self.cell_state_size, kernel_size=(1, 1), kernel_init=nn.initializers.zeros)(dx) * step_size
		update_mask = jax.random.uniform(random_key, shape=(*x.shape[:-1], 1), minval=0., maxval=1.) <= self.fire_rate
		x += dx * update_mask

		post_life_mask = cell.get_living_mask(x)
		life_mask = pre_life_mask & post_life_mask
		return x * life_mask

	@partial(jax.jit, static_argnames=("self",))
	def _get_kernel(self, angle):
		identify = jnp.array([0., 1., 0.])
		identify = jnp.outer(identify, identify)
		dx = jnp.outer(jnp.array([1., 2., 1.]), jnp.array([-1., 0., 1.])) / 8.0  # Sobel filter
		dy = dx.T
		c, s = jnp.cos(angle), jnp.sin(angle)
		kernel = jnp.stack([identify, c*dx-s*dy, s*dx+c*dy], axis=-1)[:, :, None, :]
		kernel = jnp.tile(kernel, (1, 1, 1, self.cell_state_size))
		return kernel

	def set_kernel(self, params, angle=0.):
		kernel = self._get_kernel(angle)
		params["params"]["perceive_frozen"]["kernel"] = kernel
		return params

	def get_perceive_mask(self, params):
		flat_params = traverse_util.flatten_dict(params, sep="/")
		flat_params = dict.fromkeys(flat_params, False)

		for key in flat_params:
			if "perceive_frozen" in key:
				flat_params[key] = True
		return traverse_util.unflatten_dict(flat_params, sep="/")
