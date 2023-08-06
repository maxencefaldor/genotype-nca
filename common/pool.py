from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Pool:
	cells_states: jnp.ndarray
	phenotypes_target_idx: jnp.ndarray

	@partial(jax.jit, static_argnames=("sample_size",))
	def sample(self, random_key, sample_size):
		idx = jax.random.choice(random_key, self.phenotypes_target_idx.shape[0], shape=(sample_size,), replace=False)
		cells_states = jnp.take(self.cells_states, idx, axis=0)
		phenotypes_target_idx = jnp.take(self.phenotypes_target_idx, idx, axis=0)
		return idx, cells_states, phenotypes_target_idx

	@jax.jit
	def commit(self, idx, cells_states, phenotypes_target_idx):
		return self.replace(
			cells_states=self.cells_states.at[idx].set(cells_states),
			phenotypes_target_idx=self.phenotypes_target_idx.at[idx].set(phenotypes_target_idx),)
