from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class Pool:
	init_cell_states: Callable
	items: jnp.ndarray
	targets: jnp.ndarray

	@classmethod
	def init(cls, init_cell_states: Callable[[jnp.ndarray], jnp.ndarray], targets: jnp.ndarray):
		init_items = init_cell_states(targets)
		return cls(init_cell_states=init_cell_states, items=init_items, targets=targets)

	@partial(jax.jit, static_argnames=("sample_size",))
	def sample(self, random_key, sample_size):
		idx = jax.random.choice(random_key, self.targets.shape[0], shape=(sample_size,), replace=False)
		batch = jnp.take(self.items, idx, axis=0)
		return idx, batch

	@jax.jit
	def commit(self, idx, batch):
		data = self.data.at[idx].set(batch)
		return self.replace(data=data)
