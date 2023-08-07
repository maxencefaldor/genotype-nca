from functools import partial
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import pandas as pd

from common.cell import to_rgba, make_circle_masks
from common.pool import Pool
from common.nca import NCA
from common.vae import vae_dict, vae_loss
from common.utils import Config, load_face, visualize_nca, plot_loss, export_model

import tqdm
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb


@hydra.main(version_base="1.2", config_path="configs/", config_name="face")
def main(config: Config) -> None:
	wandb.init(
		project="genotype-nca",
		name=config.exp.name,
		config=OmegaConf.to_container(config, resolve=True),
	)

	# Init a random key
	random_key = jax.random.PRNGKey(config.seed)

	# Load VAE
	vae_dir = Path(config.exp.vae_dir)
	vae_config = OmegaConf.load(vae_dir / ".hydra" / "config.yaml")

	# Load list_attr_celeba.txt file into a pandas DataFrame
	df_attr_celeba = pd.read_csv(vae_config.exp.attr_dir, sep="\s+", skiprows=1)
	df_attr_celeba.replace(to_replace=-1, value=0, inplace=True) # replace -1 by 0

	# Load list_landmarks_align_celeba.txt file into a pandas DataFrame
	df_landmarks_align_celeba = pd.read_csv(vae_config.exp.landmarks_dir, sep="\s+", skiprows=1)

	# Crop images from (218, 178) to (178, 178)
	df_landmarks_align_celeba["lefteye_y"] = df_landmarks_align_celeba["lefteye_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["righteye_y"] = df_landmarks_align_celeba["righteye_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["nose_y"] = df_landmarks_align_celeba["nose_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["leftmouth_y"] = df_landmarks_align_celeba["leftmouth_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["rightmouth_y"] = df_landmarks_align_celeba["rightmouth_y"] - (218 - 178) / 2

	# Resize images from (178, 178) to face_shape
	df_landmarks_align_celeba /= 178/vae_config.exp.face_shape[0]

	# Dataset
	if vae_config.exp.grayscale:
		dataset_faces = np.zeros((df_attr_celeba.shape[0], *vae_config.exp.face_shape, 1))
	else:
		dataset_faces = np.zeros((df_attr_celeba.shape[0], *vae_config.exp.face_shape, 3))
	for i, (index, _,) in tqdm.tqdm(enumerate(df_attr_celeba.iterrows()), total=df_attr_celeba.shape[0]):
		dataset_faces[i] = load_face(vae_config.exp.dataset_dir + index, vae_config.exp.face_shape, vae_config.exp.grayscale)
		# if i == 1000:
		# 	break
	height, width = vae_config.exp.face_shape[:2]

	# VAE
	vae = vae_dict[vae_config.exp.vae_index](img_shape=dataset_faces[0].shape, latent_size=vae_config.exp.latent_size)
	random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
	params = vae.init(random_subkey_1, dataset_faces[0], random_subkey_2)
	param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
	print("Number of parameters: ", param_count)

	# Cell states
	if vae_config.exp.grayscale:
		cell_state_size = 1 + 1 + config.exp.hidden_size + vae_config.exp.latent_size
	else:
		cell_state_size = 3 + 1 + config.exp.hidden_size + vae_config.exp.latent_size

	@jax.jit
	def phenotype_target_idx_to_genotype(phenotype_target_idx):
		return jax.nn.one_hot(phenotype_target_idx, num_classes=config.exp.genotype_size)

	@jax.jit
	def init_cell_state(genotype):
		cell_state = jnp.zeros((config.exp.phenotype_size+1+config.exp.hidden_size,))  # init cell_state
		cell_state = cell_state.at[config.exp.phenotype_size:].set(1.0)  # set alpha and hidden channels to 1.0
		return jnp.concatenate([cell_state, genotype])  # add genotype to cell_state

	@jax.jit
	def init_cells_state(genotype):
		cell_state = init_cell_state(genotype)
		cells_state = jnp.zeros((height, width, cell_state_size,))
		return cells_state.at[height//2, width//2].set(cell_state)

	# Trainset
	trainset_phenotypes_target = dataset_phenotypes_target[:config.exp.genotype_size]
	trainset_genotypes_target = jax.vmap(phenotype_target_idx_to_genotype)(jnp.arange(config.exp.genotype_size))

	# Pool
	phenotypes_target_idx_init = jax.random.choice(random_key, trainset_phenotypes_target.shape[0], shape=(config.exp.pool_size,), replace=True)
	cells_states_init = jax.vmap(init_cells_state)(jnp.take(trainset_genotypes_target, phenotypes_target_idx_init, axis=0))
	pool = Pool(cells_states=cells_states_init, phenotypes_target_idx=phenotypes_target_idx_init)

	# NCA
	nca = NCA(cell_state_size=cell_state_size, fire_rate=config.exp.fire_rate)
	random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
	params = nca.init(random_subkey_1, random_subkey_2, cells_states_init[0])
	params = nca.set_kernel(params)

	# Train state
	lr_sched = optax.linear_schedule(init_value=config.exp.learning_rate, end_value=0.1*config.exp.learning_rate, transition_steps=2000)

	def zero_grads():
		def init_fn(_):
			return ()

		def update_fn(updates, state, params=None):
			return jax.jax.tree_util.tree_map(jnp.zeros_like, updates), ()
		return optax.GradientTransformation(init_fn, update_fn)

	optimizer = optax.chain(
		optax.clip_by_global_norm(1.0),
		optax.adam(learning_rate=lr_sched),)
	tx = optax.multi_transform({False: optimizer, True: zero_grads()},
							   nca.get_perceive_mask(params))

	train_state = TrainState.create(
		apply_fn=nca.apply,
		params=params,
		tx=tx)

	# Train
	@jax.jit
	def loss_f(cell_states, phenotype):
		return jnp.mean(jnp.square(to_rgba(cell_states) - phenotype), axis=(-1, -2, -3))

	loss_log = []

	@jax.jit
	def scan_apply(carry, random_key):
		(params, cells_states,) = carry
		cells_states_ = train_state.apply_fn(params, random_key, cells_states)
		return (params, cells_states_), ()

	@partial(jax.jit, static_argnames=("n_iterations",))
	def train_step(random_key, train_state, cells_states, phenotypes_target, n_iterations):
		def loss_fn(params):
			random_keys = jax.random.split(random_key, n_iterations)
			(params, cells_states_), _ = jax.lax.scan(scan_apply, (params, cells_states,), random_keys, length=n_iterations)
			return loss_f(cells_states_, phenotypes_target).mean(), cells_states_

		(loss, cells_states_), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
		train_state = train_state.apply_gradients(grads=grads)

		return train_state, loss, cells_states_

	for i in range(1, config.exp.n_iterations+1):
		random_key, random_subkey_1, random_subkey_2, random_subkey_3, random_subkey_4, random_subkey_5, random_subkey_6 = jax.random.split(random_key, 7)

		if use_pattern_pool:
			# Sample cells' states from pool
			idx, cells_states, phenotypes_target_idx = pool.sample(random_subkey_1, config.exp.batch_size)

			# Rank by loss
			loss_rank = jnp.flip(jnp.argsort(loss_f(cells_states, jnp.take(trainset_phenotypes_target, phenotypes_target_idx, axis=0))))
			idx = jnp.take(idx, loss_rank, axis=0)
			cells_states = jnp.take(cells_states, loss_rank, axis=0)
			phenotypes_target_idx = jnp.take(phenotypes_target_idx, loss_rank, axis=0)

			# Sample new phenotype target
			new_phenotype_target_idx = jax.random.randint(random_subkey_2, shape=(), minval=0, maxval=trainset_phenotypes_target.shape[0])
			new_cells_state = init_cells_state(jnp.take(trainset_genotypes_target, new_phenotype_target_idx, axis=0))
			cells_states = cells_states.at[0].set(new_cells_state)
			phenotypes_target_idx_ = phenotypes_target_idx.at[0].set(new_phenotype_target_idx)

			if n_damages:
				damage = 1.0 - make_circle_masks(random_subkey_3, n_damages, height, width)[..., None]
				cells_states = cells_states.at[-n_damages:].set(cells_states[-n_damages:] * damage)
		else:
			genotypes = jax.random.choice(random_subkey_4, trainset_genotypes_target, shape=(config.exp.batch_size,), replace=True)
			cells_states = jax.vmap(init_cells_state)(genotypes)

		n_iterations = jax.random.randint(random_subkey_5, shape=(), minval=64, maxval=96)
		phenotypes_target_ = jnp.take(trainset_phenotypes_target, phenotypes_target_idx_, axis=0)
		train_state, loss, cells_states_ = train_step(random_subkey_6, train_state, cells_states, phenotypes_target_, int(n_iterations))

		if use_pattern_pool:
			pool = pool.commit(idx, cells_states_, phenotypes_target_idx_)

		loss_log.append(loss)
		print("\r step: %d, log10(loss): %.3f"%(i, jnp.log10(loss)), end="")

		if i % config.exp.log_period == 0:
			visualize_nca(cells_states[-16:], cells_states_[-16:], phenotypes_target_[-16:], i)
			plot_loss(loss_log)
	
	export_model(train_state.params, "nca.pickle")


if __name__ == "__main__":
	cs = ConfigStore.instance()
	cs.store(name="config", node=Config)
	main()
