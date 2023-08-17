from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
import pandas as pd

from common.vae import vae_dict, vae_loss
from common.utils import Config, load_face, visualize_vae, plot_loss, save_params

from tqdm import tqdm
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb


@hydra.main(version_base="1.2", config_path="configs/", config_name="vae")
def main(config: Config) -> None:
	wandb.init(
		project="genotype-nca",
		name=config.exp.name,
		config=OmegaConf.to_container(config, resolve=True),
	)

	# Init a random key
	random_key = jax.random.PRNGKey(config.seed)

	# Load list_attr_celeba.txt file into a pandas DataFrame
	df_attr_celeba = pd.read_csv(config.exp.attr_dir, sep="\s+", skiprows=1)
	df_attr_celeba.replace(to_replace=-1, value=0, inplace=True) # replace -1 by 0

	# Load list_landmarks_align_celeba.txt file into a pandas DataFrame
	df_landmarks_align_celeba = pd.read_csv(config.exp.landmarks_dir, sep="\s+", skiprows=1)

	# Crop images from (218, 178) to (178, 178)
	df_landmarks_align_celeba["lefteye_y"] = df_landmarks_align_celeba["lefteye_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["righteye_y"] = df_landmarks_align_celeba["righteye_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["nose_y"] = df_landmarks_align_celeba["nose_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["leftmouth_y"] = df_landmarks_align_celeba["leftmouth_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["rightmouth_y"] = df_landmarks_align_celeba["rightmouth_y"] - (218 - 178) / 2

	# Resize images from (178, 178) to face_shape
	df_landmarks_align_celeba /= 178/config.exp.face_shape[0]

	# Dataset
	if config.exp.grayscale:
		dataset_faces = np.zeros((df_attr_celeba.shape[0], *config.exp.face_shape, 1))
	else:
		dataset_faces = np.zeros((df_attr_celeba.shape[0], *config.exp.face_shape, 3))
	for i, (index, _,) in tqdm.tqdm(enumerate(df_attr_celeba.iterrows()), total=df_attr_celeba.shape[0]):
		dataset_faces[i] = load_face(config.exp.dataset_dir + index, config.exp.face_shape, config.exp.grayscale)
		# if i == 1000:
		# 	break

	# Trainset - Testset
	trainset_faces = dataset_faces[:int(0.9 * len(dataset_faces))]
	testset_faces = dataset_faces[int(0.9 * len(dataset_faces)):]

	# VAE
	vae = vae_dict[config.exp.vae_index](img_shape=dataset_faces[0].shape, latent_size=config.exp.latent_size)
	random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
	params = vae.init(random_subkey_1, random_subkey_2, dataset_faces[0])
	param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
	print("Number of parameters: ", param_count)

	# Train state
	lr_sched = optax.linear_schedule(init_value=config.exp.learning_rate, end_value=0.1*config.exp.learning_rate, transition_steps=config.exp.n_iterations)
	tx = optax.adam(learning_rate=lr_sched)

	train_state = TrainState.create(
		apply_fn=vae.apply,
		params=params,
		tx=tx)

	@jax.jit
	def get_batch(random_key, dataset):
		return jax.random.choice(random_key, dataset, shape=(config.exp.batch_size,))

	@jax.jit
	def train_step(random_key, train_state: TrainState, batch: jnp.ndarray):

		def loss_fn(params):
			logits, mean, logvar = train_state.apply_fn(params, random_key, batch)
			return vae_loss(logits, batch, mean, logvar)

		loss, grads = jax.value_and_grad(loss_fn)(train_state.params)
		train_state = train_state.apply_gradients(grads=grads)

		return train_state, loss

	@jax.jit
	def scan_train_step(carry, x):
		train_state = carry
		random_key, batch = x
		train_state, loss = train_step(random_key, train_state, batch)
		return train_state, loss

	loss_log = []
	loss_log_test = []
	for i in tqdm(range(1, config.exp.n_iterations//config.exp.log_period+1), total=config.exp.n_iterations//config.exp.log_period):
		# Train
		random_keys = jax.random.split(random_key, 1+2*config.exp.log_period)
		random_key, random_keys_batch, random_keys_scan = random_keys[-1], random_keys[:config.exp.log_period], random_keys[config.exp.log_period:-1]
		batch = jax.vmap(get_batch, in_axes=(0, None))(random_keys_batch, trainset_faces)
		train_state, loss = jax.lax.scan(
			scan_train_step,
			train_state,
			(random_keys_scan, batch,),
			length=config.exp.log_period,)

		loss_log += loss.tolist()
		print("\r step: {:d}, log10(loss): {:.3f}".format(i*config.exp.log_period, jnp.log10(jnp.mean(loss))), end="")

		# Test
		random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
		batch = get_batch(random_subkey_1, testset_faces)
		logits, mean, logvar = train_state.apply_fn(random_subkey_2, train_state.params, batch)
		loss_test = vae_loss(logits, batch, mean, logvar)
		loss_log_test.append(loss_test)

		visualize_vae(nn.sigmoid(logits)[-16:], batch[-16:], "batch_{:07d}.png".format(i))
		save_params(train_state.params, "vae_{:07d}.pickle".format(i))
		plot_loss(loss_log, loss_log_test)


if __name__ == "__main__":
	cs = ConfigStore.instance()
	cs.store(name="config", node=Config)
	main()
