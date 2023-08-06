from functools import partial

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
import pandas as pd

from common.cell import to_rgba, make_circle_masks
from common.vae import VAE
from common.utils import Config, load_face, visualize, plot_loss, export_model

import tqdm
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
	df_attr_celeba = pd.read_csv(config.exp.attr, sep="\s+", skiprows=1)
	df_attr_celeba.replace(to_replace=-1, value=0, inplace=True) # replace -1 by 0
	attr = jnp.array(df_attr_celeba)

	# Load list_landmarks_align_celeba.txt file into a pandas DataFrame
	df_landmarks_align_celeba = pd.read_csv(config.exp.landmarks, sep="\s+", skiprows=1)

	# Crop images from (218, 178) to (178, 178)
	df_landmarks_align_celeba["lefteye_y"] = df_landmarks_align_celeba["lefteye_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["righteye_y"] = df_landmarks_align_celeba["righteye_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["nose_y"] = df_landmarks_align_celeba["nose_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["leftmouth_y"] = df_landmarks_align_celeba["leftmouth_y"] - (218 - 178) / 2
	df_landmarks_align_celeba["rightmouth_y"] = df_landmarks_align_celeba["rightmouth_y"] - (218 - 178) / 2

	# Resize images from (178, 178) to face_shape
	df_landmarks_align_celeba /= 178/config.exp.face_shape[0]

	landmarks = jnp.array(df_landmarks_align_celeba)
	landmarks = jax.nn.standardize(landmarks, axis=0)

	# Dataset
	faces = []
	for index, row in tqdm.tqdm(df_attr_celeba.iterrows(), total=df_attr_celeba.shape[0]):
		faces.append(load_face(config.dataset_dir + index))
	faces = jnp.stack(faces)

	# VAE
	vae = VAE(img_shape=config.exp.img_shape, latent_size=config.exp.latent_size)
	random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
	params = vae.init(random_subkey_1, faces[0], random_subkey_2)

	# Train state
	lr_sched = optax.linear_schedule(init_value=config.exp.learning_rate, end_value=0.1*config.exp.learning_rate, transition_steps=2000)
	tx = optax.adam(learning_rate=lr_sched)

	train_state = TrainState.create(
		apply_fn=vae.apply,
		params=params,
		tx=tx)

if __name__ == "__main__":
	cs = ConfigStore.instance()
	cs.store(name="config", node=Config)
	main()
