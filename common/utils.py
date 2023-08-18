import io
from dataclasses import dataclass

import pickle
import requests
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import dm_pix as pix
import matplotlib.pyplot as plt
import PIL.Image

from . import cell


@dataclass
class Config:
	seed: int


def load_emoji_from_url(url, size):
	r = requests.get(url)
	image = PIL.Image.open(io.BytesIO(r.content))
	image.thumbnail((size, size), resample=PIL.Image.Resampling.LANCZOS)
	image = jnp.array(image, dtype=jnp.float32)/255.0
	# premultiply RGB by Alpha
	return image.at[..., :3].multiply(cell.to_alpha(image))


def load_emoji(emoji, emoji_size, emoji_padding):
	code = hex(ord(emoji))[2:].lower()
	url = "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"%code
	image = load_emoji_from_url(url, emoji_size)
	return jnp.pad(image, ((emoji_padding, emoji_padding), (emoji_padding, emoji_padding), (0, 0)))


def load_face(dir, face_shape, grayscale: bool) -> jnp.ndarray:
	if grayscale:
		image = PIL.Image.open(dir).convert("L")
		image = jnp.expand_dims(jnp.array(image, dtype=np.float32), axis=-1) / 255.
		image = pix.resize_with_crop_or_pad(image, 178, 178)
		image = jnp.clip(jax.image.resize(image, (*face_shape, 1), method="linear"), a_min=0., a_max=1.)
	else:
		image = PIL.Image.open(dir)
		image = jnp.array(image, dtype=np.float32) / 255.
		image = pix.resize_with_crop_or_pad(image, 178, 178)
		image = jnp.clip(jax.image.resize(image, (*face_shape, 3), method="linear"), a_min=0., a_max=1.)
	return image


def jnp2pil(a):
	return PIL.Image.fromarray(np.array(jnp.clip(a, a_min=0., a_max=1.) * 255, dtype=np.uint8))


def visualize_nca(cells_states_before, cells_states_after, phenotypes_target, filename=None):
	cells_states_before = jnp.hstack(cell.to_rgb(cells_states_before))
	cells_states_after = jnp.hstack(cell.to_rgb(cells_states_after))
	phenotypes_target = jnp.hstack(cell.to_rgb(phenotypes_target))
	img = jnp.vstack([cells_states_before, cells_states_after, phenotypes_target])
	img = jnp2pil(img)

	if filename is not None:
		img.save(filename)
	return img


def visualize_vae(face_recon, face_target, i):
	face_recon = jnp.hstack(face_recon)
	face_target = jnp.hstack(face_target)
	image = jnp.vstack([face_recon, face_target])

	# Save
	image = jnp2pil(image)
	image.save("batch_%04d.png"%i)


def plot_loss(loss_log, loss_log_test=None):
	plt.figure(figsize=(10, 4))
	plt.title("Loss (log10)")
	plt.plot(jnp.log10(jnp.array(loss_log)), ".", alpha=0.1)
	if loss_log_test:
		plt.plot(jnp.log10(jnp.array(loss_log_test)), ".")
	plt.savefig("loss.png")
	plt.close()


def save_params(params, filename):
	state_dict = serialization.to_state_dict(params)
	with open(filename, "wb") as params_file:
		pickle.dump(state_dict, params_file)


def load_params(params, filename):
	with open(filename, "rb") as params_file:
		state_dict = pickle.load(params_file)
	return serialization.from_state_dict(params, state_dict)
