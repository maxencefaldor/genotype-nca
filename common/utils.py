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


def load_image(url, size):
	r = requests.get(url)
	image = PIL.Image.open(io.BytesIO(r.content))
	image.thumbnail((size, size), resample=PIL.Image.Resampling.LANCZOS)
	image = jnp.array(image, dtype=jnp.float32)/255.0
	# premultiply RGB by Alpha
	return image.at[..., :3].multiply(cell.to_alpha(image))


def load_emoji(emoji, emoji_size, emoji_padding):
	code = hex(ord(emoji))[2:].lower()
	url = "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"%code
	image = load_image(url, emoji_size)
	return jnp.pad(image, ((emoji_padding, emoji_padding), (emoji_padding, emoji_padding), (0, 0)))


def load_face(dir, face_shape) -> jnp.ndarray:
	if face_shape[-1] == 1:
		image = PIL.Image.open(dir).convert("L")
		image = jnp.expand_dims(jnp.array(image, dtype=np.float32), axis=-1) / 255.
	elif face_shape[-1] == 3:
		image = PIL.Image.open(dir)
		image = jnp.array(image, dtype=np.float32) / 255.
	else:
		raise ValueError("Face must be 1 or 3 channels.")
	image = pix.resize_with_crop_or_pad(image, 178, 178)
	image = jax.image.resize(image, face_shape, method="linear")
	return image


def jnp2pil(a):
	return PIL.Image.fromarray(np.array(jnp.clip(a, a_min=0., a_max=1.) * 255, dtype=np.uint8))


def visualize(cells_states_before, cells_states_after, phenotypes_target, i):
	cells_states_before = jnp.hstack(cell.to_rgb(cells_states_before))
	cells_states_after = jnp.hstack(cell.to_rgb(cells_states_after))
	phenotypes_target = jnp.hstack(cell.to_rgb(phenotypes_target))
	img = jnp.vstack([cells_states_before, cells_states_after, phenotypes_target])

	# Save
	img = jnp2pil(img)
	img.save("batch_%04d.png"%i)


def plot_loss(loss_log):
	plt.figure(figsize=(10, 4))
	plt.title("Loss (log10)")
	plt.plot(jnp.log10(jnp.array(loss_log)), ".", alpha=0.1)
	plt.savefig("loss.png")
	plt.close()


def export_model(params, i):
	state_dict = serialization.to_state_dict(params)
	with open("model_%04d.pickle"%i, "wb") as params_file:
		pickle.dump(state_dict, params_file)
