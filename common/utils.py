import io
from dataclasses import dataclass

import pickle
import requests
import numpy as np
from flax import serialization
import PIL.Image


@dataclass
class Config:
    seed: int


def load_image(url, size):
	r = requests.get(url)
	img = PIL.Image.open(io.BytesIO(r.content))
	img.thumbnail((size, size), PIL.Image.Resampling.LANCZOS)
	img = np.float32(img)/255.0
	# premultiply RGB by Alpha
	img[..., :3] *= img[..., 3:]
	return img


def load_emoji(emoji, size):
	code = hex(ord(emoji))[2:].lower()
	url = "https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true"%code
	return load_image(url, size)


def visualize(items_before, items_after, targets, step):
    items_before = np.hstack(np.array(to_rgb(items_before)))
    items_after = np.hstack(np.array(to_rgb(items_after)))
    targets = np.hstack(np.array(to_rgb(targets)))
    img = np.vstack([items_before, items_after,targets])

    # Save
    img = PIL.Image.fromarray(np.array(img * 255, dtype=np.uint8))
    img.save("batch_%04d.png"%step)


def export_model(params, step):
    state_dict = serialization.to_state_dict(params)
    with open("model_%04d.pickle"%step, "wb") as params_file:
        pickle.dump(state_dict, params_file)
