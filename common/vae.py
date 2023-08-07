from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax


class Encoder_1(nn.Module):
    latent_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x))
        x = nn.relu(nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x))
        x = x.reshape(*x.shape[:-3], -1)
        mean = nn.Dense(features=self.latent_size)(x)
        logvar = nn.Dense(features=self.latent_size)(x)
        return mean, logvar

class Decoder_1(nn.Module):
    img_shape: Tuple[int, int, int]

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(features=16*16*32)(z))
        z = z.reshape(*z.shape[:-1], 16, 16, 32)
        z = nn.relu(nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
        z = nn.relu(nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
        z = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(z)
        return z

class VAE_1(nn.Module):
    img_shape: Tuple[int, int, int]
    latent_size: int

    def setup(self):
        self.encoder = Encoder_1(latent_size=self.latent_size)
        self.decoder = Decoder_1(img_shape=self.img_shape)

    def reparameterize(self, random_key, mean, logvar):
        eps = jax.random.normal(random_key, shape=mean.shape)
        return eps * jnp.exp(logvar * .5) + mean

    def encode(self, random_key, x):
        mean, logvar = self.encoder(x)
        return self.reparameterize(random_key, mean, logvar), mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

    def __call__(self, random_key, x):
        z, mean, logvar = self.encode(random_key, x)
        logits = self.decode(z)
        return logits, mean, logvar


class Encoder_2(nn.Module):
    latent_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x))
        x = nn.relu(nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x))
        x = nn.relu(nn.Conv(features=256, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x))
        x = x.reshape(*x.shape[:-3], -1)
        x = nn.relu(nn.Dense(features=1024)(x))
        mean = nn.Dense(features=self.latent_size)(x)
        logvar = nn.Dense(features=self.latent_size)(x)
        return mean, logvar

class Decoder_2(nn.Module):
    img_shape: Tuple[int, int, int]

    @nn.compact
    def __call__(self, z):
        z = nn.relu(nn.Dense(features=1024)(z))
        z = nn.relu(nn.Dense(features=8*8*256)(z))
        z = z.reshape(*z.shape[:-1], 8, 8, 256)
        z = nn.relu(nn.ConvTranspose(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
        z = nn.relu(nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z))
        z = nn.ConvTranspose(features=3, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(z)
        return z

class VAE_2(nn.Module):
    img_shape: Tuple[int, int, int]
    latent_size: int

    def setup(self):
        self.encoder = Encoder_2(latent_size=self.latent_size)
        self.decoder = Decoder_2(img_shape=self.img_shape)

    def reparameterize(self, random_key, mean, logvar):
        eps = jax.random.normal(random_key, shape=mean.shape)
        return eps * jnp.exp(logvar * .5) + mean

    def encode(self, random_key, x):
        mean, logvar = self.encoder(x)
        return self.reparameterize(random_key, mean, logvar), mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

    def __call__(self, random_key, x):
        z, mean, logvar = self.encode(random_key, x)
        logits = self.decode(z)
        return logits, mean, logvar


@jax.jit
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.jit
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1.0 - labels) * jnp.log(-jnp.expm1(logits)))

@jax.jit
def vae_loss(logits, targets, mean, logvar):
    bce_loss = binary_cross_entropy_with_logits(logits, targets).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    return bce_loss + kld_loss

vae_dict = {1: VAE_1, 2: VAE_2}