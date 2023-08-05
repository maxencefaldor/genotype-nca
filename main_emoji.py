import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax

from common.cell import to_rgba, to_alpha, to_rgb, get_living_mask
from common.nca import NCA
from common.utils import Config, load_emoji

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb


@hydra.main(version_base="1.2", config_path="configs/", config_name="emoji")
def main(config: Config) -> None:
    wandb.init(
        project="genotype-nca",
        name=config.exp.name,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Experiment
    use_pattern_pool = {"Growing": 0, "Persistent": 1, "Regenerating": 1}[config.exp.experiment_type]
    damage_n = {"Growing": 0, "Persistent": 0, "Regenerating": 3}[config.exp.experiment_type]

    # Init the target
    target = load_emoji(config.exp.target_emoji, config.exp.target_size)
    random_key = jax.random.PRNGKey(42)
    p = config.exp.target_padding
    pad_target = jnp.pad(target, ((p, p), (p, p), (0, 0)))
    print(pad_target.shape)

    def init_cell_states(target):
        init_item = jnp.zeros((*target.shape[:2], config.exp.channel_size))
        return init_item.at[target.shape[0]//2, target.shape[1]//2, 3:].set(1.0)

    def loss_f(item, target):
        return jnp.mean(jnp.square(to_rgba(item) - target), axis=(-1, -2, -3))

    nca = NCA(channel_size=config.exp.channel_size, fire_rate=config.exp.fire_rate)
    random_key, random_subkey_1, random_subkey_2 = jax.random.split(random_key, 3)
    params = nca.init(random_subkey_1, random_subkey_2, init_cell_states(pad_target))
    params = nca.set_kernel(params)

    loss_log = []

    lr_sched = optax.linear_schedule(init_value=config.exp.learning_rate, end_value=0.1*config.exp.learning_rate, transition_steps=2000)

    # zero update
    def zero_grads():
        def init_fn(_):
            return ()

        def update_fn(updates, state, params=None):
            return jax.jax.tree_util.tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)

    tx = optax.multi_transform({False: optax.adam(learning_rate=lr_sched),
                                True: zero_grads()}, nca.get_perceive_mask(params))

    train_state = TrainState.create(
        apply_fn=nca.apply,
        params=params,
        tx=tx)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    main()
