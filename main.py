from common.utils import Config
import hydra
from hydra.core.config_store import ConfigStore


@hydra.main(version_base="1.2", config_path="configs/", config_name="config")
def main(config: Config) -> None:
    if config.exp.name == "emoji":
        import main_emoji as main
    elif config.exp.name == "face":
        import main_face as main
    elif config.exp.name == "vae":
        import main_vae as main
    else:
        raise NotImplementedError

    main.main(config)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
