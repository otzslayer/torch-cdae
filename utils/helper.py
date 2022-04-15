import yaml


def load_config(config_path: str) -> dict:
    r"""A simple function to load yaml configuration file

    Parameters
    ----------
    config_path : str
        the path of yaml configuration file
    """
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config
