import os

import hydra
from loguru import logger


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg):
    """Function that shows how to use loguru with hydra."""
    # Get the path to the hydra output directory
    hydra_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Add a log file to the logger
    logger.add(os.path.join(hydra_path, "my_logger_hydra.log"))
    logger.info(cfg)

    logger.debug("Used for debugging your code.")
    logger.info("Informative messages from your code.")
    logger.warning("Everything works but there is something to be aware of.")
    logger.error("There's been a mistake with the process.")
    logger.critical("There is something terribly wrong and process may terminate.")


if __name__ == "__main__":
    main()
