import sys

import torch
from model import Decoder, Encoder, Model  # noqa: F401
from omegaconf import OmegaConf

if __name__ == "__main__":
    print(sys.argv)

    exp1 = sys.argv[1]
    exp2 = sys.argv[2]

    print(f"Comparing run {exp1} to {exp2}")

    model1 = torch.load(f"{exp1}/trained_model.pt", weights_only=False)
    model2 = torch.load(f"{exp2}/trained_model.pt", weights_only=False)

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.allclose(p1, p2):
            msg = "encountered a difference in parameters, your script is not fully reproducible"
            raise RuntimeError(msg)

    config1 = OmegaConf.load(f"{exp1}/.hydra/config.yaml")
    config2 = OmegaConf.load(f"{exp2}/.hydra/config.yaml")
    if config1 != config2:
        msg = "encountered a difference in hyperparameters between the two runs"
        raise RuntimeError(msg)
