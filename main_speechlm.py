import fire
from omegaconf import OmegaConf

from src.speechlm.eval import evaluate
from src.speechlm.tokenize import encode, tokenize, tokenize_slm21
from src.speechlm.train import train


class TaskRunner:
    def encode(self, config: str = "configs/speechlm/hubert.yaml", spkids: str = "1-9"):
        config = OmegaConf.load(config)
        encode(config, spkids)

    def tokenize(self, config: str = "configs/speechlm/hubert.yaml"):
        config = OmegaConf.load(config)
        tokenize(config)

    def tokenize_slm21(self, config: str = "configs/speechlm/hubert.yaml"):
        config = OmegaConf.load(config)
        tokenize_slm21(config)

    def train(self, config: str = "configs/speechlm/hubert.yaml"):
        config = OmegaConf.load(config)
        train(config)

    def eval(self, config: str = "configs/speechlm/hubert.yaml"):
        config = OmegaConf.load(config)
        evaluate(config)

    def __call__(self, config: str = "configs/speechlm/hubert.yaml", spkids: str = "1-9"):
        config = OmegaConf.load(config)
        encode(config, spkids)
        tokenize(config)
        tokenize_slm21(config)
        train(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
