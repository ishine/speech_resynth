import fire
from omegaconf import OmegaConf

from src.speechlm.eval import evaluate
from src.speechlm.tokenize import tokenize_slm21, tokenize_trainset
from src.speechlm.train import train


class TaskRunner:
    def tokenize_trainset(self, config: str = "configs/speechlm/hubert.yaml"):
        config = OmegaConf.load(config)
        tokenize_trainset(config)

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
        tokenize_trainset(config, spkids)
        tokenize_slm21(config)
        train(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
