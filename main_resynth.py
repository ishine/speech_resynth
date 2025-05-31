import fire
from omegaconf import OmegaConf

from src.bigvgan.train import train_bigvgan
from src.flow_matching.preprocess import resample
from src.flow_matching.synthesize import synthesize
from src.flow_matching.train import train_flow_matching


class TaskRunner:
    def resample(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        resample(config)

    def train_bigvgan(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        train_bigvgan(config)

    def train_flow_matching(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        train_flow_matching(config)

    def synthesize(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        synthesize(config)


if __name__ == "__main__":
    fire.Fire(TaskRunner)
