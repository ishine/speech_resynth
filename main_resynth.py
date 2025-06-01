import fire
from omegaconf import OmegaConf

from src.bigvgan.train import train_bigvgan
from src.flow_matching.preprocess import extract_features, resample
from src.flow_matching.synthesize import synthesize
from src.flow_matching.train import train_flow_matching
from src.flow_matching.utils.whisper import tokenize_dataset, train_tokenizer


class TaskRunner:
    def resample(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        resample(config)

    def extract_features(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        extract_features(config)

    def train_tokenizer(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        train_tokenizer(config)

    def tokenize_dataset(self, config: str = "configs/unit2speech/whisper-large-v3-4096-bigvgan.yaml"):
        config = OmegaConf.load(config)
        tokenize_dataset(config)

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
