from typing import Dict

from transformers import FastSpeech2ConformerHifiGanConfig, PretrainedConfig

from ..bigvgan.bigvgan import BigVGanConfig


class ConditionalFlowMatchingConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size: int = 2000,
        dim_in: int = 80,
        dim_cond_emb: int = 768,
        hidden_size: int = 256,
        depth: int = 4,
        heads: int = 2,
        intermediate_size: int = 896,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        use_unet_skip_connection: bool = False,
        conv_pos_embed_kernel_size: int = 31,
        conv_pos_embed_groups: int = 256,
        mean: float = -5.8843,
        std: float = 2.2615,
        predict_duration: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.dim_in = dim_in
        self.dim_cond_emb = dim_cond_emb
        self.hidden_size = hidden_size
        self.depth = depth
        self.heads = heads
        self.intermediate_size = intermediate_size
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.use_unet_skip_connection = use_unet_skip_connection
        self.conv_pos_embed_kernel_size = conv_pos_embed_kernel_size
        self.conv_pos_embed_groups = conv_pos_embed_groups
        self.mean = mean
        self.std = std
        self.predict_duration = predict_duration
        super().__init__(**kwargs)


class ConditionalFlowMatchingWithHifiGanConfig(PretrainedConfig):
    model_type = "flow_matching_with_hifigan"
    sub_configs = {"model_config": ConditionalFlowMatchingConfig, "vocoder_config": FastSpeech2ConformerHifiGanConfig}

    def __init__(
        self,
        model_config: Dict = None,
        vocoder_config: Dict = None,
        **kwargs,
    ):
        if model_config is None:
            model_config = {}

        if vocoder_config is None:
            vocoder_config = {}

        self.model_config = ConditionalFlowMatchingConfig(**model_config)
        self.vocoder_config = FastSpeech2ConformerHifiGanConfig(**vocoder_config)
        super().__init__(**kwargs)


class ConditionalFlowMatchingWithBigVGanConfig(PretrainedConfig):
    model_type = "flow_matching_with_bigvgan"
    sub_configs = {"model_config": ConditionalFlowMatchingConfig, "vocoder_config": BigVGanConfig}

    def __init__(
        self,
        model_config: Dict = None,
        vocoder_config: Dict = None,
        **kwargs,
    ):
        if model_config is None:
            model_config = {}

        if vocoder_config is None:
            vocoder_config = {}

        self.model_config = ConditionalFlowMatchingConfig(**model_config)
        self.vocoder_config = BigVGanConfig(**vocoder_config)
        super().__init__(**kwargs)
