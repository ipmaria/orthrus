from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from typing import Optional

class OrthrusConfig(Qwen3Config):
    model_type = "orthrus"

    def __init__(
        self,
        *args,
        block_size: Optional[int] = None,
        mask_token_id: Optional[int] = None,
        noise_reveal_top_m: Optional[int] = None,
        dblast_num_categories: Optional[int] = None,
        dblast_category_temperature: float = 1.0,
        dblast_init_scale: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        # Noise-conditioned drafting: reveal the verifier's Gumbel noise on the top-m
        # tokens of the noise-free proposal (src/noise_coupling.py). None = off.
        self.noise_reveal_top_m = noise_reveal_top_m
        # DBLAST re-implementation (src/dblast.py): K categories (None = off; 1 = the
        # plain drafter decoded with the greedy-branch rule), inference category
        # temperature Z_T, and the expander init scale (relative, / sqrt(hidden)).
        self.dblast_num_categories = dblast_num_categories
        self.dblast_category_temperature = dblast_category_temperature
        self.dblast_init_scale = dblast_init_scale