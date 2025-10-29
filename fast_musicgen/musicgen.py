from transformers import EncodecModel
import torch
from loguru import logger

from .text_encoder import TextEncoder
from .lm import LMRunner


class MusicGeneration:
    def __init__(
        self,
        lm_dir: str = "checkpoints/facebook/musicgen-medium",
        audio_decoder_dir: str = "checkpoints/facebook/encodec_32khz",
        text_encoder_dir: str = "checkpoints/t5-base",
        cuda_graph: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        prompt_max_len: int = 20,
        max_duration_s: int = 30,
    ) -> None:
        self.cuda_graph = cuda_graph
        self.max_duration_s = max_duration_s
        self.text_encoder: TextEncoder = TextEncoder(
            t5_name=text_encoder_dir, device=device
        )
        self.lm: LMRunner = LMRunner(
            checkpoint_dir=lm_dir,
            cuda_graph=cuda_graph,
            device=device,
            prompt_max_len=prompt_max_len,
            max_length=int(self.max_duration_s) * 50,  # 50 tokens per second
        )
        self.audio_decoder: EncodecModel = EncodecModel.from_pretrained(
            audio_decoder_dir
        ).to(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str = "60s happy rock",
        duration_s: int = 30,
        temperature: float = 1.0,
        top_k: int = 50,
        guidance_coef: float = 3,
    ) -> torch.Tensor:
        duration_s = min(duration_s, self.max_duration_s)
        max_steps = int(duration_s) * 50  # 50 tokens per second
        text_x = self.text_encoder.encode(prompt)
        text_len = text_x.shape[1]
        if self.cuda_graph and text_len <= self.lm.prompt_max_len:
            audio_codes = self.lm.generate_with_cuda_graph(
                text_x=text_x,
                max_steps=max_steps,
                temperature=temperature,
                top_k=top_k,
                guidance_coef=guidance_coef,
            )
        else:
            audio_codes = self.lm.generate(
                text_x=text_x,
                max_steps=max_steps,
                temperature=temperature,
                top_k=top_k,
                guidance_coef=guidance_coef,
            )
        audio = (
            self.audio_decoder.decode(audio_codes=audio_codes, audio_scales=[None])
            .audio_values.detach()
            .cpu()
            .numpy()
        )
        return audio.squeeze().squeeze()

    @property
    def sample_rate(self) -> int:
        return self.audio_decoder.config.sampling_rate
