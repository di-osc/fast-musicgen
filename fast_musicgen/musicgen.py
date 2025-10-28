from transformers import EncodecModel
import torch

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
    ) -> None:
        self.cuda_graph = cuda_graph
        self.text_encoder: TextEncoder = TextEncoder(
            t5_name=text_encoder_dir, device=device
        )
        self.lm: LMRunner = LMRunner(
            checkpoint_dir=lm_dir,
            cuda_graph=cuda_graph,
            device=device,
            prompt_max_len=prompt_max_len,
        )
        self.audio_decoder: EncodecModel = EncodecModel.from_pretrained(
            audio_decoder_dir
        ).to(device)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str = "60s happy rock",
        max_steps: int = 500,
        temperature: float = 1.0,
        top_k: int = 50,
        guidance_coef: float = 3,
    ) -> torch.Tensor:
        text_x = self.text_encoder.encode(prompt)
        if self.cuda_graph:
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
