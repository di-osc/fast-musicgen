from transformers import EncodecModel
import torch

from .text_encoder import TextEncoder
from .lm import CausalLM


class MusicGeneration:
    def __init__(
        self,
        lm_dir: str,
        audio_decoder_dir: str = "facebook/encodec_32khz",
        text_encoder_dir: str = "t5-base",
    ) -> None:
        self.text_encoder: TextEncoder = TextEncoder(t5_name=text_encoder_dir)
        self.lm: CausalLM = CausalLM.from_pretrained(ckpt_dir=lm_dir)
        self.audio_decoder: EncodecModel = EncodecModel.from_pretrained(
            audio_decoder_dir
        )

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_steps: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        guidance_coef: float = 3,
    ) -> torch.Tensor:
        text_x = self.text_encoder.encode(text)
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
