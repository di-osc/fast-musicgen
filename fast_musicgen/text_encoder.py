import torch
from transformers import AutoTokenizer, T5EncoderModel


class TextEncoder:
    def __init__(self, t5_name: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self._t5 = T5EncoderModel.from_pretrained(t5_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(t5_name)

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        x = self._t5(input_ids).last_hidden_state
        return x
