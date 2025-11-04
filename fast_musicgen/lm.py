from __future__ import annotations
from typing import List, Dict, Tuple
from pathlib import Path
from queue import Queue
import time
import threading

import torch.nn as nn
import torch
from transformers import MusicgenConfig
from tqdm import tqdm

from .layers.ffn import FeedForward
from .layers.cross_attn import CrossAttention
from .layers.causal_attn import CausalAttention
from .scheduler import AudioSequence, Scheduler


class LMRunner:
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/facebook/musicgen-medium",
        prompt_max_len: int = 20,
        max_length: int = 1500,
        max_batch_size: int = 2,
        cuda_graph: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Args:
            checkpoint_dir (str, optional): The directory containing the checkpoint. Defaults to "checkpoints/facebook/musicgen-medium".
            prompt_max_len (int, optional): The maximum length of the prompt. Defaults to 20.
            max_length (int, optional): The maximum length of the cache. Defaults to 1500.
            max_batch_size (int, optional): The maximum batch size to use. Defaults to 4.
            cuda_graph (bool, optional): Whether to use CUDA graph. Defaults to True.
            device (str, optional): The device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): The data type to use. Defaults to torch.float16.
        """
        super().__init__()
        torch.set_default_device(device)
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.model = CausalLM.from_pretrained(checkpoint_dir, device=device)
        self.model.set_cache(max_length=max_length, batch_size=max_batch_size)
        self.cuda_graph = cuda_graph
        self.prompt_max_len = prompt_max_len
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        self.warmup()
        self.graphs = {}
        self.graph_vars = {}
        if self.cuda_graph:
            self.capture_cuda_graph()

    @torch.inference_mode()
    def capture_cuda_graph(self):
        for batch_size in range(1, self.max_batch_size + 1):
            self.graph_vars[batch_size] = {}
            self.graphs[batch_size] = {}
            for i in tqdm(range(1, self.prompt_max_len + 1)):
                text_x = torch.zeros(
                    2 * batch_size,
                    i,
                    self.model.config.decoder.hidden_size,
                    dtype=self.model.dtype,
                    device=self.model.device,
                )
                audio_x = torch.full(
                    (2 * batch_size, 1, 4),
                    self.model.config.decoder.bos_token_id,
                    dtype=torch.long,
                    device=self.model.device,
                )
                input_pos = torch.zeros(
                    2 * batch_size, dtype=torch.long, device=self.model.device
                )
                cache_seqlens = torch.zeros(
                    2 * batch_size, dtype=torch.int32, device=self.model.device
                )
                cache_batch_idx = torch.zeros(
                    2 * batch_size, dtype=torch.int32, device=self.model.device
                )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self.model(
                        audio_x, text_x, input_pos, cache_seqlens, cache_batch_idx
                    )
                self.graphs[batch_size][i] = graph
                self.graph_vars[batch_size][i] = {
                    "text_x": text_x,
                    "audio_x": audio_x,
                    "input_pos": input_pos,
                    "cache_seqlens": cache_seqlens,
                    "cache_batch_idx": cache_batch_idx,
                    "output": output,
                }

    @torch.inference_mode()
    def warmup(self):
        for batch_size in range(1, self.max_batch_size + 1):
            for i in tqdm(range(2)):
                text_x = torch.randn(
                    2 * batch_size,
                    i,
                    self.model.config.decoder.hidden_size,
                    dtype=self.model.dtype,
                    device=self.model.device,
                )
                audio_x = torch.full(
                    (2 * batch_size, 1, 4),
                    self.model.config.decoder.bos_token_id,
                    dtype=torch.long,
                    device=self.model.device,
                )
                input_pos = torch.tensor([i] * 2 * batch_size, dtype=torch.long, device=self.model.device)
                cache_seqlens = torch.zeros(2 * batch_size, dtype=torch.int32, device=self.model.device)
                cache_batch_idx = torch.zeros(
                    2 * batch_size, dtype=torch.int32, device=self.model.device
                )
                _ = self.model(audio_x, text_x, input_pos, cache_seqlens, cache_batch_idx)
        self.model.reset_cache()
        torch.cuda.synchronize()

    @torch.inference_mode()
    def generate_with_cuda_graph(
        self,
        text_x: torch.Tensor,
        max_steps: int = 150,
        temperature: float = 1.0,
        top_k: int = 250,
        guidance_coef: float = 3.0,
    ) -> torch.Tensor:
        text_x = text_x.to(self.model.device).to(self.model.dtype)
        text_x = self.model.text_proj(text_x)
        # Compute conditional and unconditional logits in one batch
        text_x = torch.concatenate(
            [text_x, torch.zeros_like(text_x).to(self.model.device)], dim=0
        )

        audio_shape = (1, max_steps + 1, self.model.num_codebooks)
        audio_seq = torch.full(audio_shape, self.model.bos_token_id).to(
            self.model.device
        )

        prompt_len = text_x.shape[1]
        graph_vars = self.graph_vars[prompt_len]
        for offset in tqdm(range(max_steps)):
            audio_input = torch.tile(audio_seq[:, offset : offset + 1], [2, 1, 1])
            assert graph_vars["audio_x"].dtype == audio_input.dtype
            assert graph_vars["text_x"].dtype == text_x.dtype
            graph_vars["audio_x"].copy_(audio_input)
            graph_vars["text_x"].copy_(text_x)
            graph_vars["input_pos"].fill_(offset)
            graph_vars["cache_seqlens"].fill_(offset + 1)
            graph = self.graphs[prompt_len]
            graph.replay()
            audio_logits = graph_vars["output"]
            cond_logits, uncond_logits = audio_logits[:1], audio_logits[1:2]
            audio_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_coef
            audio_tokens = self.model.top_k_sampling(
                logits=audio_logits, top_k=top_k, temperature=temperature, dim=-2
            )
            # "delay" pattern
            audio_tokens[..., offset + 1 :] = self.model.bos_token_id
            audio_tokens[..., : -max_steps + offset] = self.model.bos_token_id
            audio_seq[:, offset + 1 : offset + 2] = audio_tokens
        # Undo delay
        for i in range(self.model.num_codebooks):
            audio_seq[:, : -self.model.num_codebooks, i] = audio_seq[
                :, i : -self.model.num_codebooks + i, i
            ]

        audio_seq = audio_seq[:, 1 : -self.model.num_codebooks + 1]
        audio_seq = torch.swapaxes(audio_seq, -1, -2)[:, torch.newaxis]
        # (2, 1, 2048, 4)
        return audio_seq

    @torch.inference_mode()
    def generate(
        self,
        text_x: torch.Tensor,
        max_steps: int = 150,
        temperature: float = 1.0,
        top_k: int = 250,
        guidance_coef: float = 3.0,
    ) -> torch.Tensor:
        text_x = text_x.to(self.model.device).to(self.model.dtype)
        text_x = self.model.text_proj(text_x)
        # Assuming no audio prompt we start with all bos token for the codebooks
        audio_shape = (1, max_steps + 1, self.model.num_codebooks)
        audio_seq = torch.full(audio_shape, self.model.bos_token_id).to(
            self.model.device
        )

        # Compute conditional and unconditional logits in one batch
        text_tokens = torch.concatenate(
            [text_x, torch.zeros_like(text_x).to(self.model.device)], dim=0
        )
        input_pos = torch.zeros(1, dtype=torch.long, device=self.model.device)
        cache_seqlens = torch.zeros(2, dtype=torch.int32, device=self.model.device)
        cache_batch_idx = torch.tensor(
            [0, 1], dtype=torch.int32, device=self.model.device
        )
        for offset in tqdm(range(max_steps)):
            audio_input = torch.tile(audio_seq[:, offset : offset + 1], [2, 1, 1])
            input_pos.fill_(offset)
            cache_seqlens.fill_(offset + 1)
            audio_logits = self.model(
                audio_input, text_tokens, input_pos, cache_seqlens, cache_batch_idx
            )
            cond_logits, uncond_logits = audio_logits[:1], audio_logits[1:2]
            audio_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_coef
            audio_tokens = self.model.top_k_sampling(
                logits=audio_logits, top_k=top_k, temperature=temperature, dim=-2
            )
            # "delay" pattern
            audio_tokens[..., offset + 1 :] = self.model.bos_token_id
            audio_tokens[..., : -max_steps + offset] = self.model.bos_token_id
            audio_seq[:, offset + 1 : offset + 2] = audio_tokens

        # Undo delay
        for i in range(self.model.num_codebooks):
            audio_seq[:, : -self.model.num_codebooks, i] = audio_seq[
                :, i : -self.model.num_codebooks + i, i
            ]

        audio_seq = audio_seq[:, 1 : -self.model.num_codebooks + 1]
        audio_seq = torch.swapaxes(audio_seq, -1, -2)[:, torch.newaxis]
        return audio_seq

    def prepare(
        self, seqs: List[AudioSequence]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        for seq in seqs:
            seq.text_x = self.model.text_proj(seq.text_x)
            seq.text_tokens = torch.concatenate(
                [seq.text_x, torch.zeros_like(seq.text_x).to(self.model.device)], dim=0
            )
            seq.input_pos = torch.zeros(1, dtype=torch.long, device=self.model.device)
            seq.cache_seqlens = torch.zeros(
                2, dtype=torch.int32, device=self.model.device
            )
            seq.cache_batch_idx = torch.tensor(
                [0, 1], dtype=torch.int32, device=self.model.device
            )

    def run_step(self, seqs: List[AudioSequence]) -> List[AudioSequence]:
        audio_input, text_tokens, input_pos, cache_seqlens, cache_batch_idx = (
            self.prepare(seqs)
        )
        if self.cuda_graph:
            cra

    def generate(self, seqs: List[AudioSequence]) -> List[torch.Tensor]:
        response_queue = Queue()
        for seq in seqs:
            self.scheduler.add(seq, response_queue)
        results = []
        while len(results) < len(seqs):
            results.append(response_queue.get())
        return results

    def start_running(self) -> None:
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self) -> None:
        while True:
            seqs = self.scheduler.schedule()
            if len(seqs) == 0:
                time.sleep(0.01)
                continue
            seqs = self.run_step(seqs)
            self.scheduler.check_finished(seqs)


class CausalLM(nn.Module):
    def __init__(
        self,
        config: MusicgenConfig,
        device: str,
        prompt_max_len: int = 20,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.config = config
        self.num_codebooks = config.decoder.num_codebooks
        self.codebook_size = config.audio_encoder.codebook_size
        self.bos_token_id = config.decoder.bos_token_id
        self.hidden_size = config.decoder.hidden_size
        self.num_attention_heads = config.decoder.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.embed = nn.ModuleList(
            [
                nn.Embedding(self.codebook_size + 1, self.hidden_size)
                for _ in range(self.num_codebooks)
            ]
        )
        self.layers: List[TransformerBlock] = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.decoder.num_hidden_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.heads = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.codebook_size, bias=False)
                for _ in range(self.num_codebooks)
            ]
        )
        self.text_proj = nn.Linear(
            config.text_encoder.d_model, self.hidden_size, bias=True
        )
        self.device = device
        self.dtype = dtype

    def forward(
        self,
        audio_x: torch.Tensor,
        text_x: torch.Tensor,
        input_pos: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        audio_x: (2, 1, num_codebooks)
        text_x: (2, seq_len, hidden_size)
        input_pos: (1,)
        cache_seqlens: (2,)
        cache_batch_idx: (2,)
        """
        audio_x = sum(
            [self.embed[k](audio_x[..., k]) for k in range(self.num_codebooks)]
        )
        pos_emb = self.create_sin_embedding(input_pos, self.hidden_size)
        audio_x += pos_emb.to(audio_x.dtype)
        for layer in self.layers:
            audio_x = layer(
                audio_x=audio_x,
                text_x=text_x,
                input_pos=input_pos,
                cache_seqlens=cache_seqlens,
                cache_batch_idx=cache_batch_idx,
            )
        audio_x = self.out_norm(audio_x)
        audio_x = torch.stack(
            [self.heads[k](audio_x) for k in range(self.num_codebooks)], dim=-1
        )
        return audio_x

    def set_cache(self, max_length: int = 1500, batch_size: int = 2):
        for layer in self.layers:
            layer.self_attn.set_cache(
                max_length=max_length,
                device=self.device,
                dtype=self.dtype,
                batch_size=batch_size * 2,
            )

    def reset_cache(self):
        for layer in self.layers:
            layer.self_attn.reset_cache()

    def create_sin_embedding(
        self, positions: torch.Tensor, dim: int, max_period: float = 10000
    ) -> torch.Tensor:
        assert dim % 2 == 0
        half_dim = dim // 2
        adim = torch.arange(half_dim).reshape(1, 1, -1)
        phase = positions[:, None, None] / (max_period ** (adim / (half_dim - 1)))
        return torch.concatenate([torch.cos(phase), torch.sin(phase)], dim=-1)

    def top_k_sampling(
        self, logits: torch.Tensor, top_k: int, temperature: float, dim: int = -2
    ) -> torch.Tensor:
        values, indices = logits.topk(k=top_k, dim=dim)
        logits = torch.full_like(logits, -float("inf")).scatter_(
            dim=dim, index=indices, src=values
        )
        probs = logits.float().div_(temperature).softmax(dim=dim)
        tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=dim, keepdim=True)
        return tokens

    @classmethod
    def sanitize(cls, states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_states = {}
        for k, arr in states.items():
            if k.startswith("transformer."):
                k = k[len("transformer.") :]
            if "emb" in k:
                k = k.replace("emb", "embed")
            if "cross_attention" in k:
                k = k.replace("cross_attention", "cross_attn")
            if "norm1" in k:
                k = k.replace("norm1", "self_attn_norm")
            if "norm_cross" in k:
                k = k.replace("norm_cross", "cross_attn_norm")
            if "norm2" in k:
                k = k.replace("norm2", "ffn_norm")
            if "linears" in k:
                k = k.replace("linears", "heads")
            if "linear1" in k:
                k = k.replace("linear1", "ffn.up_proj")
            if "linear2" in k:
                k = k.replace("linear2", "ffn.down_proj")
            if "condition_provider.conditioners.description.output_proj" in k:
                k = k.replace(
                    "condition_provider.conditioners.description.output_proj",
                    "text_proj",
                )
            if "in_proj_weight" in k:
                dim = arr.shape[0] // 3
                name = "in_proj_weight"
                new_states[k.replace(name, "q_proj.weight")] = arr[:dim]
                new_states[k.replace(name, "k_proj.weight")] = arr[dim : dim * 2]
                new_states[k.replace(name, "v_proj.weight")] = arr[dim * 2 :]
                continue
            new_states[k] = arr
        return new_states

    @classmethod
    def from_pretrained(cls, ckpt_dir: str, device: str = "cpu") -> CausalLM:
        config = MusicgenConfig.from_pretrained(ckpt_dir)
        with torch.device("meta"):
            model = CausalLM(config, device=device)
        states = torch.load(Path(ckpt_dir) / "state_dict.bin", weights_only=True)[
            "best_state"
        ]
        new_states = cls.sanitize(states)
        model.load_state_dict(new_states, assign=True)
        return model.eval().to(device)


class TransformerBlock(nn.Module):
    def __init__(self, config: MusicgenConfig):
        super().__init__()
        self.self_attn = CausalAttention(config)
        self.self_attn_norm = nn.LayerNorm(config.decoder.hidden_size, eps=1e-5)
        self.cross_attn = CrossAttention(config)
        self.cross_attn_norm = nn.LayerNorm(config.decoder.hidden_size, eps=1e-5)
        self.ffn = FeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.decoder.hidden_size, eps=1e-5)

    def forward(
        self,
        audio_x: torch.Tensor,
        text_x: torch.Tensor,
        input_pos: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cache_batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        # self-attention and residual connection
        audio_xn = self.self_attn_norm(audio_x)
        audio_x += self.self_attn(
            audio_xn,
            audio_xn,
            audio_xn,
            input_pos=input_pos,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=cache_batch_idx,
        )

        # cross-attention and residual connection
        audio_xn = self.cross_attn_norm(audio_x)
        audio_x += self.cross_attn(audio_xn, text_x, text_x)

        # feed-forward and residual connection
        audio_xn = self.ffn_norm(audio_x)
        audio_x += self.ffn(audio_xn)
        return audio_x
