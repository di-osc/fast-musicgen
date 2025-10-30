# 🎵 Fast MusicGen

High-performance music generation engine based on Meta MusicGen, featuring **3x speedup** with CUDA Graph and Flash Attention optimization!

## ✨ Key Features

- 🚀 **3x Faster**: CUDA Graph optimization achieves 120+ tokens/sec on RTX 4090 (vs ~40 tokens/sec baseline)
- ⚡ **CUDA Graph**: Pre-compiled computation graphs eliminate CPU-GPU communication overhead for extreme performance
- 🔥 **Flash Attention**: Efficient attention mechanism dramatically reduces memory usage and accelerates inference
- 🎼 **Text-to-Music**: Generate high-quality music from natural language descriptions
- 🔧 **Modular Design**: Independent text encoder, language model, and audio decoder components
- 📦 **Ready to Use**: Simple API for quick integration

## 📦 Installation

### Prerequisites

1. **Install PyTorch** with CUDA support:
   ```bash
   # Find the appropriate PyTorch wheel for your CUDA version at:
   # https://pytorch.org/get-started/locally/
   pip install torch torchvision torchaudio
   ```

2. **Install Flash Attention** (highly recommended for optimal performance):
   ```bash
   # Find the compatible wheel for your CUDA version and PyTorch version at:
   # https://github.com/Dao-AILab/flash-attention/releases
   pip install flash-attn --no-build-isolation
   ```
3. **Download the checkpoints**
   ```bash
   pip install huggingface_hub
   hf download facebook/musicgen-medium --local-dir checkpoints/facebook/musicgen-medium
   hf download facebook/encodec_32khz --local-dir checkpoints/facebook/encodec_32khz
   hf download t5-base --local-dir checkpoints/t5-base
   ```

### Install Fast MusicGen

```bash
pip install fast-musicgen
```

## 🛠️ Quick Start

```python
from fast_musicgen import MusicGeneration
import torchaudio

# Enable CUDA Graph acceleration (enabled by default)
model = MusicGeneration(cuda_graph=True)

# Generate music
audio = model.generate("happy rock", duration_s=10)

# Save audio
torchaudio.save("music.wav", audio.cpu(), model.sample_rate)
```

## 📊 Performance Comparison

| Model | Configuration | Original MusicGen | Fast MusicGen | Speedup |
|-------|---------------|-------------------|---------------|---------|
|MusicGen-Medium| RTX 4090     | ~40 tokens/sec    | 130+ tokens/sec | **3x** |

---

*Built on Meta MusicGen • [License: MIT](LICENSE)*