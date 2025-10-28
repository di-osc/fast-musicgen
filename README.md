# 🎵 Fast MusicGen

High-performance music generation from text using Meta MusicGen with CUDA Graph optimization.

## ✨ Features

- 🚀 **2.5x Faster**: CUDA Graph optimization (100+ tokens/sec vs ~40 tokens/sec on RTX 4090)
- 🎼 **Text-to-Music**: Generate music from text descriptions
- 🔧 **Modular**: Text encoder, language model, and audio decoder components
- 📦 **Easy to Use**: Simple API for quick integration

## 🛠️ Quick Start

```bash
pip install -e .
```

```python
from fast_musicgen import MusicGeneration

model = MusicGeneration(cuda_graph=True)  # Enable CUDA Graph for speed

# Generate music
audio = model.generate("Happy jazz music, perfect for cafe background")

# Save audio
import torchaudio
torchaudio.save("music.wav", audio.cpu(), 32000)
```

## 🎯 Architecture

```
Text → TextEncoder → LMRunner → AudioDecoder → Audio
                    ↑
              CUDA Graph
```

---

*Built on Meta MusicGen • [License: MIT](LICENSE)*