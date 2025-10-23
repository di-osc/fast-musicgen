from pydantic import BaseModel

from .text_encoder import TextEncoder


class MusicGeneration(BaseModel):
    text_encoder: TextEncoder
