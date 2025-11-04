from collections import deque
from typing import Deque, List, Dict
from enum import Enum, auto
from queue import Queue
import uuid

import torch


class Scheduler:
    def __init__(self, max_batch_size: int = 2):
        super().__init__()
        self.max_batch_size = max_batch_size
        self.free_cache_batch_ids: Deque[int] = deque(
            range(max_batch_size * 2)
        )  # 2 batchs per audio sequence
        self.used_cache_batch_ids: Deque[int] = deque()
        self.waiting: Deque[AudioSequence] = deque()
        self.running: List[AudioSequence] = []
        self.response_queues: Dict[str, Queue] = {}

    def add(self, audio_sequence: "AudioSequence", response: Queue = None) -> None:
        if audio_sequence.id in self.response_queues:
            raise ValueError(f"Audio sequence {audio_sequence.id} already exists")
        if response is None:
            response = Queue()
        self.response_queues[audio_sequence.id] = response
        self.waiting.append(audio_sequence)

    def schedule(self) -> List["AudioSequence"]:
        if len(self.running) == self.max_batch_size:
            return self.running
        else:
            while len(self.running) < self.max_batch_size:
                if len(self.waiting) > 0 and self.can_allocate_cache():
                    audio_sequence = self.waiting.popleft()
                    self.running.append(audio_sequence)
                    self.allocate_cache(audio_sequence)
                else:
                    break
            return self.running

    def allocate_cache(self, audio_sequence: "AudioSequence") -> "AudioSequence":
        cache_batch_idx = []
        for _ in range(2):
            cache_batch_idx.append(self.free_cache_batch_ids.popleft())
        audio_sequence.cache_batch_idx = cache_batch_idx
        audio_sequence.status = Status.RUNNING
        return audio_sequence

    def can_allocate_cache(self) -> bool:
        return len(self.free_cache_batch_ids) >= 2

    def deallocate_cache(self, audio_sequence: "AudioSequence") -> None:
        assert (
            len(audio_sequence.cache_batch_idx) == 2
        ), f"audio_sequence.cache_batch_idx: {audio_sequence.cache_batch_idx}"
        for batch_idx in audio_sequence.cache_batch_idx:
            self.free_cache_batch_ids.append(batch_idx)
        audio_sequence.cache_batch_idx = []
        audio_sequence.status = Status.FINISHED
        self.waiting.append(audio_sequence)

    def check_finished(self) -> None:
        for audio_sequence in self.running:
            if audio_sequence.is_finished:
                self.deallocate_cache(audio_sequence)
                self.running.remove(audio_sequence)
                self.response_queues[audio_sequence.id].put(audio_sequence)
                del self.response_queues[audio_sequence.id]


class Status(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()


class AudioSequence:
    def __init__(self, max_steps: int = 500) -> None:
        super().__init__()
        self.id = str(uuid.uuid4())
        self.status = Status.WAITING
        self.cache_batch_idx = []
        self.max_steps = max_steps
        self.offset: int = 0

        self.audio_seq: torch.Tensor = None
        self.text_tokens: torch.Tensor = None
        self.text_x: torch.Tensor = None

    @property
    def is_finished(self) -> bool:
        return self.offset >= (self.max_steps - 1)
