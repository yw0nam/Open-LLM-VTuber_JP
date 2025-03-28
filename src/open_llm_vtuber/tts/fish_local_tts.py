from typing import Literal
from loguru import logger
from .tts_interface import TTSInterface
from pathlib import Path

import ormsgpack
import requests
import os
from pydantic import BaseModel, Field, conint, model_validator
import base64
from typing_extensions import Annotated


class ServeReferenceAudio(BaseModel):
    audio: bytes
    text: str

    @model_validator(mode="before")
    def decode_audio(cls, values):
        audio = values.get("audio")
        if (
            isinstance(audio, str) and len(audio) > 255
        ):  # Check if audio is a string (Base64)
            try:
                values["audio"] = base64.b64decode(audio)
            except Exception as e:
                # If the audio is not a valid base64 string, we will just ignore it and let the server handle it
                pass
        return values

    def __repr__(self) -> str:
        return f"ServeReferenceAudio(text={self.text!r}, audio_size={len(self.audio)})"


class ServeTTSRequest(BaseModel):
    text: str
    chunk_length: Annotated[int, conint(ge=100, le=300, strict=True)] = 200
    # Audio format
    format: Literal["wav", "pcm", "mp3"] = "wav"
    # References audios for in-context learning
    references: list[ServeReferenceAudio] = []
    # Reference id
    # For example, if you want use https://fish.audio/m/7f92f8afb8ec43bf81429cc1c9199cb1/
    # Just pass 7f92f8afb8ec43bf81429cc1c9199cb1
    reference_id: str | None = None
    seed: int | None = None
    use_memory_cache: Literal["on", "off"] = "off"
    # Normalize text for en & zh, this increase stability for numbers
    normalize: bool = True
    # not usually used below
    streaming: bool = False
    max_new_tokens: int = 1024
    top_p: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7
    repetition_penalty: Annotated[float, Field(ge=0.9, le=2.0, strict=True)] = 1.2
    temperature: Annotated[float, Field(ge=0.1, le=1.0, strict=True)] = 0.7

    class Config:
        # Allow arbitrary types for pytorch related types
        arbitrary_types_allowed = True


class TTSEngine(TTSInterface):
    """
    Fish TTS that calls the FishTTS API service with Coqui-style config in __init__.
    """

    file_extension: str = "wav"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        api_key: str | None = None,
        reference_audio_paths: str | None = None,
        reference_texts: str | None = None,
        format: str = "wav",
        seed: int | None = None,
        streaming: bool = False,
        use_memory_cache: Literal["on", "off"] = "off",
        chunk_length: int = 200,
        max_new_tokens: int = 1024,
        top_p: float = 0.7,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
    ):
        """
        Initialize FishTTS Engine with all synthesis parameters.

        Args:
            base_url: TTS API URL
            api_key: Authentication token
            reference_audio_paths: string of audio paths (use <sep> for multiple)
            reference_texts: string of matching reference texts (use <sep> for multiple)
            format: Output format (wav/mp3/flac)
            seed: Seed for deterministic generation
            streaming: Whether to stream audio output
            use_memory_cache: Whether to cache reference encodings
            chunk_length: Length of each synthesis chunk
            max_new_tokens: Max new tokens to generate
            top_p: Top-p sampling value
            repetition_penalty: Penalty for repeating phrases
            temperature: Sampling temperature
        """

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.tts_url = f"{self.base_url}/v1/tts"

        if reference_audio_paths is not None:
            if not isinstance(reference_audio_paths, str):
                raise TypeError("reference_audio_paths must be a string if provided")
            self.reference_audio_paths = reference_audio_paths.split("<sep>")
        else:
            self.reference_audio_paths = []

        if reference_texts is not None:
            if not isinstance(reference_texts, str):
                raise TypeError("reference_texts must be a string if provided")
            self.reference_texts = reference_texts.split("<sep>")
        else:
            self.reference_texts = []

        self.format = format
        self.seed = seed
        self.streaming = streaming
        self.use_memory_cache = use_memory_cache
        self.chunk_length = chunk_length
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature

        if self.reference_audio_paths:
            byte_audios = [self.audio_to_bytes(p) for p in self.reference_audio_paths]
            ref_texts = [self.read_ref_text(t) for t in self.reference_texts]
            self.references = [
                ServeReferenceAudio(audio=a or b"", text=t)
                for a, t in zip(byte_audios, ref_texts)
            ]
        else:
            self.references = []

        logger.info(
            f"FishTTS initialized at {self.base_url}, reference_audio_paths: {self.reference_audio_paths}"
        )

    def audio_to_bytes(self, file_path: str) -> bytes | None:
        if not file_path or not Path(file_path).exists():
            return None
        with open(file_path, "rb") as f:
            return f.read()

    def read_ref_text(self, ref_text: str) -> str:
        path = Path(ref_text)
        if path.exists() and path.is_file():
            with path.open("r", encoding="utf-8") as f:
                return f.read()
        return ref_text

    def generate_audio(self, text: str, file_name_no_ext: str = None) -> str | None:
        file_name = self.generate_cache_file_name(file_name_no_ext, self.file_extension)

        request_data = ServeTTSRequest(
            text=text,
            format=self.format,
            references=self.references,
            reference_id=None,  # We assume in-context only here
            seed=self.seed,
            use_memory_cache=self.use_memory_cache,
            streaming=self.streaming,
            chunk_length=self.chunk_length,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
        )

        try:
            response = requests.post(
                self.tts_url,
                data=ormsgpack.packb(
                    request_data, option=ormsgpack.OPT_SERIALIZE_PYDANTIC
                ),
                stream=self.streaming,
                headers={
                    "authorization": f"Bearer {self.api_key or ''}",
                    "content-type": "application/msgpack",
                },
            )

            if response.status_code != 200:
                logger.error(f"TTS API error {response.status_code}: {response.text}")
                return None

            with open(file_name, "wb") as f:
                if self.streaming:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                else:
                    f.write(response.content)

            return file_name

        except Exception as e:
            logger.exception("Failed to generate audio")
            return None


# %%
"".split("<sep>")
# %%
