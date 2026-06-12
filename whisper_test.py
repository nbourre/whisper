import mlx_whisper

result = mlx_whisper.transcribe(
    "data/test.mp3",
    path_or_hf_repo="mlx-community/whisper-large-v3-mlx"
)
print(result["text"])