# ============================================================
# Projeto: Transcrição de Chamadas
# Equipe: [Gustavo Detoni, Felipe Wurcker]
# Data de criação: [28-03-2025]
# Descrição: Este arquivo contém a análise exploratória dos dados de transcrição de chamadas.
# O objetivo é compreender as características dos dados e preparar para a modelagem posterior.
# Histórico de alterações:
# - [23-03-2025], [Gustavo], [Configuração do modelo whisper]
# - [24-03-2025], [Felipe], [Configuração da transcrição]
# - [24-03-2025], [Felipe], [Ajuste na leitura dos arquivos de áudio]
# ============================================================

import time
import torch
from faster_whisper import WhisperModel
from datetime import datetime
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "base"
compute_type = "float16" if device == "cuda" else "float32"

model = WhisperModel(model_size, device=device, compute_type=compute_type)

audio_dir = Path("/dataset")

if not audio_dir.exists():
    print(f"Pasta '{audio_dir}' não encontrada!")
    exit()

audio_files = [audio_file for audio_file in audio_dir.glob("*") if audio_file.suffix in {".mp3", ".wav", ".opus", ".mp4"}]

results = []

for audio_file in audio_files:
    print(f"Testando {audio_file.name}...")
    
    start_time = time.time()
    segments, _ = model.transcribe(str(audio_file), beam_size=5, word_timestamps=False, condition_on_previous_text=False)
    end_time = time.time()
    
    transcription = " ".join(segment.text for segment in segments)
    processing_time = end_time - start_time
    
    results.append({
        "file": audio_file.name,
        "transcription": transcription,
        "processing_time": processing_time,
        "timestamp": str(datetime.now())
    })

for result in results:
    print("\nArquivo:", result["file"])
    print("Tempo de processamento:", round(result["processing_time"], 2), "segundos")
    print("Transcrição:", result["transcription"])
