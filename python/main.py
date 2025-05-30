# ============================================================
# Projeto: Transcrição de Chamadas
# Equipe: [Gustavo Detoni, Felipe Wurcker]
# Data de criação: [28-03-2025]
# Descrição: Este arquivo contém a análise exploratória dos dados de transcrição de chamadas.
# O objetivo é compreender as características dos dados e preparar para a modelagem posterior.
# Histórico de alterações:
# - [23-03-2025], [Gustavo], [Configuração do modelo whisper]
# - [24-03-2025], [Felippe], [Configuração da transcrição]
# - [24-03-2025], [Felippe], [Ajuste na leitura dos arquivos de áudio]
# ============================================================

import time
import torch
from faster_whisper import WhisperModel
from datetime import datetime
from pathlib import Path
import pandas as pd
import random
import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "base"
compute_type = "float16" if device == "cuda" else "float32"

model = WhisperModel(model_size, device=device, compute_type=compute_type)

audio_dir = Path("./dataset")

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

# Salvar os resultados em CSV
df = pd.DataFrame(results)
df.to_csv("./results/transcricoes_resultados.csv", index=False)
print("Transcrições salvas em: ./results/transcricoes_resultados.csv")

# Categorias fictícias típicas de cold calls de investimento
categorias = ['interessado', 'recusa', 'pedido_de_informacao', 'desligou', 'sem_resposta']
random.seed(42)  # Para reprodutibilidade

# Adiciona uma nova coluna com rótulo aleatório
df['label'] = [random.choice(categorias) for _ in range(len(df))]

# Salva novamente com rótulos
df.to_csv("./results/transcricoes_com_rotulos.csv", index=False)
print("Arquivo com rótulos salvo em: ./results/transcricoes_com_rotulos.csv")

# Executando o arquivo classificacao_textual.py
print("Executando pipeline de classificação...")
subprocess.run(["python", "python/classificacao_textual.py"])
