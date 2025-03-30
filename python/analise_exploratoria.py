# ============================================================
# Projeto: Análise Exploratória de Dados
# Equipe: [Gustavo Detoni, Felipe Wurcker]
# Data de criação: [28-03-2025]
# Descrição: Este arquivo contém a análise exploratória dos dados de transcrição de chamadas.
# O objetivo é compreender as características dos dados e preparar para a modelagem posterior.
# Histórico de alterações:
# - [25-03-2025], [Gustavo], [Configuração do modelo whisper]
# - [25-03-2025], [Gustavo], [Configuração da transcrição]
# - [26-03-2025], [Gustavo], [Configuração do modelo whisper]
# - [26-03-2025], [Gustavo], [Adiciondo toda a análise exploratória]
# - [28-03-2025], [Felipe], [Configuração da dos gráficos]
# - [28-03-2025], [Felipe], [Ajustes/Tramentamento dos dados]
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
from faster_whisper import WhisperModel
from datetime import datetime
from pathlib import Path
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"
model_size = "base"
compute_type = "float16" if device == "cuda" else "float32"
model = WhisperModel(model_size, device=device, compute_type=compute_type)

audio_dir = Path("../dataset")

audio_files = [audio_file for audio_file in audio_dir.glob("*") if audio_file.suffix in {".mp3", ".wav", ".opus", ".mp4"}]

results = []

for audio_file in audio_files:
    print(f"Transcrevendo {audio_file.name}...")
    
    start_time = time.time()
    segments, info = model.transcribe(str(audio_file), beam_size=5)
    end_time = time.time()
    
    transcription = " ".join(segment.text for segment in segments)
    processing_time = end_time - start_time
    
    results.append({
        "file": audio_file.name,
        "transcription": transcription,
        "processing_time": processing_time,
        "duration": info.duration,
        "language": info.language
    })

# Converter para DataFrame
df = pd.DataFrame(results)

# Adicionar contagem de palavras
df['word_count'] = df['transcription'].apply(lambda x: len(str(x).split()))
df['words_per_second'] = df['word_count'] / df['duration']

print(f"Número de transcrições: {len(df)}")
print(f"Duração média dos áudios: {df['duration'].mean():.2f} segundos")
print(f"Número médio de palavras: {df['word_count'].mean():.2f}")
print(f"Taxa média de palavras por segundo: {df['words_per_second'].mean():.2f}")

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
sns.barplot(data=df, x='file', y='duration')
plt.title('Duração dos Áudios')
plt.xticks(rotation=45)
plt.tight_layout()

plt.subplot(2, 1, 2)
sns.barplot(data=df, x='file', y='word_count')
plt.title('Número de Palavras por Transcrição')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('analise_simples.png')
plt.close()

# Juntar todas as transcrições
all_text = " ".join(df['transcription'].fillna(""))
words = [word.lower() for word in all_text.split() if len(word) > 3]
word_counts = Counter(words)

print("Palavras mais frequentes:")
for word, count in word_counts.most_common(10):
    print(f"  - {word}: {count}")

plt.figure(figsize=(10, 6))
words_df = pd.DataFrame(word_counts.most_common(15), columns=['palavra', 'contagem'])
sns.barplot(data=words_df, x='contagem', y='palavra')
plt.title('15 Palavras Mais Frequentes')
plt.tight_layout()
plt.savefig('palavras_frequentes.png')

print("\nAnálise exploratória simplificada concluída!")
print("Resultados foram salvos em 'transcricoes_simples.csv'")
print("Gráficos foram salvos como 'analise_simples.png' e 'palavras_frequentes.png'")