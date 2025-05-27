# ============================================================
# Projeto: Transcrição de Chamadas
# Equipe: [Gustavo Detoni, Felippe Wurcker]
# Data de criação: [24-05-2025]
# Descrição: Este arquivo contém o pipeline de pré-processamento
# textual, vetorização e classificação supervisionada das transcrições.
# O objetivo é aplicar modelos de machine learning para classificar 
# interações telefônicas com base nas transcrições geradas.
# Histórico de alterações:
# - [22-05-2025], [Felippe], [Criação do pipeline de classificação textual]
# - [24-05-2025], [Gustavo], [Adicionado o modelos de classificação]
# ============================================================

import pandas as pd
import spacy
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Baixar stopwords do NLTK (só precisa fazer uma vez)
nltk.download('stopwords')
from nltk.corpus import stopwords

# === 1. Carregar os dados ===
df = pd.read_csv("./results/transcricoes_com_rotulos.csv")

# === 2. Pré-processamento com SpaCy ===
nlp = spacy.load("pt_core_news_sm")
stop_words = set(stopwords.words("portuguese"))
punctuations = string.punctuation

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.text not in stop_words and token.text not in punctuations and not token.is_space
    ]
    return " ".join(tokens)

df["texto_limpo"] = df["transcription"].astype(str).apply(preprocess)

# === 3. Vetorização TF-IDF ===
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["texto_limpo"])
y = df["label"]

# === 4. Divisão treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Modelos de Classificação ===

# a) Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# b) SVM
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# === 6. Avaliação ===
print("=== Naive Bayes ===")
print(classification_report(y_test, y_pred_nb))

print("=== SVM ===")
print(classification_report(y_test, y_pred_svm))

# === 7. Matriz de confusão ===
cm = confusion_matrix(y_test, y_pred_svm, labels=svm.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=svm.classes_, yticklabels=svm.classes_, cmap="Blues")
plt.title("Matriz de Confusão - SVM")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("./results/matriz_confusao_svm.png")
plt.show()
