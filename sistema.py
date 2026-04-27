import numpy as np
import yt_dlp
from moviepy.editor import VideoFileClip
import whisper
from transformers import pipeline
import json
from datetime import datetime

# --- Video download and trim ---
url = "https://www.youtube.com/watch?v=KQd7--8acuM&t=3s" # Default URL, can be changed
ydl_opts = {
    'format': 'mp4',
    'outtmpl': 'video.mp4',
    'quiet': True
}

print("Downloading video...")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
print("Video downloaded.")

clip = VideoFileClip("video.mp4").subclip(0, 300)  # first 5 min
clip.write_videofile("video_corto.mp4", verbose=False, logger=None)
archivo = "video_corto.mp4"
print("Video trimmed to 'video_corto.mp4'.")

# --- Whisper model and transcription ---
print("Loading Whisper model...")
model = whisper.load_model("tiny") # Using tiny model for speed
print("Transcribing audio...")
resultado = model.transcribe(archivo)
segmentos = resultado["segments"]
print(f"Audio transcribed. Found {len(segmentos)} segments.")

# --- Candidate definition ---
CANDIDATOS = {
    "clara lopez": ["clara eugenia lopez obregon", "clara lopez"],
    "oscar lizcano": ["oscar mauricio lizcano arango", "oscar lizcano", "lizcano"],
    "raul botero": ["raul santiago botero jaramillo", "raul botero", "botero"],
    "miguel uribe": ["miguel uribe londono", "miguel uribe", "uribe"],
    "sondra macollins": ["sondra macollins garvin pinto"],
    "ivan cepeda": ["ivan cepeda castro", "ivan cepeda", "cepeda"],
    "abelardo de la espriella": ["abelardo gabriel de la espriella", "de la espriella"],
    "claudia lopez": ["claudia nayibe lopez hernandez", "claudia lopez"],
    "paloma valencia": ["paloma susana valencia laserna", "paloma valencia", "valencia"],
    "sergio fajardo": ["sergio fajardo valderrama", "sergio fajardo", "fajardo"],
    "roy barreras": ["roy leonardo barreras montealegre", "roy barreras", "barreras"]
}

# --- detectar_candidatos function ---
def detectar_candidatos(texto):
    texto = texto.lower()
    encontrados = []
    for candidato, alias in CANDIDATOS.items():
        for a in alias:
            if a in texto:
                encontrados.append(candidato)
                break
    return encontrados

# --- clasificar_tono function ---
print("Loading sentiment analysis model...")
sentiment = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def clasificar_tono(texto):
    res = sentiment(texto)[0]
    estrellas = int(res["label"][0])

    if estrellas <= 2:
        return "negativo"
    elif estrellas == 3:
        return "neutral"
    else:
        return "positivo"

# --- Populate tiempos ---
tiempos = {}
print("Analyzing candidate speaking times...")
for seg in segmentos:
    texto = seg["text"]
    duracion = seg["end"] - seg["start"]
    candidatos = detectar_candidatos(texto)
    for c in candidatos:
        tiempos[c] = tiempos.get(c, 0) + duracion

# --- Populate resultados_totales and sentiment ---
resultados_totales = []
print("Analyzing sentiment for segments...")
for seg in segmentos:
    candidatos = detectar_candidatos(seg["text"])
    tono = clasificar_tono(seg["text"])

    resultados_totales.append({
        "texto": seg["text"],
        "inicio": seg["start"],
        "fin": seg["end"],
        "candidatos": candidatos,
        "tono": tono
    })

# --- Calculate equidad, balance, presencia, pluralismo and estado ---
# Calculate equidad
valores = np.array(list(tiempos.values()))

if len(valores) > 0:
    promedio = np.mean(valores)
    equidad = 1 - (np.sum(np.abs(valores - promedio)) / np.sum(valores))
else:
    equidad = 0
print(f"Equidad: {equidad:.2f}")

# Calculate balance
positive_count = 0
negative_count = 0
for r in resultados_totales:
    if r["tono"] == "positivo":
        positive_count += 1
    elif r["tono"] == "negativo":
        negative_count += 1

pos = positive_count
neg = negative_count

if (pos + neg) > 0:
    balance = 1 - abs(pos - neg) / (pos + neg)
else:
    balance = 1
print(f"Balance de tono: {balance:.2f}")

# Calculate presencia
total_candidatos = len(CANDIDATOS)
presentes = len(tiempos)
presencia = presentes / total_candidatos
print(f"Presencia: {presencia:.2f}")

# Calculate pluralismo and estado
pluralismo = (equidad + balance + presencia) / 3

if pluralismo >= 0.8:
    estado = "🟢 Alto pluralismo"
elif pluralismo >= 0.6:
    estado = "🟡 Aceptable"
elif pluralismo >= 0.4:
    estado = "🟠 Riesgo"
else:
    estado = "🔴 Crítico"

print(f"\nÍndice de Pluralismo: {pluralismo:.2f} ({estado})")

# --- Save audit log (from fZFnEv9_mNBR) ---
log = {
    "fecha": str(datetime.now()),
    "video": archivo,
    "total_segmentos": len(segmentos),
    "resultados": tiempos
}

with open("log_auditoria.json", "w") as f:
    json.dump(log, f, indent=4)

print("Audit log saved to 'log_auditoria.json'.")
