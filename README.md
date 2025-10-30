# 🧠 RAG Förderkatalog — Semantische & Hybride Suche in Forschungsförderprojekten

Eine **Retrieval-Augmented Generation (RAG)** Anwendung zur semantischen und keyword-basierten Suche in deutschen Förderprojekten.
Das Projekt nutzt **Ollama** für Embeddings, **FAISS** für Vektorsuche, **Gradio** für die Oberfläche und **GROQ / LLMClient** für Antworten aus Kontextdaten.

---

## 🚀 Features

✅ **Semantische Suche** — auf Basis von Embeddings (Ollama + FAISS)
✅ **Keyword-basierte Suche** — string- und tokenbasiert, schnelle Ergänzung zur semantischen Suche
✅ **Hybride Suche** — Kombination beider Ansätze mit gewichteter Ergebnisaggregation
✅ **Batch-Embeddings** — für große CSV-Dateien (192 MB Förderkatalog)
✅ **RAG-Pipeline** — kontextbasiertes LLM (Antworten auf Suchanfragen)
✅ **Gradio GUI** — intuitive Weboberfläche zur Suche und Ergebnisexploration
✅ **Logging & Persistenz** — automatisch im `logs/` Verzeichnis

---

## 🧱 Projektstruktur

```
rag-foerderprojekte/
├── input/
│   └── foerderkatalog_export.csv      # Förderkatalog (192 MB, vom BMBF)
├── data/
│   ├── vector.index                   # FAISS Index
│   └── embeddings_map.json            # Mapping (id -> row)
├── logs/
│   └── app_YYYYMMDD.log
├── src/
│   ├── app.py                         # Gradio UI mit hybrider Suche
│   ├── config.py                      # Zentrale Pfade & Defaults
│   ├── search/
│   │   └── engine.py                  # ProjectSearchEngine (semantische Suche)
│   ├── embeddings/
│   │   └── faiss_store.py             # FAISS-Index Management
│   ├── llm/
│   │   └── llm_wrapper.py             # LLMClient + Ollama-Integration
│   └── utils/
│       └── logging_config.py          # Zentrale Logging-Konfiguration
├── main.py                            # Einstiegspunkt (Batch Embeddings + App Start)
├── environment.yml                    # Anaconda Umgebung
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Umgebung erstellen

```bash
conda env create -f environment.yml
conda activate rag_foerderkatalog
```

### 2️⃣ Ollama & Modelle installieren

Ollama muss lokal installiert und ein Embedding-fähiges Modell vorhanden sein, z. B.:

```bash
ollama pull nomic-embed-text
```

### 3️⃣ LLMClient vorbereiten

Erstelle eine Datei `.env` oder `secrets.env` mit deinem API-Key:

```
GROQ_API_KEY=dein_api_key
```

### 4️⃣ CSV-Datei ablegen

Lege den Förderkatalog in `input/foerderkatalog_export.csv`.

---

## ▶️ Nutzung

### Start des Systems (mit Embedding-Erstellung)

```bash
python main.py --batch-size 256
```

👉 Dies lädt die CSV, erzeugt die Embeddings (in Batches) und startet die Gradio-App automatisch im Browser (`http://localhost:7860`).

### Start ohne erneute Embeddings

Falls der Index (`data/vector.index`) bereits existiert:

```bash
python main.py --no-embeddings
```

---

## 🔍 Suchmodi

| Modus        | Beschreibung                                                   |
| ------------ | -------------------------------------------------------------- |
| **Semantic** | reine semantische Suche (Vektorsuche via FAISS)                |
| **Keyword**  | reine Schlüsselwortsuche über Textfelder                       |
| **Hybrid**   | kombiniert beide (semantische Treffer + Top-5 keyword Treffer) |

---

## 💡 Beispieleingaben

| Beispiel                            | Beschreibung                                |
| ----------------------------------- | ------------------------------------------- |
| `Künstliche Intelligenz Hochschule` | Sucht Projekte zu KI im Hochschulkontext    |
| `Wasserstoff Energie NRW`           | Projekte mit Fokus auf Wasserstoff in NRW   |
| `Digitalisierung Bildung`           | Bildungsprojekte im Bereich Digitalisierung |

---

## 🧠 Funktionsweise

1. **CSV-Import & Bereinigung**
   Alle Spalten werden dekodiert und normalisiert (inkl. „Fördersumme in EUR“).

2. **Batchweise Embedding-Erstellung**
   Texte aus relevanten Spalten (`Zuwendungsempfänger`, `Thema`, `Klartext Leistungsplansystematik`) werden mit **Ollama Embeddings** in Vektoren überführt und in einem **FAISS Index** gespeichert.

3. **Semantische Suche**
   Abfragen werden eingebettet und mit Cosine Similarity gegen den Index verglichen.

4. **Keyword-Suche**
   Einfache Token- und Teilstring-Suche über ausgewählte Spalten, top-5 Ergebnisse.

5. **Hybride Aggregation**
   Ergebnisse werden zusammengeführt, priorisiert nach semantischer Relevanz, ergänzt durch keyword-Treffer.

6. **LLM-Antwort (RAG)**
   Die Top-Ergebnisse werden an das LLM (über `LLMClient`) als Kontext übergeben, um eine konsolidierte, belegte Antwort zu erzeugen.

---

## 🧾 Logging

Logs werden automatisch unter `logs/app_YYYYMMDD.log` gespeichert.
Das Logging-Level kann über `--log-level DEBUG` erhöht werden.

---

## 🧪 Entwicklungsnotizen

* Projekt getestet mit **Python 3.11**, **FAISS 1.8**, **Ollama 0.3+**, **Gradio 5+**.
* Embedding-Dauer für 192 MB CSV kann mehrere Stunden betragen (je nach Hardware).
* Für parallele Embedding-Erzeugung kann später Multiprocessing ergänzt werden.

---

## 🖼️ GUI-Vorschau

*(Screenshot-Platzhalter für GitHub-README)*

![RAG Förderkatalog UI Vorschau](docs/screenshot.png)

---

## 🙏 Dank & Attribution

Dieses Projekt entstand teilweise inspiriert durch das Repository
➡️ [https://github.com/ibaleri/Foerderprojekt](https://github.com/ibaleri/Foerderprojekt)

sowie durch die Integration von
➡️ [https://github.com/dgaida/llm_client](https://github.com/dgaida/llm_client)

Beide Projekte haben wertvolle Ideen und Schnittstellen geliefert, auf denen diese Anwendung aufbaut.

---

**© 2025 – RAG Förderkatalog (Open Source / MIT Lizenz)**
