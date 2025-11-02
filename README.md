# 🧠 RAG Förderkatalog — Semantische & Hybride Suche in Forschungsförderprojekten

Eine **Retrieval-Augmented Generation (RAG)** Anwendung zur semantischen und keyword-basierten Suche in deutschen Förderprojekten.
Das Projekt nutzt **Ollama** für Embeddings, **FAISS** für Vektorsuche, **Gradio** für die Oberfläche und **GROQ / LLMClient** für Antworten aus Kontextdaten.

---

![CI/CD Pipeline](https://github.com/dgaida/rag_foerderkatalog/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/dgaida/rag_foerderkatalog/actions/workflows/ci.yml)
[![Test Suite](https://github.com/dgaida/rag_foerderkatalog/workflows/Test%20Suite/badge.svg)](https://github.com/dgaida/rag_foerderkatalog/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/dgaida/rag_foerderkatalog/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/rag_foerderkatalog)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/rag_foerderkatalog/graphs/commit-activity)

## 🚀 Features

✅ **Semantische Suche** — auf Basis von Embeddings (Ollama + FAISS)
✅ **Keyword-basierte Suche** — string- und tokenbasiert, schnelle Ergänzung zur semantischen Suche
✅ **Hybride Suche** — Kombination beider Ansätze mit gewichteter Ergebnisaggregation
✅ **Erweiterte Embeddings** — inkl. Laufzeit-Extraktion (z.B. "2002 - 2005"), Bundesland, Förderprofil
✅ **Batch-Embeddings** — für große CSV-Dateien (192 MB Förderkatalog)
✅ **RAG-Pipeline** — kontextbasiertes LLM (Antworten auf Suchanfragen)
✅ **Gradio GUI** — intuitive Weboberfläche zur Suche und Ergebnisexploration
✅ **Logging & Persistenz** — automatisch im `logs/` Verzeichnis
✅ **Unit Tests** — umfassende Test-Suite mit pytest (>80% Coverage)

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
│   ├── app_YYYYMMDD.log              # Application Logs
│   └── prompts/                       # Gespeicherte LLM-Prompts
├── src/
│   ├── app.py                         # Gradio UI mit hybrider Suche
│   ├── config.py                      # Zentrale Pfade & Defaults
│   ├── search/
│   │   └── engine.py                  # ProjectSearchEngine
│   ├── embeddings/
│   │   └── faiss_store.py             # FAISS-Index Management
│   ├── llm/
│   │   └── llm_wrapper.py             # LLMClient + Ollama
│   └── utils/
│       └── logging_config.py          # Zentrale Logging-Konfiguration
├── tests/                              # Test-Suite
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_llm_wrapper.py
│   └── test_engine.py
├── main.py                            # Einstiegspunkt
├── environment.yml                    # Anaconda Umgebung
├── requirements-test.txt              # Test-Dependencies
├── pytest.ini                         # pytest-Konfiguration
└── README.md                          # Diese Datei
```

---

## ⚙️ Installation

### 1️⃣ Umgebung erstellen

```bash
conda env create -f environment.yml
conda activate rag_foerderkatalog

# ODER mit pip (empfohlen für vollständige Installation)
pip install -e .
```

### 2️⃣ Ollama & Modelle installieren

Ollama muss lokal installiert und ein Embedding-fähiges Modell vorhanden sein:

```bash
ollama pull nomic-embed-text
```

### 3️⃣ LLMClient vorbereiten

Erstelle eine Datei `.env` oder `secrets.env` mit deinem API-Key:

```env
GROQ_API_KEY=dein_api_key
```

### 4️⃣ CSV-Datei ablegen

Lege den Förderkatalog in `input/foerderkatalog_export.csv`.

---

## 🤗 Embedding-Provider

Das Projekt unterstützt zwei Embedding-Provider, die parallel genutzt werden können:

### 1. Ollama (Default)

```bash
# Standard-Installation wie bisher
python main.py --batch-size 5000
```

- **Modell:** nomic-embed-text (768 Dimensionen)
- **Index:** `data/vector.index`
- **Vorteil:** Keine zusätzlichen Python-Dependencies

### 2. HuggingFace (Neu) 🆕

```bash
# HuggingFace-Support installieren
pip install -r requirements_huggingface.txt

# Mit HuggingFace-Embeddings indizieren
python main.py --provider huggingface --batch-size 5000
```

- **Standard-Modell:** sentence-transformers/all-mpnet-base-v2 (768 Dimensionen)
- **Index:** `data/vector_hf.index`
- **Vorteil:** Mehr Modellauswahl, kein Ollama nötig

#### Verfügbare HuggingFace-Modelle

| Modell | Dimension | Geschwindigkeit | Qualität |
|--------|-----------|-----------------|----------|
| `intfloat/e5-small-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | ⭐⭐⭐ |
| `intfloat/e5-base-v2` | 768 | ⚡⚡ | ⭐⭐⭐⭐ |
| `sentence-transformers/all-mpnet-base-v2` | 768 | ⚡⚡ | ⭐⭐⭐⭐⭐ |

```bash
# Spezifisches Modell verwenden
python main.py \
  --provider huggingface \
  --embed-model "sentence-transformers/all-mpnet-base-v2" \
  --batch-size 5000
```

### Provider wechseln

```bash
# Mit Ollama suchen
python main.py --provider ollama --no-embeddings

# Mit HuggingFace suchen
python main.py --provider huggingface --no-embeddings
```

### Index-Übersicht anzeigen

```bash
# Zeigt Info zu allen verfügbaren Indizes
python main.py --index-info
```

**Detaillierte Dokumentation:** Siehe [README_HUGGINGFACE.md](docs/README_HUGGINGFACE.md)

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

### Debug-Modus (limitierte Anzahl Projekte)

Für schnelles Testing nur die ersten 1000 Projekte indizieren:

```bash
python main.py --limit 1000 --log-level DEBUG
```

---

## 🧪 Tests ausführen

### Alle Tests

```bash
pytest
```

### Mit Coverage-Report

```bash
pytest --cov=src --cov-report=html
```

HTML-Report öffnen: `htmlcov/index.html`

### Einzelne Test-Datei

```bash
pytest tests/test_engine.py -v
```

### Nur Unit-Tests (schnell)

```bash
pytest -m unit
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
| `Digitalisierung Bildung 2020-2025` | Bildungsprojekte mit Zeitraum               |
| `Quantencomputer Bayern`            | Regionale Suche nach Quantencomputing       |

---

## 🧠 Funktionsweise

### 1. CSV-Import & Bereinigung
Alle Spalten werden dekodiert und normalisiert:
- Entfernung von Excel-Formatierung (`="`, `"`)
- Konvertierung der Fördersummen (`.` → `""`, `,` → `.`)
- Extraktion von Start-/Endjahr und Erstellung der Laufzeit-Spalte

### 2. Erweiterte Embedding-Erstellung
Texte aus relevanten Spalten werden mit **Ollama Embeddings** in Vektoren überführt:
- `Zuwendungsempfänger`
- `Thema`
- `Klartext Leistungsplansystematik`
- `Ausführende Stelle` ✨
- `Stadt/Gemeinde` ✨
- `Bundesland` ✨
- `__laufzeit` (z.B. "2002 - 2005") ✨
- `Förderprofil` ✨
- `Verbundprojekt` ✨

### 3. Semantische Suche
Abfragen werden eingebettet und mit Cosine Similarity gegen den FAISS-Index verglichen.

### 4. Keyword-Suche
Einfache Token- und Teilstring-Suche über ausgewählte Spalten, top-5 Ergebnisse.

### 5. Hybride Aggregation
Ergebnisse werden zusammengeführt, priorisiert nach semantischer Relevanz, ergänzt durch keyword-Treffer.

### 6. LLM-Antwort (RAG)
Die Top-Ergebnisse werden an das LLM übergeben mit:
- **System-Prompt**: Spezialisiert auf BMBF-Förderung
- **Strukturiertem User-Prompt**: Mit Kontext und klaren Ausgabeanweisungen
- **Quellenangaben**: FKZ-Nummern zur Nachvollziehbarkeit

---

## 📊 Embedding-Qualität

### Inhalte
- 9 Informationsquellen inkl. Laufzeit
- Regionale Suche (Stadt, Bundesland)
- Zeitraumbasierte Queries möglich ("2020-2025")
- Verbundprojekt-Erkennung

### Beispiel Embedding-Text (neu)
```
Hochschulrektorenkonferenz. Aufbau des Bulgarisch-Rumänischen Interuniversitären Europazentrums. Wissenschaftliche Zusammenarbeit. Hochschulrektorenkonferenz. Bonn. Nordrhein-Westfalen. 2002 - 2003. Technologie- und Innovationsförderung.
```

---

## 🧾 Logging

Logs werden automatisch unter `logs/app_YYYYMMDD.log` gespeichert.
LLM-Prompts werden zusätzlich in `logs/prompts/prompt-TIMESTAMP.md` abgelegt.

Log-Level kann über `--log-level DEBUG` erhöht werden.

---

## 🧪 Entwicklungsnotizen

* Projekt getestet mit **Python 3.11**, **FAISS 1.8**, **Ollama 0.3+**, **Gradio 5+**
* Embedding-Dauer für 192 MB CSV: ca. 2-4 Stunden (je nach Hardware)
* Test-Coverage-Ziel: >80%
* Type-Checking: `mypy src/` (empfohlen)
* Code-Formatierung: `black src/` und `isort src/`

---

## 🛠️ Development Workflow

### Code-Qualität prüfen

```bash
# Formatierung
black src/
isort src/

# Type-Checking
mypy src/

# Linting
flake8 src/

# Tests mit Coverage
pytest --cov=src --cov-report=term-missing
```

### Pre-Commit Hook einrichten

```bash
# .git/hooks/pre-commit
#!/bin/bash
black src/ tests/
isort src/ tests/
pytest tests/ -v
```

---

## 🙏 Dank & Attribution

Dieses Projekt entstand inspiriert durch:
- ➡️ [ibaleri/Foerderprojekt](https://github.com/ibaleri/Foerderprojekt)

Das Projekt hat wertvolle Ideen und Schnittstellen geliefert, auf denen diese Anwendung aufbaut.

---

## 📋 TODO / Roadmap

- [ ] Multiprocessing für parallele Embedding-Erzeugung
- [ ] Docker-Container für einfaches Deployment
- [ ] API-Endpunkt (FastAPI) zusätzlich zur Gradio-UI
- [ ] Integration weiterer Embedding-Modelle (SentenceTransformers)
- [ ] Caching-Layer für häufige Queries
- [ ] Admin-Dashboard für Index-Verwaltung
- [ ] Export-Funktion für Suchergebnisse (CSV, Excel)
- [ ] A/B-Testing verschiedener Prompt-Varianten

---

## 📄 Lizenz

**MIT License** — siehe [LICENSE](LICENSE)

---

**© 2025 – RAG Förderkatalog**
