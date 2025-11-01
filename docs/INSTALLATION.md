# Installation Guide - RAG Förderkatalog v0.1.0

Vollständige Installationsanleitung für verschiedene Szenarien.

---

## 📋 Voraussetzungen

### System-Anforderungen

- **Betriebssystem**: Linux, macOS oder Windows
- **Python**: 3.11 oder höher
- **RAM**: Mindestens 4 GB (empfohlen: 8 GB)
- **Disk Space**: ~3 GB für Index + CSV
- **Internet**: Für Installation der Dependencies

### Benötigte Software

1. **Python 3.11+**
   - Download: https://www.python.org/downloads/
   - Prüfen: `python --version`

2. **Ollama** (für Embeddings)
   - Download: https://ollama.ai/
   - Installation: https://ollama.ai/download

3. **Git** (für Entwickler)
   - Download: https://git-scm.com/

---

## 🚀 Schnellinstallation (Empfohlen)

Für Nutzer, die schnell starten möchten mit dem vorbereiteten Index.

### Schritt 1: Repository klonen

```bash
git clone https://github.com/dgaida/rag_foerderkatalog.git
cd rag_foerderkatalog
```

### Schritt 2: Python-Umgebung erstellen

**Option A: Mit Conda (empfohlen)**

```bash
conda env create -f environment.yml
conda activate rag_foerderkatalog
```

**Option B: Mit venv**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# ODER
.venv\Scripts\activate  # Windows

pip install -e .
```

### Schritt 3: Ollama einrichten

```bash
# Ollama starten (falls nicht bereits aktiv)
ollama serve

# In neuem Terminal: Embedding-Modell herunterladen
ollama pull nomic-embed-text
```

### Schritt 4: Pre-Index herunterladen

```bash
# Download von GitHub Release v0.1.0
wget https://github.com/dgaida/rag_foerderkatalog/releases/download/v0.1.0/rag_foerderkatalog_index_v0.1.0.zip

# Entpacken
unzip rag_foerderkatalog_index_v0.1.0.zip -d data/

# Prüfen
ls -lh data/vector.index data/embeddings_map.json
```

### Schritt 5: CSV-Datei ablegen

```bash
# Laden Sie foerderkatalog_export.csv vom BMBF herunter
# Quelle: https://foerderportal.bund.de/foekat/

# Kopieren Sie die Datei nach input/
mkdir -p input
cp /pfad/zur/foerderkatalog_export.csv input/
```

### Schritt 6: API-Key konfigurieren

```bash
# Erstellen Sie eine .env Datei
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# GROQ API Key erhalten Sie unter: https://console.groq.com/
```

### Schritt 7: Anwendung starten

```bash
python main.py --no-embeddings
```

Die Anwendung öffnet sich automatisch im Browser unter `http://localhost:7860`

---

## 🔨 Vollständige Installation (Selbst indizieren)

Für Entwickler oder wenn Sie den Index selbst erstellen möchten.

### Schritte 1-6: Wie oben

Folgen Sie den Schritten 1-6 der Schnellinstallation, aber **überspringen** Sie Schritt 4 (Pre-Index Download).

### Schritt 7: Index selbst erstellen

```bash
# Vollständige Indizierung (dauert 2-4 Stunden)
python main.py --batch-size 5000

# ODER: Nur erste 10.000 Projekte (zum Testen)
python main.py --limit 10000 --batch-size 1000
```

**Fortschritt verfolgen:**
- Live im Terminal
- Log-Datei: `logs/app_YYYYMMDD.log`
- Fortschritt: `data/indexing_progress.json`

**Bei Unterbrechung:**
Einfach erneut `python main.py --batch-size 5000` ausführen – macht da weiter, wo aufgehört wurde.

**Index neu aufbauen:**
```bash
python main.py --batch-size 5000 --force-rebuild
```

---

## 🐳 Docker Installation (Geplant für v0.2.0)

Docker-Support ist für Release v0.2.0 geplant.

---

## 🧪 Entwickler-Installation

Für Contributors und Entwickler mit allen Tools.

### Zusätzliche Dependencies installieren

```bash
# Nach Schritt 2 der Schnellinstallation
pip install -e ".[dev]"

# Oder mit Conda
conda activate rag_foerderkatalog
pip install -e ".[dev]"
```

### Pre-Commit Hooks einrichten

```bash
# Git Hooks aktivieren
chmod +x scripts/pre-commit.sh
ln -s ../../scripts/pre-commit.sh .git/hooks/pre-commit
```

### Tests ausführen

```bash
# Alle Tests
pytest

# Mit Coverage
pytest --cov=src --cov-report=html

# Nur schnelle Tests
pytest -m "not slow"
```

### Code-Qualität prüfen

```bash
# Formatierung
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/

# Type-Checking
mypy src/
```

---

## 🔧 Konfiguration

### Umgebungsvariablen (.env)

```bash
# LLM API Keys
GROQ_API_KEY=gsk_...

# Optional: Custom Ollama Host
OLLAMA_HOST=http://localhost:11434

# Optional: Custom LLM Model
LLM_DEFAULT_MODEL=llama3
```

### Config-Datei (src/config.py)

Wichtige Einstellungen können in `src/config.py` angepasst werden:

```python
# Pfade
INPUT_CSV = ROOT / "input" / "foerderkatalog_export.csv"
DATA_DIR = ROOT / "data"

# Modelle
OLLAMA_EMBED_MODEL = "nomic-embed-text"
LLM_DEFAULT_MODEL = "moonshotai/kimi-k2-instruct-0905"

# Limits
TOP_K_DEFAULT = 50
MAX_DOCS_FOR_LLM = 30
```

---

## 🧰 Fehlerbehebung

### Problem: "Ollama not found"

**Lösung:**
```bash
# Prüfen ob Ollama läuft
curl http://localhost:11434/api/tags

# Falls nicht: Ollama starten
ollama serve
```

### Problem: "Model not found"

**Lösung:**
```bash
# Modell herunterladen
ollama pull nomic-embed-text

# Verfügbare Modelle prüfen
ollama list
```

### Problem: "CSV not found"

**Lösung:**
```bash
# Prüfen ob CSV existiert
ls -lh input/foerderkatalog_export.csv

# CSV vom BMBF herunterladen
# https://foerderportal.bund.de/foekat/
```

### Problem: "GROQ API Error"

**Lösung:**
```bash
# API Key prüfen
cat .env

# Neuen Key generieren
# https://console.groq.com/

# Key testen
curl https://api.groq.com/openai/v1/models \
  -H "Authorization: Bearer $GROQ_API_KEY"
```

### Problem: "Out of Memory"

**Lösung:**
```bash
# Kleinere Batch-Size verwenden
python main.py --batch-size 1000

# Oder: Nur Teil der Daten indizieren
python main.py --limit 50000 --batch-size 5000
```

### Problem: "Dimension Mismatch"

**Lösung:**
```bash
# Index löschen und neu erstellen
rm -f data/vector.index data/embeddings_map.json data/indexing_progress.json
python main.py --batch-size 5000 --force-rebuild
```

---

## 📊 Verifikation

### Installation prüfen

```bash
# Python-Version
python --version  # Sollte 3.11+ sein

# Dependencies
pip list | grep -E "faiss|gradio|ollama|pandas"

# Ollama
ollama list | grep nomic-embed-text

# Dateien
tree -L 2 -I "__pycache__|*.pyc|.venv"
```

**Erwartete Ausgabe:**
```
.
├── data/
│   ├── vector.index (optional)
│   └── embeddings_map.json (optional)
├── input/
│   └── foerderkatalog_export.csv
├── src/
├── tests/
├── .env
└── ...
```

### Funktionalität testen

```bash
# Start der Anwendung
python main.py --no-embeddings

# In neuem Terminal: Test-Query
curl http://localhost:7860/
```

---

## 📚 Weiterführende Dokumentation

- [README.md](../README.md) - Projekt-Übersicht
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Entwickler-Guide
- [CHANGELOG.md](../CHANGELOG.md) - Versionshistorie

---

## 💡 Tipps & Best Practices

### Performance-Optimierung

1. **Nutzen Sie den Pre-Index**: Spart 2-4 Stunden Indizierungszeit
2. **SSD empfohlen**: Schnellere Index-Zugriffe
3. **RAM**: 8 GB optimal für große Datasets
4. **Batch-Size**: 5000-10000 für schnelle Systeme

### Workflow-Tipps

1. **Conda Environment**: Verhindert Dependency-Konflikte
2. **Log-Files**: `logs/` regelmäßig prüfen
3. **Backups**: `data/` Verzeichnis regelmäßig sichern
4. **Updates**: `git pull` für neueste Features

---

## 🆘 Support

**Probleme?**
- [GitHub Issues](https://github.com/dgaida/rag_foerderkatalog/issues)
- [Discussions](https://github.com/dgaida/rag_foerderkatalog/discussions)

**Dokumentation:**
- [Wiki](https://github.com/dgaida/rag_foerderkatalog/wiki)
- [FAQ](https://github.com/dgaida/rag_foerderkatalog/wiki/FAQ)

---

**Viel Erfolg bei der Installation! 🚀**
