# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2025-11-03

### 🎉 Complete Release

Vollständiges Paket mit CSV und Index für vereinfachten Setup.

#### Added
- **Complete Release Package**: ZIP enthält jetzt `input/` und `data/` Ordner
  - `input/foerderkatalog_export.csv` (~200 MB)
  - `data/vector.index` (~800 MB)
  - `data/embeddings_map.json` (~3 MB)
  - `data/vector_hf.index` (~80 MB)
  - `data/embeddings_map_hf.json` (~1 MB)
- **Vereinfachter Colab-Workflow**: Nur ein Download statt mehrere
- **Live-Logging Tab**: Debug-UI mit Echtzeit-Log-Anzeige (optional)
- **Auto-Refresh**: Logs können automatisch aktualisiert werden
- **Verbesserte Validierung**: Prüft alle Dateien vor dem Start

#### Changed
- **Colab-Notebook**: Angepasst für Complete Release v0.3.0
- **README**: Aktualisierte Installationsanleitung

### Notes
- **Empfehlung**: Verwenden Sie v0.3.0 für neue Installationen
- **Kompatibilität**: Alte v0.2.0 Releases funktionieren weiterhin

---

## [0.2.0] - 2025-11-02

### 🎉 Major Release - Cloud Ready

#### Added

**🚀 Google Colab Support**
- Vollständiges Jupyter Notebook für Google Colab
- Automatischer Download des vorbereiteten Index
- Schritt-für-Schritt-Anleitung für Cloud-Nutzung
- GROQ API Key Management im Notebook
- Share-Link-Funktion für öffentlichen Zugriff

**🤖 HuggingFace Embedding Provider**
- Neue Provider-Architektur neben Ollama
- Support für 4+ HuggingFace-Modelle:
  - `intfloat/e5-small-v2` (384 dim, schnell)
  - `sentence-transformers/all-MiniLM-L6-v2` (384 dim)
  - `intfloat/e5-base-v2` (768 dim, bessere Qualität)
  - `sentence-transformers/all-mpnet-base-v2` (768 dim, beste Qualität)
- Automatisches Modell-Caching
- Lazy-Loading für optimalen Speicher-Verbrauch

**🔍 Index-Validierung & Management**
- Automatische Index-Vollständigkeitsprüfung beim Start
- Neue `IndexValidator`-Klasse
- Identifikation fehlender Projekte
- Erkennung verwaister Einträge
- Statistik-Report (Synchronisation, Abdeckung)
- Inkrementelle Nachindizierung nur für fehlende Einträge

**📋 Erweiterte CLI-Optionen**
- `--provider [ollama|huggingface]` - Provider-Auswahl
- `--embed-model MODEL` - Spezifisches Embedding-Modell
- `--index-info` - Übersicht über alle Indizes
- `--validate-only` - Nur Index-Validierung ohne Start
- `--show-missing` - Details zu fehlenden Projekten
- `--force-rebuild` - Index komplett neu aufbauen

**📚 Dokumentation**
- `notebooks/RAG_Foerderkatalog_Colab.ipynb` - Google Colab Notebook
- `docs/README_HUGGINGFACE.md` - HuggingFace Guide
- `docs/INDEX_VALIDATION.md` - Index-Validierung
- `requirements_huggingface.txt` - HuggingFace Dependencies
- `examples/compare_providers.py` - Provider-Vergleichs-Script

**🧪 Tests**
- `tests/test_index_validator.py` - Vollständige Test-Suite für Index-Validierung
- Erweiterte Tests für Multi-Provider-Support
- Mock-basierte Tests für HuggingFace-Provider
- Test-Coverage: 82%+ (+ 2% gegenüber v0.1.0)

#### Changed

**🏗️ Architektur**
- Provider-spezifische Index-Dateien:
  - Ollama: `vector.index`, `embeddings_map.json`
  - HuggingFace: `vector_hf.index`, `embeddings_map_hf.json`
- Neue `get_index_files()` Funktion in `src/config.py`
- `EmbeddingProvider` Type Literal ("ollama" | "huggingface")
- Erweiterte `ProjectSearchEngine` mit Provider-Parameter

**⚡ Performance**
- Optimierte Embedding-Erzeugung mit Batch-Verarbeitung
- Intelligentere Index-Validierung (nur bei Bedarf)
- Lazy-Loading für HuggingFace-Modelle
- Fortschritts-Persistierung für beide Provider

**🎨 UI/UX**
- Provider-Anzeige in Gradio-Oberfläche
- Index-Informationen im Footer
- Verbesserte Fehler-Meldungen
- Detailliertere Logging-Ausgaben

**📝 Code-Qualität**
- Type-Hints für alle neuen Funktionen
- Docstrings im Google-Style
- Einheitliche Fehlerbehandlung
- Verbesserte Modularität

#### Fixed

**🐛 Bugfixes**
- Dimension Mismatch bei Provider-Wechsel wird verhindert
- Leere Embeddings werden erkannt und übersprungen
- Progress-Speicherung funktioniert zuverlässig
- CSV-Encoding-Fehler behoben

**🔧 Stability**
- Robustere Fehlerbehandlung bei API-Fehlern
- Bessere Validierung von Input-Daten
- Graceful Degradation bei fehlenden Modellen
- Timeout-Handling für Downloads

#### Deprecated

- ⚠️ Direkte Verwendung von `FAISS_INDEX_FILE` ohne Provider (wird in v0.3.0 entfernt)
- ⚠️ Alte Progress-Datei `indexing_progress.json` ohne Provider-Suffix

#### Removed

- ❌ Legacy-Code für alte Embedding-Formate

#### Security

- 🔒 Validierung von User-Input in CLI
- 🔒 Sichere Speicherung von API-Keys
- 🔒 Keine Secrets in Logs

---

## [0.1.0] - 2025-11-01

### 🎉 Initial Release

#### Added
- **Semantische Suche**: FAISS-basierte Vektorsuche über 300.000+ Förderprojekte
- **Keyword-Suche**: Schnelle textbasierte Suche als Ergänzung
- **Hybride Suche**: Intelligente Kombination aus semantischer und Keyword-Suche
- **RAG-Pipeline**: Kontextbasierte LLM-Antworten mit Quellenangaben (FKZ)
- **Gradio Web-UI**: Moderne, benutzerfreundliche Weboberfläche
- **Erweiterte Embeddings**:
  - Laufzeit-Extraktion (Start-/Endjahr)
  - Bundesland-basierte Suche
  - Förderprofil-Integration
  - Verbundprojekt-Erkennung
- **Inkrementelle Indizierung**: Batch-weise Embedding-Erzeugung (5000 Projekte pro Start)
- **FKZ-Detail-Ansicht**: Klickbare Projekt-Details in der UI
- **Logging & Persistenz**:
  - Automatisches Logging in `logs/`
  - Prompt-Speicherung für Debugging
  - Fortschrittsspeicherung für Indizierung
- **CI/CD Pipeline**:
  - Automatische Tests mit pytest
  - Code-Qualitätsprüfung (Black, isort, Flake8, mypy)
  - Multi-OS Testing (Ubuntu, macOS, Windows)
  - Coverage-Reporting mit Codecov

#### Features im Detail

**Suche:**
- Semantische Suche mit cosine similarity
- Top-K Konfiguration (5-100 Treffer)
- Drei Suchmodi: Semantic, Keyword, Hybrid
- Automatische Deduplizierung von Ergebnissen
- Score-basiertes Ranking

**LLM-Integration:**
- Ollama für Embeddings (nomic-embed-text)
- GROQ API für Chat-Completion
- Strukturierte Prompts mit System/User-Trennung
- Automatische Quellenangaben (FKZ)
- Statistiken: Anzahl Projekte, Gesamtfördersumme, Zeitraum

**Datenverarbeitung:**
- CSV-Import mit Excel-Formatierungs-Bereinigung
- Fördersummen-Konvertierung (String → Float)
- Laufzeit-Extraktion aus Datumsspalten
- 9 Informationsquellen für Embeddings
- Robuste Fehlerbehandlung

**UI/UX:**
- Modernes Dark-Theme Design
- Responsive Layout
- Tabellen mit Sticky Headers
- Scrollbare Ergebnislisten
- Beispiel-Queries
- Live-Statistiken
- FKZ-basierte Detail-Ansicht

#### Technical Stack
- **Python**: 3.11+
- **Vector Store**: FAISS (CPU)
- **Embeddings**: Ollama (nomic-embed-text)
- **LLM**: GROQ/LLMClient
- **UI**: Gradio 4.0+
- **Data Processing**: Pandas, NumPy
- **Testing**: pytest, pytest-cov, pytest-mock

#### Documentation
- Umfassendes README mit Installationsanleitung
- CONTRIBUTING.md mit Entwickler-Guidelines
- Code-Kommentare und Docstrings
- Beispiele und Use Cases

#### Testing
- 80%+ Code Coverage
- Unit-Tests für alle Kernmodule
- Mock-basierte Tests für externe APIs
- CI/CD Integration

### Known Issues
- Embedding-Erzeugung für große Datasets dauert 2-4 Stunden
- FAISS-Index benötigt ~2GB RAM bei vollen 300k Projekten
- Ollama muss lokal installiert sein

---

## [Unreleased]

### Geplant für v0.3.0

**Features:**
- [ ] **FastAPI REST-Endpunkt** für programmatischen Zugriff
- [ ] **Export-Funktion** für Suchergebnisse (CSV, Excel, JSON)
- [ ] **Multiprocessing** für parallele Embedding-Erzeugung
- [ ] **Docker-Container** für einfaches Deployment
- [ ] **Admin-Dashboard** für Index-Verwaltung
- [ ] **Query-Caching** für häufige Suchen
- [ ] **Erweiterte Filter** (Datum, Bundesland, Fördersumme)
- [ ] **Weitere Embedding-Modelle** (OpenAI, Cohere)

**Verbesserungen:**
- [ ] Bessere Performance für große Batch-Größen
- [ ] Optimierte Memory-Usage
- [ ] Progressive Web App (PWA) Support
- [ ] Mehrsprachige Oberfläche (EN/DE)

**Dokumentation:**
- [ ] Video-Tutorials
- [ ] API-Dokumentation (Sphinx)
- [ ] Best Practices Guide
- [ ] Deployment-Szenarien

---

## Migration Guides

### v0.1.0 → v0.2.0

**Schritt 1: Repository aktualisieren**

```bash
git pull origin master
git checkout v0.2.0
```

**Schritt 2: Dependencies aktualisieren**

```bash
# Basis-Update
pip install --upgrade -e .

# Für HuggingFace-Support
pip install -r requirements_huggingface.txt
```

**Schritt 3: Index-Struktur migrieren**

Ihr bestehender Ollama-Index bleibt erhalten:

```bash
# Prüfen Sie Ihren Index
python main.py --index-info

# Weiter mit Ollama
python main.py --no-embeddings

# Oder: Neuen HuggingFace-Index erstellen
python main.py --provider huggingface --batch-size 5000
```

**Schritt 4: Konfiguration anpassen**

Falls Sie `src/config.py` modifiziert haben:

- Verwenden Sie `get_index_files(provider)` für Pfade
- Aktualisieren Sie Imports: `from src.config import EmbeddingProvider`

**Keine Breaking Changes** für Standard-Nutzer! 🎉

---

## Versioning

- **Major** (X.0.0): Breaking Changes, neue Architektur
- **Minor** (0.X.0): Neue Features, abwärtskompatibel
- **Patch** (0.0.X): Bugfixes, kleine Verbesserungen

---

[0.2.0]: https://github.com/dgaida/rag_foerderkatalog/releases/tag/v0.2.0
[0.1.0]: https://github.com/dgaida/rag_foerderkatalog/releases/tag/v0.1.0
