# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-01

### 🎉 Initial Release

#### Added
- **Semantische Suche**: FAISS-basierte Vektorsuche über 300.000+ Förderprojekte
- **Keyword-Suche**: Schnelle textbasierte Suche als Ergänzung zur semantischen Suche
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

##### Suche
- Semantische Suche mit cosine similarity
- Top-K Konfiguration (5-100 Treffer)
- Drei Suchmodi: Semantic, Keyword, Hybrid
- Automatische Deduplizierung von Ergebnissen
- Score-basiertes Ranking

##### LLM-Integration
- Ollama für Embeddings (nomic-embed-text)
- GROQ API für Chat-Completion
- Strukturierte Prompts mit System/User-Trennung
- Automatische Quellenangaben (FKZ)
- Statistiken: Anzahl Projekte, Gesamtfördersumme, Zeitraum

##### Datenverarbeitung
- CSV-Import mit Excel-Formatierungs-Bereinigung
- Fördersummen-Konvertierung (String → Float)
- Laufzeit-Extraktion aus Datumsspalten
- 9 Informationsquellen für Embeddings
- Robuste Fehlerbehandlung

##### UI/UX
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

### Notes
- Erste stabile Version
- Produktionsreif für lokale Nutzung
- Benötigt GROQ API Key für LLM-Funktionen

---

## [Unreleased]

### Geplant für v0.2.0
- [ ] Docker-Container für einfaches Deployment
- [ ] FastAPI REST-Endpunkt
- [ ] Multiprocessing für parallele Embedding-Erzeugung
- [ ] Export-Funktion für Suchergebnisse (CSV, Excel)
- [ ] Admin-Dashboard für Index-Verwaltung
- [ ] Caching-Layer für häufige Queries
- [ ] Integration weiterer Embedding-Modelle

---

[0.1.0]: https://github.com/dgaida/rag_foerderkatalog/releases/tag/v0.1.0
