# Release v0.1.0 - Initial Release 🎉

## 🌟 Highlights

Erste offizielle Release-Version von **RAG Förderkatalog** – einer KI-gestützten Suchmaschine für über 300.000 deutsche Forschungsförderprojekte des BMBF.

### Key Features

✨ **Semantische Suche** mit FAISS und Ollama
✨ **Hybride Suche** (Vektor + Keyword)
✨ **RAG-Pipeline** für kontextbasierte LLM-Antworten
✨ **Moderne Web-UI** mit Gradio
✨ **Inkrementelle Indizierung** für große Datasets

---

## 📦 Downloads

### Haupt-Release

| Asset | Beschreibung | Größe   |
|-------|--------------|---------|
| [rag_foerderkatalog_index_v0.1.0.zip](./rag_foerderkatalog_index_v0.1.0.zip) | Pre-indizierter FAISS-Index + Mapping | ~0.8 GB |
| [Source code (zip)](./source.zip) | Quellcode als ZIP | ~500 KB |
| [Source code (tar.gz)](./source.tar.gz) | Quellcode als Tarball | ~450 KB |

### Checksums (SHA256)

```
rag_foerderkatalog_index_v0.1.0.zip: [SHA256-Hash wird automatisch generiert]
```

---

## 🚀 Schnellstart

### Installation

```bash
# 1. Repository klonen
git clone https://github.com/dgaida/rag_foerderkatalog.git
cd rag_foerderkatalog

# 2. Environment erstellen
conda env create -f environment.yml
conda activate rag_foerderkatalog

# 3. Pre-Index herunterladen und entpacken
unzip rag_foerderkatalog_index_v0.1.0.zip -d data/

# 4. CSV-Datei ablegen
# Laden Sie foerderkatalog_export.csv vom BMBF herunter
cp /pfad/zur/foerderkatalog_export.csv input/

# 5. API Key konfigurieren
echo "GROQ_API_KEY=your_key_here" > .env

# 6. Starten
python main.py --no-embeddings
```

### Erste Schritte

Nach dem Start öffnet sich automatisch die Web-UI (Port 7860):

1. Wählen Sie einen **Suchmodus** (Hybrid empfohlen)
2. Geben Sie eine **Suchanfrage** ein
3. Klicken Sie auf **🚀 Suchen**
4. Erkunden Sie die **Ergebnisse** und **KI-Analyse**

---

## 📋 Was ist neu?

### Features

#### Suche
- ✅ Semantische Vektorsuche mit FAISS (cosine similarity)
- ✅ Keyword-basierte Textsuche
- ✅ Hybride Suche kombiniert beide Ansätze
- ✅ Konfigurierbare Top-K (5-100 Treffer)
- ✅ Score-basiertes Ranking

#### LLM-Integration
- ✅ Ollama für Embeddings (nomic-embed-text, 768 dim)
- ✅ GROQ API für Chat-Completion
- ✅ Strukturierte Prompts mit Quellenangaben
- ✅ Automatische Statistiken (Anzahl, Fördersumme, Zeitraum)

#### Datenverarbeitung
- ✅ CSV-Import mit Excel-Formatierungs-Bereinigung
- ✅ Laufzeit-Extraktion aus Datumsspalten
- ✅ 9 Informationsquellen für Embeddings:
  - Zuwendungsempfänger
  - Thema
  - Leistungsplansystematik
  - Ausführende Stelle
  - Stadt/Gemeinde
  - Bundesland
  - **Laufzeit** (neu extrahiert)
  - Förderprofil
  - Verbundprojekt

#### UI/UX
- ✅ Modernes Dark-Theme Design
- ✅ Responsive Layout
- ✅ FKZ-basierte Detail-Ansicht
- ✅ Beispiel-Queries
- ✅ Live-Statistiken
- ✅ Scrollbare Tabellen mit Sticky Headers

#### Technical
- ✅ Inkrementelle Indizierung (5000 Projekte/Batch)
- ✅ Fortschritts-Persistierung
- ✅ Umfangreiches Logging
- ✅ 80%+ Test Coverage
- ✅ CI/CD Pipeline (GitHub Actions)

### Dokumentation
- ✅ Umfassendes README
- ✅ CONTRIBUTING.md für Entwickler
- ✅ CHANGELOG.md
- ✅ Code-Kommentare und Docstrings

---

## 🛠️ Technische Details

### Systemanforderungen

- **Python**: 3.11 oder höher
- **RAM**: Mindestens 4 GB (empfohlen: 8 GB)
- **Disk**: ~3 GB für Index + CSV
- **OS**: Linux, macOS, Windows

### Abhängigkeiten

**Core:**
- pandas >= 2.0.0
- numpy >= 1.24.0
- faiss-cpu >= 1.8.0
- gradio >= 4.0.0
- ollama >= 0.3.0

**LLM:**
- [llm_client](https://github.com/dgaida/llm_client)
- python-dotenv

**Optional (Development):**
- pytest, pytest-cov
- black, isort, flake8, mypy

---

## 📊 Projekt-Statistiken

- **Zeilen Code**: ~3.500
- **Test-Module**: 3
- **Test Cases**: 45+
- **Code Coverage**: 80%+
- **Unterstützte Projekte**: 300.000+

---

## 🐛 Bekannte Probleme

### Performance
- Vollständige Indizierung dauert 2-4 Stunden
  - **Workaround**: Nutzen Sie den vorbereiteten Index
- FAISS-Index benötigt ~2 GB RAM
  - **Workaround**: Batch-Size reduzieren für kleinere Datasets

### Abhängigkeiten
- Ollama muss lokal installiert sein
  - **Geplant**: Cloud-Embeddings in v0.2.0
- CSV-Datei nicht im Repository enthalten
  - **Grund**: Lizenz-/Größenbeschränkungen

---

## 🔮 Roadmap v0.2.0

Geplante Features für das nächste Release:

- [ ] Docker-Container
- [ ] FastAPI REST-Endpunkt
- [ ] Multiprocessing für Embeddings
- [ ] Export-Funktion (CSV, Excel, JSON)
- [ ] Admin-Dashboard
- [ ] Query-Caching
- [ ] Alternative Embedding-Modelle (SentenceTransformers)
- [ ] Erweiterte Filter (Datum, Bundesland, Fördersumme)

---

## 🤝 Mitwirken

Contributions sind willkommen! Siehe [CONTRIBUTING.md](./CONTRIBUTING.md).

**Quick Links:**
- [Issues](https://github.com/dgaida/rag_foerderkatalog/issues)
- [Discussions](https://github.com/dgaida/rag_foerderkatalog/discussions)
- [Pull Requests](https://github.com/dgaida/rag_foerderkatalog/pulls)

---

## 📄 Lizenz

[MIT License](./LICENSE)

---

## 🙏 Credits

- Inspiriert von [ibaleri/Foerderprojekt](https://github.com/ibaleri/Foerderprojekt)
- BMBF für Förderdaten
- Ollama und GROQ für KI-Infrastruktur
- Alle Contributors

---

## 📞 Support

- **Bugs**: [GitHub Issues](https://github.com/dgaida/rag_foerderkatalog/issues)
- **Fragen**: [GitHub Discussions](https://github.com/dgaida/rag_foerderkatalog/discussions)

---

**Erstellt**: 2025-11-01
**Autor**: @dgaida
**Version**: 0.1.0
