# Release Notes v0.1.0 🎉

**Release Date**: November 1, 2025
**Version**: 0.1.0 (Initial Release)

---

## 🌟 Highlights

Dies ist die erste offizielle Release-Version von **RAG Förderkatalog** – einer KI-gestützten Suchmaschine für deutsche Forschungsförderprojekte des BMBF.

### Was ist neu?

✨ **Semantische Suche** mit FAISS und Ollama Embeddings
✨ **Hybride Suche** kombiniert Vektorsuche und Keyword-Matching
✨ **RAG-Pipeline** für kontextbasierte LLM-Antworten
✨ **Moderne Web-UI** mit Gradio
✨ **300.000+ Förderprojekte** durchsuchbar
✨ **Inkrementelle Indizierung** für große Datasets

---

## 📦 Was ist enthalten?

### Kernel-Features

1. **Drei Suchmodi**
   - **Semantisch**: KI-basierte Vektorsuche für thematische Ähnlichkeit
   - **Keyword**: Schnelle textbasierte Suche
   - **Hybrid**: Beste aus beiden Welten

2. **Intelligente Antwortgenerierung**
   - Kontextbasierte LLM-Antworten
   - Automatische Quellenangaben (FKZ)
   - Statistiken (Anzahl, Fördersumme, Zeitraum)

3. **Erweiterte Projektinformationen**
   - Laufzeit-Extraktion (z.B. "2002 - 2005")
   - Bundesland-Filter
   - Förderprofil-Kategorisierung
   - Verbundprojekt-Erkennung

4. **Benutzerfreundliche UI**
   - Modernes Dark-Theme Design
   - FKZ-basierte Detail-Ansicht
   - Beispiel-Queries zum Ausprobieren
   - Export-Möglichkeit der Ergebnisse

### Technische Features

- **Batch-Embeddings**: 5000 Projekte pro Start
- **Fortschritts-Persistierung**: Weitermachen wo aufgehört
- **Umfangreiches Logging**: Debug-Support
- **Test-Suite**: 80%+ Coverage
- **CI/CD Pipeline**: Automatische Quality Checks

---

## 🚀 Installation

### Voraussetzungen

- Python 3.11 oder höher
- Ollama installiert mit `nomic-embed-text` Modell
- GROQ API Key (für LLM-Funktionen)

### Schnellstart

```bash
# 1. Repository klonen
git clone https://github.com/dgaida/rag_foerderkatalog.git
cd rag_foerderkatalog

# 2. Umgebung erstellen
conda env create -f environment.yml
conda activate rag_foerderkatalog

# 3. Pre-indizierte Daten herunterladen (optional)
# Download: rag_foerderkatalog_index_v0.1.0.zip
unzip rag_foerderkatalog_index_v0.1.0.zip -d data/

# 4. API Key konfigurieren
echo "GROQ_API_KEY=your_api_key" > .env

# 5. Starten
python main.py --no-embeddings  # mit vorhandenem Index
# ODER
python main.py --batch-size 5000  # selbst indizieren
```

---

## 📊 Inkludierte Daten

### rag_foerderkatalog_index_v0.1.0.zip

Diese ZIP-Datei enthält:

- **`vector.index`** (ca. 300 MB, enthält bisher nur ca. 100.000 Projekte)
  FAISS-Index mit Embeddings für alle Projekte

- **`embeddings_map.json`** (ca. 2 MB)
  Mapping zwischen FAISS-IDs und CSV-Zeilen

**Hinweis**: CSV-Datei (`foerderkatalog_export.csv`) muss separat vom BMBF bezogen werden.

### Warum Pre-Indiziert?

Die Embedding-Erzeugung für 300.000+ Projekte dauert **2-4 Stunden**. Mit dem vorbereiteten Index können Sie sofort starten!

---

## 💡 Erste Schritte

### 1. Beispiel-Suche

Starten Sie die Anwendung und probieren Sie:

```
"Künstliche Intelligenz Hochschule Bayern"
"Wasserstoff Energie NRW 2020-2025"
"Quantencomputing Forschung"
```

### 2. Suchmodi verstehen

- **Semantisch**: Findet thematisch ähnliche Projekte (auch ohne exakte Keyword-Übereinstimmung)
- **Keyword**: Schnelle Textsuche, gut für spezifische Begriffe
- **Hybrid**: Kombiniert beide Ansätze für umfassende Ergebnisse

### 3. FKZ-Details erkunden

Klicken Sie in der "KI-Analyse" auf ein FKZ, um detaillierte Projektinformationen anzuzeigen.

---

## 🔧 Konfiguration

### Wichtige Dateien

- **`.env`**: API Keys
- **`src/config.py`**: Pfade und Defaults
- **`logs/`**: Application Logs
- **`data/`**: FAISS-Index und Mappings

### Performance-Tuning

```bash
# Mehr Treffer anzeigen
python main.py --no-embeddings
# In UI: k-Slider auf 50-100 erhöhen

# Eigene Indizierung mit größeren Batches
python main.py --batch-size 10000

# Debug-Modus
python main.py --log-level DEBUG
```

---

## 📚 Dokumentation

- **README.md**: Vollständige Projektdokumentation
- **CONTRIBUTING.md**: Guidelines für Entwickler
- **CHANGELOG.md**: Detaillierte Änderungshistorie

Online-Dokumentation: [GitHub Wiki](https://github.com/dgaida/rag_foerderkatalog/wiki)

---

## 🐛 Bekannte Probleme

1. **Embedding-Dauer**: Vollständige Indizierung dauert lange
   - **Lösung**: Nutzen Sie den vorbereiteten Index

2. **Speicherbedarf**: FAISS-Index benötigt ~2GB RAM
   - **Lösung**: Für große Datasets Batch-Size reduzieren

3. **Ollama-Abhängigkeit**: Muss lokal installiert sein
   - **Lösung**: Zukünftig auch Cloud-Embeddings

---

## 🤝 Mitwirken

Wir freuen uns über Contributions! Bitte lesen Sie [CONTRIBUTING.md](CONTRIBUTING.md) für Details.

### Quick Links

- [Issue Tracker](https://github.com/dgaida/rag_foerderkatalog/issues)
- [Discussions](https://github.com/dgaida/rag_foerderkatalog/discussions)
- [Pull Requests](https://github.com/dgaida/rag_foerderkatalog/pulls)

---

## 📄 Lizenz

MIT License - siehe [LICENSE](LICENSE) für Details.

---

## 🙏 Danksagungen

- Inspiriert von [ibaleri/Foerderprojekt](https://github.com/ibaleri/Foerderprojekt)
- BMBF für die Bereitstellung der Förderdaten
- Ollama und GROQ für die KI-Infrastruktur

---

## 📞 Support

- **Bugs**: [GitHub Issues](https://github.com/dgaida/rag_foerderkatalog/issues)
- **Fragen**: [GitHub Discussions](https://github.com/dgaida/rag_foerderkatalog/discussions)
- **E-Mail**: daniel.gaida@th-koeln.de

---

**Viel Erfolg mit RAG Förderkatalog! 🚀**

---

## Checksums (SHA256)

```
# Für Integritätsprüfung
rag_foerderkatalog_index_v0.1.0.zip: [wird nach Erstellung generiert]
```

Zur Verifikation:
```bash
sha256sum rag_foerderkatalog_index_v0.1.0.zip
```
