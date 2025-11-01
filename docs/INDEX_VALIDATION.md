# Index-Validierung und Synchronisation

Automatische Prüfung und Vervollständigung des FAISS-Index beim Start.

---

## 📋 Übersicht

Die Index-Validierung stellt sicher, dass alle Projekte aus der CSV-Datei im FAISS-Index vorhanden sind. Dies ist besonders wichtig, wenn:

- Eine neue CSV-Datei heruntergeladen wird
- Der Index aus einem Backup wiederhergestellt wurde
- Die Indizierung unterbrochen wurde
- Neue Projekte zur Datenbank hinzugefügt wurden

---

## 🎯 Funktionsweise

### Automatische Validierung beim Start

Beim Starten von `main.py` wird automatisch geprüft:

1. ✅ **Vollständigkeit**: Sind alle CSV-Einträge indiziert?
2. ⚠️ **Fehlende Einträge**: Welche Projekte fehlen?
3. 🔍 **Verwaiste Einträge**: Gibt es Index-Einträge ohne CSV-Zeile?
4. 📊 **Statistiken**: Synchronisationsgrad, Anzahl Einträge

### Intelligente Indizierung

Das System indiziert **nur die fehlenden Projekte**, nicht den gesamten Datensatz erneut.

```bash
# Beispiel: 200.000 Projekte in CSV, 150.000 im Index
# → Nur 50.000 neue Projekte werden indiziert
```

---

## 🚀 Verwendung

### Standard-Start (mit automatischer Validierung)

```bash
python main.py --batch-size 5000
```

**Ausgabe:**
```
🔍 Prüfe Index-Vollständigkeit...

═══════════════════════════════════════════════════════════
  Index-Validierungsbericht
═══════════════════════════════════════════════════════════

📊 Statistiken:
   CSV-Einträge gesamt:  234567
   Indizierte Einträge:  200000
   Synchronisation:      85.3%

⚠️  Fehlende Einträge:   34567
   Erste fehlende IDs: [200000, 200001, 200002, ...]

═══════════════════════════════════════════════════════════

🔄 Starte Indizierung der fehlenden Projekte...
📊 Indiziere 5000 von 34567 fehlenden Projekten (14.5%)
...
```

### Nur Validierung (ohne Indizierung)

```bash
python main.py --validate-only
```

Zeigt nur den Validierungsbericht an, startet keine Indizierung oder App.

### Details zu fehlenden Projekten anzeigen

```bash
python main.py --show-missing --validate-only
```

**Ausgabe:**
```
📋 34567 neue Projekte gefunden:

  • FKZ 13BDB60030
    Universität Musterstadt
    Forschung zu Künstlicher Intelligenz in der Medizin

  • FKZ 01AB12345
    Institut für Technologie
    Entwicklung nachhaltiger Energiesysteme

  ... und 34565 weitere Projekte
```

### Vollständige Neu-Indizierung

```bash
python main.py --force-rebuild --batch-size 10000
```

Löscht den bestehenden Index und baut ihn komplett neu auf.

---

## 📊 Validierungsreport

### Beispiel: Vollständiger Index

```
═══════════════════════════════════════════════════════════
  Index-Validierungsbericht
═══════════════════════════════════════════════════════════

📊 Statistiken:
   CSV-Einträge gesamt:  234567
   Indizierte Einträge:  234567
   Synchronisation:      100.0%

✅ Index ist vollständig synchronisiert!

═══════════════════════════════════════════════════════════
```

### Beispiel: Teilweise indiziert

```
═══════════════════════════════════════════════════════════
  Index-Validierungsbericht
═══════════════════════════════════════════════════════════

📊 Statistiken:
   CSV-Einträge gesamt:  234567
   Indizierte Einträge:  180000
   Synchronisation:      76.7%

⚠️  Fehlende Einträge:   54567
   Erste fehlende IDs: [180000, 180001, 180002, ...]

═══════════════════════════════════════════════════════════
```

### Beispiel: Verwaiste Einträge

```
═══════════════════════════════════════════════════════════
  Index-Validierungsbericht
═══════════════════════════════════════════════════════════

📊 Statistiken:
   CSV-Einträge gesamt:  230000
   Indizierte Einträge:  234567
   Synchronisation:      98.0%

⚠️  Fehlende Einträge:   4567

⚠️  Verwaiste Einträge:  4567
   Diese IDs sind im Index, aber nicht in CSV:
   Erste verwaiste IDs: [230000, 230001, 230002, ...]

═══════════════════════════════════════════════════════════
```

**Hinweis**: Verwaiste Einträge entstehen, wenn die CSV aktualisiert wurde und alte Projekte entfernt wurden. Diese beeinträchtigen die Suche nicht negativ.

---

## 🔧 Programmatische Nutzung

### In Python-Code verwenden

```python
from src.utils.index_validator import IndexValidator, check_index_completeness
from src.search.engine import ProjectSearchEngine

# Engine initialisieren
engine = ProjectSearchEngine()
engine.load_and_clean()

# Variante 1: Einfache Prüfung
is_complete, missing_count = check_index_completeness(
    engine.faiss,
    engine.df
)

if not is_complete:
    print(f"{missing_count} Projekte fehlen")

# Variante 2: Detaillierte Validierung
validator = IndexValidator(engine.faiss, engine.df)
is_valid, stats = validator.validate_index()

print(f"Synchronisation: {stats['sync_percentage']:.1f}%")
print(f"Fehlend: {stats['missing_count']}")
print(f"Verwaist: {stats['orphaned_count']}")

# Fehlende Projekte als DataFrame
missing_df = validator.get_missing_projects(limit=100)
print(missing_df[['="FKZ"', '="Thema"']])
```

### Fehlende IDs ermitteln

```python
from src.utils.index_validator import IndexValidator

validator = IndexValidator(engine.faiss, engine.df)

# Liste aller fehlenden IDs
missing_ids = validator.get_missing_indices()
print(f"Fehlende IDs: {missing_ids[:10]}...")  # Erste 10

# Verwaiste IDs
orphaned_ids = validator.get_orphaned_indices()
if orphaned_ids:
    print(f"Warnung: {len(orphaned_ids)} verwaiste Einträge")
```

---

## 🎯 Use Cases

### Use Case 1: Neue CSV-Datei

**Situation**: Sie haben eine neue CSV vom BMBF heruntergeladen.

**Lösung**:
```bash
# 1. Neue CSV in input/ ablegen
cp neue_foerderkatalog.csv input/foerderkatalog_export.csv

# 2. App starten (automatische Validierung)
python main.py --batch-size 5000

# → Nur neue Projekte werden indiziert
```

### Use Case 2: Unterbrochene Indizierung

**Situation**: Indizierung wurde unterbrochen (Ctrl+C, Crash, etc.).

**Lösung**:
```bash
# Einfach erneut starten - macht da weiter wo aufgehört
python main.py --batch-size 5000

# → Validierung erkennt fehlende Einträge automatisch
```

### Use Case 3: Index aus Backup

**Situation**: Sie haben einen alten Index aus einem Backup wiederhergestellt.

**Lösung**:
```bash
# 1. Prüfen was fehlt
python main.py --validate-only --show-missing

# 2. Fehlende Einträge indizieren
python main.py --batch-size 10000

# → Nur die fehlenden Projekte werden nachindiziert
```

### Use Case 4: Neue Projekte in CSV

**Situation**: BMBF hat neue Projekte zur Datenbank hinzugefügt.

**Lösung**:
```bash
# 1. Aktualisierte CSV herunterladen
# 2. Normal starten
python main.py --batch-size 5000

# → Automatische Erkennung und Indizierung neuer Projekte
```

---

## ⚙️ Konfiguration

### Batch-Size anpassen

```bash
# Schnelle Systeme: Größere Batches
python main.py --batch-size 10000

# Langsame Systeme: Kleinere Batches
python main.py --batch-size 1000
```

### Log-Level erhöhen

```bash
# Detaillierte Debug-Informationen
python main.py --log-level DEBUG --show-missing
```

---

## 🔍 Fehlersuche

### Problem: "Alle Projekte fehlen im Index"

**Ursache**: Index-Datei fehlt oder ist korrupt.

**Lösung**:
```bash
# Index neu aufbauen
python main.py --force-rebuild --batch-size 5000
```

### Problem: "Viele verwaiste Einträge"

**Ursache**: CSV wurde stark aktualisiert, alte Projekte entfernt.

**Lösung**: Verwaiste Einträge sind harmlos, aber für sauberen Index:
```bash
# Index komplett neu aufbauen
python main.py --force-rebuild --batch-size 5000
```

### Problem: "Dimension Mismatch"

**Ursache**: Embedding-Modell wurde geändert.

**Lösung**:
```bash
# Index mit neuem Modell neu erstellen
rm data/vector.index data/embeddings_map.json
python main.py --force-rebuild --batch-size 5000
```

---

## 📈 Performance

### Geschwindigkeit

- **Validierung**: < 1 Sekunde (auch bei 200k Projekten)
- **Indizierung**: ~2-4 Stunden für volle 200k Projekte
  - 5.000 Projekte ≈ 30-45 Minuten
  - 10.000 Projekte ≈ 60-90 Minuten

### Speicherbedarf

- **CSV**: ~200 MB
- **Index**: ~1.8 GB
- **RAM**: 4-8 GB empfohlen

---

## 🧪 Testing

### Unit-Tests ausführen

```bash
# Nur Index-Validator Tests
pytest tests/test_index_validator.py -v

# Mit Coverage
pytest tests/test_index_validator.py --cov=src.utils.index_validator
```

### Manueller Test

```python
# In Python-Shell
from src.utils.index_validator import IndexValidator
from src.search.engine import ProjectSearchEngine

engine = ProjectSearchEngine()
engine.load_and_clean()

validator = IndexValidator(engine.faiss, engine.df)
validator.log_validation_report()
```

---

## 📚 API-Referenz

### IndexValidator

**Klasse**: `src.utils.index_validator.IndexValidator`

**Methoden**:
- `get_indexed_ids()` → Set[int]
- `get_csv_ids()` → Set[int]
- `get_missing_indices()` → List[int]
- `get_orphaned_indices()` → List[int]
- `validate_index()` → Tuple[bool, dict]
- `log_validation_report()` → None
- `get_missing_projects(limit)` → pd.DataFrame

### Funktionen

**check_index_completeness**(faiss, df) → Tuple[bool, int]
- Schnelle Prüfung der Index-Vollständigkeit

**get_new_projects_summary**(validator) → str
- Erstellt lesbare Zusammenfassung neuer Projekte

---

## 🔮 Zukünftige Features

- [ ] Automatische Index-Optimierung
- [ ] Parallel-Indizierung mehrerer Batches
- [ ] Web-UI für Index-Management
- [ ] Export-Report als PDF/HTML
- [ ] Scheduler für automatische Checks

---

## 📞 Support

**Probleme mit der Validierung?**
- [GitHub Issues](https://github.com/dgaida/rag_foerderkatalog/issues)
- Label: `validation` oder `indexing`

**Dokumentation:**
- [README.md](../README.md)
- [INSTALLATION.md](./INSTALLATION.md)
