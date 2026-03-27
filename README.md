# 🐦 CodeCanary – Stimmungsmonitoring durch Sprachanalyse

CodeCanary ist ein KI-gestütztes System zur täglichen Stimmungsüberwachung. Patienten laden täglich eine kurze Sprachaufnahme hoch; das System extrahiert daraus akustische Merkmale und erkennt auf Basis statistischer Zeitreihenanalyse (CUSUM) Anzeichen von Depressionen oder Manien – vollautomatisch, ohne manuelle Eingaben.

---

## Inhaltsverzeichnis

- [Features](#features)
- [Systemübersicht](#systemübersicht)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Starten der Anwendung](#starten-der-anwendung)
- [Nutzung](#nutzung)
- [Technische Architektur](#technische-architektur)
  - [Verarbeitungs-Pipeline](#verarbeitungs-pipeline)
  - [Stimmungspunkte (Mood Score)](#stimmungspunkte-mood-score)
  - [Unterstützte Audioformate](#unterstützte-audioformate)
- [Datenbankschema](#datenbankschema)
- [Projektstruktur](#projektstruktur)
- [Bekannte Einschränkungen](#bekannte-einschränkungen)

---

## Features

- 🎤 **Browserbasierte Aufnahme** – Sprachaufnahme direkt im Browser oder Upload einer Audiodatei
- 📊 **15 akustische Merkmale** – Grundfrequenz, Jitter, Shimmer, HNR, Sprechrate, Pausen, Energie, Spektralmerkmale und MFCCs
- 📅 **Persönliche Baseline** – Die ersten drei Aufnahmetage dienen als individuelle Referenz
- 🔍 **CUSUM-Erkennung** – Sensitive Erkennung von Zustandsveränderungen über Zeit (Depressions- und Maniesignale)
- 📈 **Verlaufsansicht** – Historische Scores und Klassifikationen auf einen Blick
- ⚡ **Echtzeit-Fortschritt** – Live-Fortschrittsanzeige während der Analyse via Server-Sent Events
- 🇩🇪 **Deutsche Oberfläche** – Komplett deutschsprachige Benutzeroberfläche

---

## Systemübersicht

```
Sprachaufnahme (Browser / Datei-Upload)
          │
          ▼
  ┌───────────────────┐
  │  Audio-Vorver-    │  Rauschreduktion, SNR-Schätzung,
  │  arbeitung        │  Qualitätsbewertung (clean / degraded / reject)
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Feature-         │  15 akustische Merkmale (parallelisiert)
  │  Extraktion       │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Z-Score &        │  Normalisierung gegen populationsbasierte
  │  State-Score      │  und persönliche Referenz
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  CUSUM-Analyse    │  Bidirektionale Trenderkennung
  │                   │  (Depression ↓ / Manie ↑)
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Fusion &         │  Hysterese: Zustandswechsel benötigt
  │  Klassifikation   │  ≥ 2 von 3 aufeinanderfolgenden Tagen
  └────────┬──────────┘
           │
           ▼
  Mood Score (−100 … +100) + Zustandsklasse + Konfidenz
```

---

## Voraussetzungen

- Python **3.10** oder neuer
- pip
- Mikrozugang im Browser (für direkte Aufnahmen)

---

## Installation

```bash
# 1. Repository klonen
git clone https://github.com/jonny-fr/SystemOfAPawn-CodeCanary.git
cd SystemOfAPawn-CodeCanary

# 2. Virtuelle Umgebung erstellen (empfohlen)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Abhängigkeiten installieren
pip install -r requirements.txt
```

---

## Starten der Anwendung

```bash
python app.py
```

Die Anwendung ist anschließend unter [http://localhost:5000](http://localhost:5000) erreichbar.

---

## Nutzung

| Schritt | Beschreibung |
|---------|--------------|
| 1 | Öffne [http://localhost:5000](http://localhost:5000) im Browser |
| 2 | Nimm eine kurze Sprachprobe auf (mind. 8 Sekunden) oder lade eine Audiodatei hoch |
| 3 | Klicke auf **Analysieren** – ein Fortschrittsbalken zeigt den Verlauf |
| 4 | Das Ergebnis (Score, Zustand, Konfidenz, alle Merkmale) wird angezeigt |
| 5 | Unter **Verlauf** sind alle bisherigen Tage mit ihren Scores einsehbar |

**Phasen:**
- **Tage 1–3:** Baseline-Phase (keine Klassifikation, reine Referenzdatenerfassung)
- **Ab Tag 4:** Vollständige Klassifikation mit persönlicher und populationsbasierter Referenz

---

## Technische Architektur

### Verarbeitungs-Pipeline

Die Pipeline (`classifier/pipeline.py`, ~1 200 Zeilen) besteht aus acht Modulen:

1. **Audio-Vorverarbeitung** – Dekodierung via PyAV (WAV, WebM, MP4, OGG), spektrale Rauschsubtraktion, SNR-Schätzung
2. **Feature-Extraktion** – 15 Merkmale, parallelisiert mit `ThreadPoolExecutor`
3. **Neutral-Referenz** – Robuster Median+MAD aus Kontrollprobanden (Patient D & E)
4. **Z-Score & State-Score** – Gewichtete Normalisierung; höchste Gewichte für `speech_rate` und `pause_ratio` (je 1.8)
5. **Score-Glättung** – 3-Tage-Medianfilter gegen Einzeltag-Artefakte
6. **CUSUM-Analyse** – Vier Kanäle (dep_up, dep_down, man_up, man_down); konservative Parameter (k=0.3, h=4.0)
7. **Fusion & Hysterese** – Kombination aus absolutem Schwellenwert und CUSUM-Richtung
8. **Einzel-Tag-Analyzer** (`classifier/analyzer.py`) – Inkrementeller Wrapper für Echtzeitbetrieb

### Stimmungspunkte (Mood Score)

Der Score liegt im Bereich **−100 (stark depressiv) bis +100 (stark manisch)**:

| Bereich | Bedeutung |
|---------|-----------|
| ≥ 60 | Stark manisch |
| 25 – 60 | Manisch |
| 8 – 25 | Leicht manisch |
| −8 – 8 | Normal / Stabil |
| −25 – −8 | Leicht depressiv |
| −60 – −25 | Depressiv |
| < −60 | Stark depressiv |

**Formel (vereinfacht):**
```
raw   = tanh((M_t − D_t) / 10.0) × 55
boost = ±tanh(dominant / 8.0) × 55 × conf_weight   [wenn Zustand bestätigt]
score = clip(raw + boost, −100, +100) × quality_damping
```

Qualitätsdämpfung: `clean` → 1.0 | `degraded` → 0.85 | `reject` → kein Score

### Unterstützte Audioformate

WAV, FLAC, WebM, MP4, OGG

---

## Datenbankschema

Die SQLite-Datenbank (`results.db`) enthält eine Tabelle `results`:

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| `id` | INTEGER | Primärschlüssel |
| `day_number` | INTEGER | Aufnahmetag (1, 2, 3, …) |
| `created_at` | TEXT | Zeitstempel (ISO 8601) |
| `is_baseline` | INTEGER | 1 = Baseline-Tag |
| `score` | REAL | Mood Score (−100 bis +100) |
| `state` | TEXT | Klassifikation (z. B. `depression-onset`) |
| `confidence` | REAL | Konfidenz der Klassifikation (0–1) |
| `dep_score` | REAL | Depressions-Teilscore |
| `man_score` | REAL | Manie-Teilscore |
| `quality` | TEXT | Audioqualität (`clean` / `degraded` / `reject`) |
| `f0_mean` … `mfcc_4` | REAL | 15 akustische Merkmale |

---

## Projektstruktur

```
SystemOfAPawn-CodeCanary/
├── app.py                          # Flask-Webanwendung (Routen, SSE, Threading)
├── database.py                     # SQLite-Datenbanklogik
├── result.py                       # Ergebnis-Datenmodell
├── requirements.txt                # Python-Abhängigkeiten
├── classifier/
│   ├── pipeline.py                 # Kern-ML-Pipeline (8 Module)
│   ├── analyzer.py                 # Einzel-Tag-Wrapper (inkrementell)
│   ├── scoring.py                  # Mood-Score-Berechnung
│   └── Hackathon_Dataset_Final/    # Beispieldatensatz (Patienten A–E)
├── static/
│   ├── css/style.css
│   └── js/
│       ├── index.js
│       ├── result.js
│       └── chart.min.js
├── templates/
│   ├── index.html                  # Hauptseite (Aufnahme / Upload)
│   ├── history.html                # Verlaufsansicht
│   └── result.html                 # Einzelergebnis-Detailseite
└── results/                        # Ausgabe-CSVs (Beispiele)
```

---

## Bekannte Einschränkungen

- **Kleine Referenzbasis:** Die Neutral-Referenz basiert auf ~28 Datenpunkten (14 Tage × 2 Patienten) und ist explorativ, nicht klinisch validiert.
- **Rauschreduktion:** Spektrale Subtraktion funktioniert nur bei stationärem Rauschen (kein Verkehr, keine Stimmen im Hintergrund).
- **Erkennungsverzögerung:** CUSUM benötigt ca. 4–5 Tage, um einen Zustandswechsel sicher zu detektieren.
- **Jitter/Shimmer:** Zuverlässigkeit reduziert bei Aufnahmen unter 8 Sekunden.
- **Kein Authentifizierungssystem:** Die Anwendung ist für den Einzelnutzer-Betrieb ausgelegt und enthält keine Benutzerverwaltung.

---

> **Hinweis:** Dieses System ist ein Forschungsprototyp und **kein zertifiziertes Medizinprodukt**. Es ersetzt keine ärztliche oder therapeutische Beurteilung.
