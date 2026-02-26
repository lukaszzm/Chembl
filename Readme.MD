# ChEMBL

Projekt analizy danych bioaktywności z bazy ChEMBL 36 dla przewidywania aktywności związków chemicznych.

## Struktura

```
├── notebooks/
│   ├── 00_intro.ipynb              # Wprowadzenie i setup
│   ├── 01_data_exploration.ipynb   # Eksploracja danych
│   └── 02_data_cleaning_eda.ipynb  # Czyszczenie
├── data/
│   ├── raw/                         # Surowe dane
│   └── processed/                   # Oczyszczone dane
└── db/
    └── chembl_36.db                 # Baza ChEMBL (do rozpakowania samemu)
```

## Wymagania

- Python 3.12+
- pandas 3.0.1
- rdkit 2025.9.5
- SQLAlchemy, matplotlib, seaborn
