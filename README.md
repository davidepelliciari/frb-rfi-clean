# frb-rfi-clean

Pipeline per la rimozione degli RFI (in particolare impulsivi, di durata ~ms) da
dati SIGPROC filterbank, prima della ricerca di FRB con Heimdall + FETCH.

## Contenuto attuale

| File | Scopo |
|---|---|
| `tools/fil_cut.py` | info e tagli di filterbank grandi senza caricarli in memoria (solo libreria standard) |
| `tools/make_noise_fil.py` | filterbank di solo rumore noto, con l'header di un file reale (per calibrare l'S/N su Heimdall) |
| `injection/plan_injections.py` | pianifica una campagna di iniezione (`.params`, `injections.csv`, `campaign.json`) ed eventualmente la esegue |
| `injection/inject_frb.py` | iniettore di FRB in un solo passaggio, a blocchi; scrive il catalogo di verità `*.truth.csv` |
| `tools/rfi_survey.py` | censimento degli RFI (qualsiasi banda, tsamp, nbits 1-32): statistiche per canale, eventi impulsivi a DM 0 in banda intera e sottobande, periodicità (radar, DME), canali sospetti, grafici |
| `frbclean/` | pacchetto con lettura/scrittura SIGPROC a blocchi (`sigproc.py`) e il survey (`survey.py`) |

Dipendenze: `numpy`, `scipy`, `matplotlib` per i grafici (`pytest` per i test).
Installazione opzionale come pacchetto: `pip install -e .`

## Censimento degli RFI

```bash
python3 tools/rfi_survey.py osservazione.fil --out survey_osservazione/
```

Un solo passaggio sul file (circa 2.5x più veloce del tempo reale per 256 canali a 16 µs
su 4 core). Prodotti in `survey_osservazione/`:

- `summary.json` – riassunto: eventi/ora sopra varie soglie di S/N, frazione di tempo occupata,
  eventi per larghezza e per frazione di banda, i 20 eventi più forti, canali sospetti, periodicità
- `events.csv` – un evento per riga: tempo, larghezza, S/N (banda intera e massimo sulle sottobande),
  sottobande colpite (in ordine di frequenza crescente)
- `periodicity.csv` – periodicità nella serie a DM 0 e nei treni di eventi (tutti e solo a banda larga);
  per i treni di eventi il periodo è affinato con il test di Rayleigh
- `flagged_channels.txt` – canali sospetti in formato `min max` (come il flag file di FRB_HF_Finder)
- `channels.png`, `dynspec.png`, `events.png`, `zerodm.png` – grafici diagnostici
- `survey.npz` – dati grezzi del survey (statistiche per blocco e canale, spettri dinamici ridotti,
  serie a DM 0 a ~1 ms)

Opzioni utili: `--tmax 120` (solo i primi 120 s), `--max-width 0.064`, `--nsub 32`, `--thresh 6`.

```bash
python3 -m pytest -q tests
```

## Calibrazione dell'S/N su Heimdall

1. `tools/make_noise_fil.py --like obs.fil --out noise.fil --duration 600 --bandpass-from-data`
2. `injection/plan_injections.py --fil noise.fil --outdir calib --nfrb 300 --snr 6:40 --dm ... --width ... --run --smear`
3. Heimdall sul file iniettato, poi associazione dei candidati a `*.truth.csv` e fit
   `S/N_heimdall = pendenza * S/N_modello + offset`.
4. Le campagne successive usano `--snr-calib PENDENZA,OFFSET`.
