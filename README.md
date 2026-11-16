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

Dipendenze: `numpy`, `scipy` (`pytest` per i test).

```bash
python3 -m pytest -q tests
```

## Calibrazione dell'S/N su Heimdall

1. `tools/make_noise_fil.py --like obs.fil --out noise.fil --duration 600 --bandpass-from-data`
2. `injection/plan_injections.py --fil noise.fil --outdir calib --nfrb 300 --snr 6:40 --dm ... --width ... --run --smear`
3. Heimdall sul file iniettato, poi associazione dei candidati a `*.truth.csv` e fit
   `S/N_heimdall = pendenza * S/N_modello + offset`.
4. Le campagne successive usano `--snr-calib PENDENZA,OFFSET`.
