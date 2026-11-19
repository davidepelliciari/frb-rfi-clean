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
| `tools/rfi_clean.py` | **primo stadio di pulizia**: maschera dei canali, normalizzazione per canale che segue i salti di guadagno, sostituzione delle celle tempo-frequenza con varianza anomala, veto di dispersione per non toccare i segnali dispersi |
| `tools/rfi_periodic.py` | **step 2**: sottrazione canale per canale delle componenti periodiche (default 50 Hz e armoniche) |
| `frbclean/` | pacchetto: I/O SIGPROC a blocchi (`sigproc.py`), survey (`survey.py`), step 1 (`clean.py`), step 2 (`periodic.py`), ricerca a singolo impulso semplificata per i test (`minisearch.py`) |
| `docs/report_no0004/` | report LaTeX sull'osservazione ag278m_mc_no0004 e script di analisi riproducibile |

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

## Pulizia (primo stadio)

```bash
python3 tools/rfi_clean.py osservazione.fil osservazione_clean.fil
```

L'output ha lo stesso header e lo stesso numero di bit e va passato a Heimdall al posto dell'originale.
Per dati a 8 bit l'output ha media 128 e deviazione standard 8 (dinamica di +-16 sigma).
Accanto all'output vengono scritti `osservazione_clean.fil.clean.json` (riassunto: canali mascherati,
salti di livello trovati, frazione di celle sostituite, eventi protetti dal veto) e `.clean.npz`
(livello e rumore per canale nel tempo, maschera).

Passaggi (tre letture del file):

1. **Salti di livello**: filtro a gradino sulle serie a DM 0 di 16 sottobande (risoluzione 0.5 ms),
   con test di persistenza (il nuovo livello deve restare per almeno 0.2 s, a 0.1 s di distanza), così
   un burst non viene scambiato per un salto.
2. **Livello e rumore per canale** su finestre da 0.2 s che non attraversano i salti; media e
   deviazione standard clippate; mediana mobile su 5 finestre dentro ogni segmento. Maschera
   automatica dei canali con gli stessi criteri del survey (oppure `--mask-file`, `--mask-chans`).
3. **Pulizia**: normalizzazione, poi celle da 8 ms per canale con varianza (dalle differenze tra
   campioni consecutivi, insensibile ai segnali lisci) oltre 6 sigma robuste vengono sostituite con
   rumore; se più di metà dei canali è anomala si sostituisce l'intero intervallo. Prima di sostituire
   un evento che coinvolge almeno il 2% dei canali si applica un **veto di dispersione**: se il segnale
   è più forte a un DM tra 15 e 1500 che a DM 0, le celle lungo la traccia dispersa non vengono toccate.

Validazione (test in `tests/test_clean.py` e verifiche su file con le specifiche di Medicina):
salti di guadagno, lampi di rumore e glitch del tipo osservato in ag278m_mc_no0004 non producono più
candidati; gli FRB iniettati (0.05-50 ms, S/N 12-1000, DM 30-500) conservano l'S/N entro -6%/+10%
(il guadagno viene dalla normalizzazione, che dà a tutti i canali lo stesso peso).
Gli impulsi radar a DM 0 **non** sono ancora rimossi (secondo stadio).

## Step 2: componenti periodiche

```bash
python3 tools/rfi_periodic.py step1.fil step2.fil            # 50 Hz e armoniche
python3 tools/rfi_periodic.py step1.fil step2.fil --search   # affina la frequenza per ogni tratto
```

Per ogni tratto di 10 s e canale si stima il profilo medio su un periodo (64 intervalli di fase,
media clippata a 4 sigma) e lo si sottrae in base alla fase; vengono rimosse tutte le armoniche
fino alla 32a. In ag278m_mc_no0004 la componente a 50.0000 Hz (stabile entro 0.001 Hz, quindi
probabilmente non la rete elettrica) spiega in mediana il 12% della varianza della serie a DM 0 a 1 ms.
Validazione (`tests/test_periodic.py`): righe a 50/100/150 Hz ridotte da >1000 a <50 volte il fondo;
nessun effetto su dati senza periodicità; S/N degli FRB uguale a quello dello stesso segnale in dati
senza modulazione (entro il rumore di misura).
Riassunto in `step2.fil.periodic.json` (frequenza, ampiezza del profilo e varianza rimossa per tratto).

### Diagnostica delle righe periodiche

```bash
python3 tools/rfi_survey.py step1.fil --out survey_step1/ --spectrum
python3 tools/rfi_survey.py step2.fil --out survey_step2/ --spectrum
python3 tools/compare_spectra.py survey_step1/ survey_step2/ --labels "step 1" "step 2" --out confronto/
```

Con `--spectrum` il survey calcola anche lo spettro di potenza della serie a DM 0 (banda intera e 16
sottobande; media degli spettri dei blocchi con finestra di Hann; fino a 2 kHz, risoluzione ~0.24 Hz
con blocchi da 4.19 s) e produce `spectrum.png`, `spectrum_lines.csv` (righe oltre 10 volte il continuo,
con la famiglia armonica n x f0) e `spectrum.npz`. `compare_spectra.py` sovrappone due spettri, ne fa il
rapporto e, per ogni riga del primo, riporta la frazione di potenza (sopra il continuo) rimasta nel secondo.
