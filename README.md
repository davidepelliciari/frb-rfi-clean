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
| `tools/rfi_impulses.py` | **step 3**: rimozione degli impulsi a DM 0 (radar, impulsi a banda intera o parziale) con veto di dispersione |
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

## Step 2: RFI periodico (automatico)

```bash
python3 tools/rfi_periodic.py step1.fil step2.fil              # ricerca automatica
python3 tools/rfi_periodic.py step1.fil step2.fil --freq 50    # solo le frequenze indicate
```

1. **Analisi**: spettro della serie a DM 0 a risoluzione nativa (fino a Nyquist, senza aliasing),
   righe oltre 8 volte il continuo (frequenza interpolata), raggruppamento in famiglie armoniche
   scegliendo la fondamentale che spiega piu' righe oltre il caso (es. 50 Hz e 166.67 Hz -> 16.667 Hz),
   affinamento della frequenza a stadi (spostamento del profilo piegato nel tempo).
2. **Test di dispersione**: i 4 quarti di banda vengono piegati e confrontati; una famiglia con
   ritardi compatibili con DM >= 2 (es. una pulsar) **non viene sottratta**; anche le famiglie per
   cui il test non e' conclusivo non vengono sottratte (prudenza) e sono segnalate.
3. **Sottrazione**: per ogni canale e tratto (>= 200 periodi) il profilo medio clippato **a
   risoluzione nativa** viene sottratto in base alla fase: tutte le armoniche fino a Nyquist, senza
   creare righe nuove. Arrotondamento stocastico per uscite intere.
4. **Verifica**: spettro dell'output; se restano famiglie sottraibili si ripete (default 2 iterazioni).

Prodotti: `step2.fil.periodic.json` (righe prima/dopo, famiglie, test di dispersione, decisioni),
`step2.fil.periodic_spectrum.png` e `.npz` (spettro prima/dopo).

Validazione (`tests/test_periodic.py`): con RFI a 50 Hz e un treno di impulsi a 166.67 Hz viene trovata la
fondamentale comune 16.667 Hz e restano solo le righe di una pulsar (DM 40) che viene riconosciuta e
lasciata intatta; gli FRB hanno lo stesso S/N (>= 95%) del file senza RFI; su dati senza periodicita'
l'output e' identico all'input. Su 60 s con le specifiche di Medicina (256 canali, 16 us): 190 righe
prima, 0 dopo, 37 s di calcolo.

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

## Step 3: impulsi a DM 0

```bash
python3 tools/rfi_impulses.py step2.fil step3.fil
python3 tools/rfi_impulses.py step2.fil step3.fil --thresholds 10 8 6 5
```

1. **Rivelazione** sulle serie a DM 0 della banda intera e di 16 sottobande, boxcar da 1 campione a
   32 ms, con **soglia iterativa** (default 8 poi 6): la sigma robusta di ogni serie viene ristimata
   escludendo i tratti gia' oltre soglia.
2. **Veto di dispersione** per ogni evento:
   - un evento piu' stretto di meta' della dispersione a DM 15 nella serie in cui e' visto non puo'
     essere un FRB a DM >= 15: e' RFI;
   - altrimenti si cerca una traccia dispersa che passi per l'evento, sommando le serie delle
     sottobande su una griglia di DM (0-1000) e di larghezze; l'evento e' protetto se il miglior S/N a
     DM >= 15 supera di 1.05 volte quello a DM < 15, e' almeno 7, e lungo la traccia almeno il 40% delle
     sottobande ha S/N > 2 (un FRB e' esteso in frequenza; una traccia che raccoglie poche sottobande
     forti e' un'altra RFI attraversata per caso). Per gli eventi a banda parziale si usano solo le
     sottobande non colpite.
3. **Sostituzione** con rumore del solo rettangolo tempo x sottobande colpite; tutti gli altri
   campioni restano identici all'input.

Prodotti: `step3.fil.impulses.json` (eventi rimossi/protetti, frazione di dati sostituita) e
`step3.fil.impulses_events.csv`.

Validazione (`tests/test_impulses.py` e griglia DM x larghezza su dati sintetici con radar a banda intera
e parziale e impulsi stretti): eventi a DM 0 ridotti di circa l'88%, 0.1-0.4% dei dati sostituiti,
nessuna modifica su rumore puro. **Limite**: un FRB abbastanza brillante da essere visto a DM 0 e'
protetto se la sua larghezza e' al massimo ~0.7 volte lo sweep del suo DM (con 64 MHz a 1.4 GHz:
~4 ms a DM 30, tutte le larghezze fino a 16 ms a DM >= 50); oltre, e per DM < 20, puo' essere rimosso.
Gli FRB deboli, non rivelati a DM 0, non vengono toccati.
