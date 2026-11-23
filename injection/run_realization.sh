#!/bin/bash
# run_realization.sh - una realizzazione della campagna di completezza:
#   iniezione nel file ORIGINALE -> ricerca; step 1 -> ricerca; step 2 -> ricerca; step 3 -> ricerca;
#   step 4 -> ricerca.
#
# Uso:
#   bash run_realization.sh ORIGINALE.fil real_01.params WORKDIR [SEED]
#
# Struttura prodotta (adatta a FRB_HF_Finder.py --proc_path):
#   WORKDIR/orig/OBS/NOME.fil   WORKDIR/step1/OBS/NOME.fil   WORKDIR/step2/...   ...   WORKDIR/step4/...
#   WORKDIR/truth.csv           catalogo dei burst iniettati
#   WORKDIR/logs/*.log          output di ogni passo
# OBS e' la cartella del file originale; NOME e' il suo nome, con il suffisso _processed.fil se manca:
# FRB_HF_Finder.py analizza solo i file *_processed.fil (di almeno 1 MB). I prodotti della ricerca restano in
# WORKDIR/<step>/OBS/ (come in produzione), i .json/.csv della pulizia accanto ai .fil.
#
# Per risparmiare disco, ogni .fil viene cancellato appena e' stato usato dallo step successivo e
# dalla ricerca (picco: due file). KEEP=1 li conserva tutti.
#
# Variabili d'ambiente:
#   CODE    cartella di frb-rfi-clean (default: quella di questo script/..)
#   FINDER  comando di ricerca SENZA --proc_path (default: quello di produzione qui sotto)
#   KEEP    1 = non cancellare i .fil intermedi
#   STEPS   step su cui fare la ricerca (default "orig step1 step2 step3 step4"); con STEPS=orig ci si ferma
#           dopo la ricerca sul file iniettato (file di riferimento di solo rumore)
set -euo pipefail

if [ $# -lt 3 ]; then
    sed -n '2,24p' "$0"; exit 1
fi
ORIG=$(readlink -f "$1"); PARAMS=$(readlink -f "$2"); WORK=$3; SEED=${4:-1}
CODE=${CODE:-$(readlink -f "$(dirname "$0")/..")}
FINDER=${FINDER:-"python /home/ageminardi/GEM_FRB/FRB_Finder/FRB_HF_Finder.py --telescope MC --boxcar_max 8192 --gold_prob 0.7 --snr_cut 8 --fetch_model a --gpu_id 1 --NOT_mv_analysed --NOT_slack --dm_low 25 --dm_high 700"}
KEEP=${KEEP:-0}
STEPS=${STEPS:-"orig step1 step2 step3 step4"}
OBS=$(basename "$(dirname "$ORIG")"); NAME=$(basename "$ORIG")
[[ "$NAME" == *_processed.fil ]] || NAME="${NAME%.fil}_processed.fil"
mkdir -p "$WORK/logs"; WORK=$(readlink -f "$WORK")
for s in orig step1 step2 step3 step4; do mkdir -p "$WORK/$s/$OBS"; done
f() { echo "$WORK/$1/$OBS/$NAME"; }
log() { echo "[$(date '+%F %T')] $*"; }

search() {   # ricerca sullo step $1
    log "ricerca su $1"
    $FINDER --proc_path "$WORK/$1/" > "$WORK/logs/search_$1.log" 2>&1
    if ! find "$WORK/$1" -name '*_all_candidates.csv' | grep -q .; then
        log "ERRORE: la ricerca su $1 non ha prodotto *_all_candidates.csv (vedi $WORK/logs/search_$1.log)"
        exit 1
    fi
}

drop() {     # cancella il .fil dello step $1 (salvo KEEP=1)
    [ "$KEEP" = "1" ] || { rm -f "$(f "$1")"; log "cancellato $(f "$1")"; }
}

log "realizzazione: $PARAMS -> $WORK (codice in $CODE)"
log "iniezione"
python3 "$CODE/injection/inject_frb.py" --fil "$ORIG" --params "$PARAMS" --out "$(f orig)" \
    --truth "$WORK/truth.csv" --seed "$SEED" -q > "$WORK/logs/inject.log" 2>&1
[[ " $STEPS " == *" orig "* ]] && search orig
if [ "$STEPS" = "orig" ]; then
    drop orig; log "fatto (solo ricerca sul file iniettato)"; exit 0
fi

log "step 1"
python3 "$CODE/tools/rfi_clean.py" "$(f orig)" "$(f step1)" > "$WORK/logs/step1.log" 2>&1
drop orig
[[ " $STEPS " == *" step1 "* ]] && search step1

log "step 2"
python3 "$CODE/tools/rfi_periodic.py" "$(f step1)" "$(f step2)" > "$WORK/logs/step2.log" 2>&1
drop step1
[[ " $STEPS " == *" step2 "* ]] && search step2

log "step 3"
python3 "$CODE/tools/rfi_impulses.py" "$(f step2)" "$(f step3)" > "$WORK/logs/step3.log" 2>&1
drop step2
[[ " $STEPS " == *" step3 "* ]] && search step3

log "step 4"
python3 "$CODE/tools/rfi_commonmode.py" "$(f step3)" "$(f step4)" > "$WORK/logs/step4.log" 2>&1
drop step3
[[ " $STEPS " == *" step4 "* ]] && search step4
drop step4

log "fatto. Associazione:"
echo "python3 $CODE/injection/recover.py match --plan PIANO/plan.csv --realization N \\"
for s in $STEPS; do echo "    --step $s $WORK/$s \\"; done
echo "    --impulses step3 $WORK/step3/$OBS/$NAME.impulses_events.csv --out PIANO/matched_rNN.csv"
