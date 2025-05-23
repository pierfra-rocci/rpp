#!/bin/bash

set -e
set -x

# Ensure /tmp exists
if [ ! -d /tmp ]; then
  mkdir -p /tmp
fi

filepath="$1"
filepath="${filepath//\\//}"

# Check if file exists
if [ ! -f "$filepath" ]; then
  echo "Errore: il file '$filepath' non esiste."
  exit 1
fi

# Check if siril-cli is available
if ! command -v siril-cli >/dev/null 2>&1; then
  echo "Errore: siril-cli non trovato nel PATH."
  exit 1
fi

filename="$(basename "$filepath")"
basename="${filename%.*}"
directory="$(dirname "$filepath")"
solvedname="${basename}_solved.fits"
solvedpath="${directory}/${solvedname}"
ssfPath="${filepath}_solve_script.ssf"
sirilLog="${filepath}_siril_cli.log"

# Write Siril script directly to file
cat > "$ssfPath" << EOF
requires 1.2.6

cd "$directory"
load "$filename"
platesolve
save "$solvedname"
close
EOF

# Debug: mostra i primi byte del file
echo -n "Primi byte del file .ssf: "
xxd -p -l 8 "$ssfPath" | sed 's/../& /g'

# Function to cleanup temporary files
cleanup() {
  rm -f "$ssfPath" "$sirilLog"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Esegui Siril CLI e salva log
if ! siril-cli -s "$ssfPath" > "$sirilLog" 2>&1; then
  echo "Errore: Siril non ha risolto l'immagine. Log di Siril:"
  cat "$sirilLog"
  
  # Check for specific error patterns
  if grep -q "No valid WCS found" "$sirilLog"; then
    echo ""
    echo "SUGGERIMENTO: Il file FITS non contiene le informazioni WCS necessarie."
    echo "Potrebbe essere necessario:"
    echo "1. Verificare che l'immagine sia stata acquisita con informazioni di telescopio corrette"
    echo "2. Utilizzare un software di plate solving esterno (come astrometry.net)"
    echo "3. Aggiungere manualmente le informazioni di coordinate approssimative"
  elif grep -q "Plate solving failed" "$sirilLog"; then
    echo ""
    echo "SUGGERIMENTO: Il plate solving è fallito. Possibili cause:"
    echo "1. Stelle insufficienti o poco visibili nell'immagine"
    echo "2. Campo di vista non riconosciuto dal catalogo stelle"
    echo "3. Immagine troppo rumorosa o con artefatti"
  fi
  
  exit 2
fi

# Controlla se il file risolto è stato creato
if [ ! -f "$solvedpath" ]; then
  echo "Errore: il file risolto '$solvedpath' non è stato creato."
  echo "Log di Siril:"
  cat "$sirilLog"
  exit 3
fi

echo "Plate solving completato con successo!"
echo "File risolto salvato come: $solvedpath"
echo ""
echo "Log di Siril:"
cat "$sirilLog"