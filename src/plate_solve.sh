#!/bin/bash

set -e
set -x

# Ensure /tmp exists
if [ ! -d /tmp ]; then
  mkdir -p /tmp
fi

filepath="$1"
filepath="${filepath//\\//}"  # Sostituisce eventuali backslash con slash

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
solvedname="${basename}_solved.fit"
solvedpath="${directory}/${solvedname}"
ssfPath="/tmp/solve_script.ssf"
sirilLog="/tmp/siril_cli.log"

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

# Esegui Siril CLI e salva log
if ! siril-cli -s "$ssfPath" > "$sirilLog" 2>&1; then
  echo "Errore: Siril non ha risolto l'immagine. Log di Siril:"
  cat "$sirilLog"
  exit 2
fi

# Controlla se il file risolto è stato creato
if [ ! -f "$solvedpath" ]; then
  echo "Errore: il file risolto '$solvedpath' non è stato creato."
  echo "Log di Siril:"
  cat "$sirilLog"
  exit 3
fi

# Mostra il log di Siril anche in caso di successo
echo "Log di Siril:"
cat "$sirilLog"
