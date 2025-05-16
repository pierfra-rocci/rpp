#!/bin/bash

set -e
set -x

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment not activated. Please run 'source venv/bin/activate' first."
    exit 1
fi

# Verifica argomento
if [ -z "$1" ]; then
  echo "Uso: $0 /percorso/immagine.fit"
  exit 1
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

# Contenuto dello script Siril
read -r -d '' scriptContent <<EOF
requires 1.2.6

cd "$directory"
load "$filename"
platesolve -force
save "$solvedname"
close
EOF

# Scrivi il file in UTF-8 senza BOM
printf '%s\n' "$scriptContent" > "$ssfPath"

# Debug: mostra i primi byte del file
echo -n "Primi byte del file .ssf: "
xxd -p -l 8 "$ssfPath" | sed 's/../& /g'

# Esegui Siril CLI e salva log
if ! siril-cli -s "$ssfPath" > "$sirilLog" 2>&1; then
  echo "Errore: Siril non ha risolto l'immagine. Log di Siril:"
  cat "$sirilLog"
  exit 2
fi

