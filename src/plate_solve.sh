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

# Check for both possible extensions
solvedname="${basename}_solved.fits"
solvedpath="${directory}/${solvedname}"
solvedname_alt="${basename}_solved.fit"
solvedpath_alt="${directory}/${solvedname_alt}"

ssfPath="/tmp/solve_script.ssf"
sirilLog="/tmp/siril_cli.log"

# Write Siril script directly to file
cat > "$ssfPath" << EOF
requires 1.2.6

cd "$directory"
load "$filename"
# Try with local catalog and a longer timeout
platesolve -local -timeout 60
save "$solvedname"
close
EOF

# Debug: mostra i primi byte del file
echo -n "Primi byte del file .ssf: "
xxd -p -l 8 "$ssfPath" | sed 's/../& /g'

# Esegui Siril CLI e salva log
siril-cli -s "$ssfPath" > "$sirilLog" 2>&1
result=$?

# Mostra sempre il log di Siril
echo "Log di Siril:"
cat "$sirilLog"

# Check if plate solving failed
if [ $result -ne 0 ]; then
  echo "Errore: Siril ha restituito un errore (codice $result)."
  exit 2
fi

# Controlla se il file risolto è stato creato (prova entrambe le estensioni)
if [ ! -f "$solvedpath" ] && [ ! -f "$solvedpath_alt" ]; then
  echo "Errore: il file risolto '$solvedpath' o '$solvedpath_alt' non è stato creato."
  exit 3
fi

# Trova quale file è stato creato
if [ -f "$solvedpath" ]; then
  echo "File risolto: $solvedpath"
elif [ -f "$solvedpath_alt" ]; then
  echo "File risolto: $solvedpath_alt"
fi

echo "Plate solving completed successfully."