#!/bin/bash

# Verifica argomento
if [ -z "$1" ]; then
  echo "Uso: $0 /percorso/immagine.fit"
  exit 1
fi

filepath="$1"
filepath="${filepath//\\//}"  # Sostituisce eventuali backslash con slash

filename="$(basename "$filepath")"
basename="${filename%.*}"
directory="$(dirname "$filepath")"
solvedname="${basename}_solved.fit"
solvedpath="${directory}/${solvedname}"
ssfPath="/tmp/solve_script.ssf"

# Contenuto dello script Siril
read -r -d '' scriptContent <<EOF
requires 1.4.1-0

cd "$directory"
load "$filename"
platesolve -order=2
save "$solvedname"
close
EOF

# Scrivi il file in UTF-8 senza BOM
printf '%s\n' "$scriptContent" > "$ssfPath"

# Debug: mostra i primi byte del file
echo -n "Primi byte del file .ssf: "
xxd -p -l 8 "$ssfPath" | sed 's/../& /g'

# Esegui Siril CLI
siril-cli -s "$ssfPath"

