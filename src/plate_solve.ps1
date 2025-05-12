param (
    [Parameter(Mandatory = $true)]
    [string]$filepath
)

$filepath = $filepath -replace '\\', '/'
$filename = [System.IO.Path]::GetFileName($filepath)
$basename = [System.IO.Path]::GetFileNameWithoutExtension($filepath)
$directory = [System.IO.Path]::GetDirectoryName($filepath) -replace '\\', '/'
$solvedname = "$basename`_solved.fits"

$scriptContent = @"
requires 1.3.4

cd "$directory"
load "$filename"
platesolve -localasnet
save "$solvedname"
close
"@

$ssfPath = "$env:TEMP\solve_script.ssf"

# Scrittura in UTF-8 senza BOM
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($ssfPath, $scriptContent, $utf8NoBom)

# Debug: visualizza i primi byte per confermare assenza del BOM
# $bytes = [System.IO.File]::ReadAllBytes($ssfPath)[0..7]
# $hex = ($bytes | ForEach-Object { "{0:X2}" -f $_ }) -join ' '
# Write-Host "Primi byte del file `.ssf`: $hex"

# Esegui Siril normalmente
siril-cli.exe -s "$ssfPath"
