<#
.SYNOPSIS
    Build the image with the newest Ollama.

.DESCRIPTION
    The PowerShell twin of build.sh, for Windows without a bash shell.

    Why this exists rather than a plain `docker build`: the Ollama install is a
    RUN layer, and Docker caches those by command text alone. A plain rebuild
    therefore keeps whatever Ollama the layer was first built with, however old
    — which is why a newly released model can report as unsupported in an image
    you just rebuilt. Resolving the version here and passing it in makes the
    layer key change when the version does, so you get the newest Ollama while
    rebuilds stay fast when nothing has moved.

.EXAMPLE
    .\build.ps1
.EXAMPLE
    .\build.ps1 -Image myname:tag
.EXAMPLE
    .\build.ps1 -OllamaVersion 0.32.0     # pin, to roll back or reproduce
#>
param(
    [string]$Image = "lmmasters/llm_extractinator:latest",
    [string]$OllamaVersion = ""
)

$ErrorActionPreference = "Stop"

if (-not $OllamaVersion) {
    try {
        $release = Invoke-RestMethod -TimeoutSec 15 `
            -Uri "https://api.github.com/repos/ollama/ollama/releases/latest"
        $OllamaVersion = $release.tag_name -replace '^v', ''
    }
    catch {
        Write-Warning "Could not reach GitHub to resolve the latest Ollama version."
        Write-Warning "Building without a version, so the install script picks its own latest."
        Write-Warning "Note: if that layer is already cached, the Ollama in the image will NOT"
        Write-Warning "change. Use 'docker build --no-cache' if you need it refreshed."
        docker build -t $Image .
        exit $LASTEXITCODE
    }
}

Write-Host "Building $Image with Ollama $OllamaVersion"
docker build --build-arg "OLLAMA_VERSION=$OllamaVersion" -t $Image .
exit $LASTEXITCODE
