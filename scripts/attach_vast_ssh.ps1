# Attach a PUBLIC key to an existing Vast Docker instance using the official API.
# The API token is entered interactively, never stored in a file or CLI argument.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateRange(1, 2147483647)]
    [int]$InstanceId,
    [string]$PublicKeyPath = (Join-Path $env:USERPROFILE '.ssh/id_ed25519_stretto_vast.pub')
)

$ErrorActionPreference = 'Stop'
if (-not $PublicKeyPath.EndsWith('.pub') -or -not (Test-Path -LiteralPath $PublicKeyPath -PathType Leaf)) {
    throw 'Provide an existing .pub public-key file.'
}
$taskPublicKey = [System.IO.File]::ReadAllText($PublicKeyPath).Trim()
if ($taskPublicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/]+={0,2}( [^\r\n]+)?$') {
    throw 'Expected a single-line OpenSSH Ed25519 public key.'
}
Write-Host "Target Vast instance: $InstanceId"
Write-Host "Public key file: $PublicKeyPath"
Write-Host 'This only adds this public key. It does not stop, destroy, or rent instances.'
$taskSecureToken = Read-Host 'Paste Vast API key (hidden input)' -AsSecureString
$taskTokenPointer = [IntPtr]::Zero
$taskHeaders = $null
$taskToken = $null
$taskFailed = $false
try {
    $taskTokenPointer = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($taskSecureToken)
    $taskToken = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($taskTokenPointer).Trim()
    if ([string]::IsNullOrWhiteSpace($taskToken)) { throw 'Empty API token' }
    $taskHeaders = @{ Authorization = 'Bearer ' + $taskToken }
    $taskBody = @{ ssh_key = $taskPublicKey } | ConvertTo-Json -Compress
    $taskResponse = Invoke-RestMethod -Method Post `
        -Uri "https://console.vast.ai/api/v0/instances/$InstanceId/ssh" `
        -Headers $taskHeaders -ContentType 'application/json' -Body $taskBody `
        -TimeoutSec 30 -MaximumRedirection 0
    if ($taskResponse.success -ne $true) { throw 'API did not report success' }
    Write-Host 'SSH public key attached successfully. Return to Codex and say it is done.'
}
catch {
    $taskFailed = $true
    $taskStatus = 'unavailable'
    if ($_.Exception.Response -and $_.Exception.Response.StatusCode) {
        $taskStatus = [int]$_.Exception.Response.StatusCode
    }
    # Do not echo raw HTTP responses, headers, token values, or exception text.
    Write-Host "Attach failed (HTTP status: $taskStatus). Check the instance ID and API-key permissions."
}
finally {
    if ($taskHeaders) { $taskHeaders.Clear() }
    $taskToken = $null
    if ($taskTokenPointer -ne [IntPtr]::Zero) {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($taskTokenPointer)
    }
    if ($taskSecureToken) { $taskSecureToken.Dispose() }
}
if ($taskFailed) { exit 1 }
