#requires -RunAsAdministrator
[CmdletBinding()]
param(
    [string]$ListenAddress = '10.0.0.141',
    [ValidateRange(1, 65535)][int]$ListenPort = 8009,
    [ValidateRange(1, 65535)][int]$ApiPort = 8008,
    [string]$InterfaceAlias = 'Ethernet 2',
    [string]$ClientSubnet = '10.0.0.0/24',
    [switch]$Disable
)
$ErrorActionPreference = 'Stop'
$ruleName = "FLORA-LAN-$ListenPort"
$taskRoot = Split-Path -Parent $PSScriptRoot
$resultPath = Join-Path $taskRoot '.runtime/lan-setup-result.json'
try {
    if ($Disable) {
        netsh interface portproxy delete v4tov4 "listenaddress=$ListenAddress" "listenport=$ListenPort"
        Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
        $result = @{ success = $true; enabled = $false }
    } else {
        $address = Get-NetIPAddress -InterfaceAlias $InterfaceAlias -AddressFamily IPv4 |
            Where-Object IPAddress -eq $ListenAddress
        if (-not $address) { throw "$ListenAddress is not assigned to $InterfaceAlias" }
        $apiSocket = [Net.Sockets.TcpClient]::new()
        try { $apiSocket.Connect('127.0.0.1', $ApiPort) } finally { $apiSocket.Dispose() }
        Start-Service iphlpsvc
        netsh interface portproxy add v4tov4 "listenaddress=$ListenAddress" "listenport=$ListenPort" "connectaddress=127.0.0.1" "connectport=$ApiPort"
        if ($LASTEXITCODE -ne 0) { throw "Port proxy setup failed with exit code $LASTEXITCODE" }
        Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
        New-NetFirewallRule -Name $ruleName -DisplayName "FLORA LAN API $ListenPort" `
            -Direction Inbound -Action Allow -Protocol TCP -LocalAddress $ListenAddress `
            -LocalPort $ListenPort -RemoteAddress $ClientSubnet -InterfaceAlias $InterfaceAlias `
            -Profile Any | Out-Null
        $result = @{ success = $true; enabled = $true; url = "http://${ListenAddress}:$ListenPort";
                     clients = $ClientSubnet; interface = $InterfaceAlias }
    }
    $result | ConvertTo-Json | Set-Content -LiteralPath $resultPath -Encoding UTF8
    $result | ConvertTo-Json
} catch {
    @{ success = $false; error = $_.Exception.Message } | ConvertTo-Json |
        Set-Content -LiteralPath $resultPath -Encoding UTF8
    Write-Error $_
    exit 1
}
