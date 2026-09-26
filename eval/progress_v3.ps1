# Live progress window for the v3 run. Single file; relaunch any time with:
#   powershell -NoExit -ExecutionPolicy Bypass -File eval\progress_v3.ps1
# Reads only the run JSONL files, results/v3/status.json, results/v3/ALERT.txt and gateway log counts.
# Refreshes every 30 s. Source is ASCII only: box-drawing characters are built from their code points.

param([switch]$Once)
$ErrorActionPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$W = 92                                     # inner width of the panel
$CeilingH = 15.0
$H = [string][char]0x2550; $V = [string][char]0x2551
$TL = [char]0x2554; $TR = [char]0x2557; $BL = [char]0x255A; $BR = [char]0x255D; $ML = [char]0x2560; $MR = [char]0x2563
$Full = [string][char]0x2588; $Light = [string][char]0x2591
$Ok = [string][char]0x2714; $Bad = [string][char]0x2718; $Dot = [string][char]0x25CF

$Phases = @(
    @{ Name = 'DEV: B2 (v3 engine)';  File = 'results\v3\dev\runs\b2_dev.jsonl';   Total = 50;  Llm = $false },
    @{ Name = 'DEV: v3';              File = 'results\v3\dev\runs\v3_dev.jsonl';   Total = 50;  Llm = $true  },
    @{ Name = 'TEST: B2 (v3 engine)'; File = 'results\v3\runs\test_b2_v3.jsonl';   Total = 170; Llm = $false },
    @{ Name = 'TEST: v3';             File = 'results\v3\runs\test_v3.jsonl';      Total = 170; Llm = $true  }
)
$Cache = @{}                                # file -> @{ Lines = n; Rows = list of summaries }

function Read-Rows($file) {
    if (-not (Test-Path $file)) { return @() }
    $lines = [IO.File]::ReadAllLines((Join-Path $Root $file))
    if (-not $Cache.ContainsKey($file)) { $Cache[$file] = @{ Lines = 0; Rows = New-Object System.Collections.ArrayList } }
    $c = $Cache[$file]
    for ($i = $c.Lines; $i -lt $lines.Count; $i++) {
        if ($lines[$i].Trim() -eq '') { continue }
        $r = $lines[$i] | ConvertFrom-Json
        [void]$c.Rows.Add([pscustomobject]@{ Id = $r.id; Control = [bool]$r.score.is_control; Top1 = $r.score.top1;
            FalseAlarm = $r.score.false_alarm; Error = [string]$r.error; Latency = [double]$r.latency_s })
    }
    $c.Lines = $lines.Count
    # latest attempt per scenario
    $by = [ordered]@{}; foreach ($x in $c.Rows) { $by[$x.Id] = $x }
    return @($by.Values)
}

function Seg([string]$t, [string]$col = 'Gray') { return @{ T = $t; C = $col } }

function Row($segs) {
    Write-Host -NoNewline $V -ForegroundColor DarkCyan
    $n = 0
    foreach ($s in $segs) { Write-Host -NoNewline $s.T -ForegroundColor $s.C; $n += $s.T.Length }
    if ($n -lt $W) { Write-Host -NoNewline (' ' * ($W - $n)) }
    Write-Host $V -ForegroundColor DarkCyan
}
function Rule($l, $r) { Write-Host ("$l" + ($H * $W) + "$r") -ForegroundColor DarkCyan }
function Fmt([double]$min) { if ($min -lt 0) { $min = 0 }; $h = [math]::Floor($min / 60); $m = [math]::Round($min - 60 * $h); return ('{0}h {1:00}m' -f $h, $m) }
function Bar([int]$done, [int]$total, [int]$width = 28) {
    $f = if ($total -gt 0) { [math]::Floor($width * $done / $total) } else { 0 }
    return ($Full * $f) + ($Light * ($width - $f))
}

while ($true) {
    $now = Get-Date
    $start = [datetime]::Parse((Get-Content 'results\v3\run_start.txt' -Raw).Trim(), [Globalization.CultureInfo]::InvariantCulture)
    $elapsedMin = ($now - $start).TotalMinutes
    $leftMin = $CeilingH * 60 - $elapsedMin
    $status = $null; if (Test-Path 'results\v3\status.json') { $status = Get-Content 'results\v3\status.json' -Raw | ConvertFrom-Json }
    $host.UI.RawUI.WindowTitle = ('KPI RCA v3  |  elapsed {0} of 15h 00m  |  budget left {1}' -f (Fmt $elapsedMin), (Fmt $leftMin))

    $info = @(); $current = $null; $alerts = @()
    foreach ($p in $Phases) {
        $rows = Read-Rows $p.File
        $done = @($rows | Where-Object { -not $_.Error.StartsWith('llm error') }).Count
        $api = @($rows | Where-Object { $_.Error.StartsWith('llm error') }).Count
        $model = @($rows | Where-Object { $_.Error -and -not $_.Error.StartsWith('llm error') }).Count
        $x = [pscustomobject]@{ P = $p; Rows = $rows; Done = $done; Api = $api; Model = $model }
        $info += $x
        if (-not $current -and $done -lt $p.Total) { $current = $x }
        if ($api -gt 0) { $alerts += "$($p.Name): $api API failure(s) (will be retried)" }
    }
    $stop = $null; if ($status -and $status.time_budget_stop) { $stop = $status.time_budget_stop }
    if ($stop) { $alerts += "Stopped for the 15-hour time budget in $($stop.phase) after $($stop.completed) scenarios (not an error)"; $current = $null }
    if (Test-Path 'results\v3\ALERT.txt') { foreach ($l in Get-Content 'results\v3\ALERT.txt') { if ($l.Trim()) { $alerts += $l.Trim() } } }
    $r429 = 0; try { $r429 = ((docker logs --since 5m llm-gateway-gateway-1 2>&1) | Out-String | Select-String -Pattern '429 Too Many' -AllMatches).Matches.Count } catch {}

    Clear-Host
    Rule $TL $TR
    Row @((Seg '  KPI ROOT-CAUSE AGENT  v3 RUN' 'White'), (Seg ('      updated ' + $now.ToString('yyyy-MM-dd HH:mm:ss', [Globalization.CultureInfo]::InvariantCulture)) 'DarkGray'))
    Row @((Seg ('  Elapsed ' + (Fmt $elapsedMin) + ' of 15h 00m') 'Cyan'), (Seg ('    Budget left ' + (Fmt $leftMin)) ($(if ($leftMin -lt 60) { 'Red' } else { 'Cyan' }))),
          (Seg ('    Workers: 1 (locked)   429s/5min: ' + $r429) 'DarkGray'))
    Rule $ML $MR
    Row @((Seg ('  {0,-22} {1,-28}  {2,9}  {3,8}  {4,10}' -f 'PHASE', 'PROGRESS', 'DONE', 'TOP-1', 'ETA') 'DarkGray'))
    $totalEtaMin = 0.0
    foreach ($x in $info) {
        $p = $x.P; $done = $x.Done; $total = $p.Total
        $planted = @($x.Rows | Where-Object { -not $_.Control })
        $t1 = @($planted | Where-Object { $_.Top1 -eq $true }).Count
        $t1s = if ($planted.Count -gt 0) { '{0:0.0}%' -f (100.0 * $t1 / $planted.Count) } else { '-' }
        $lat = @($x.Rows | Where-Object { $_.Latency -gt 0 } | ForEach-Object { $_.Latency })
        $avgMin = if ($lat.Count -gt 0) { ($lat | Measure-Object -Average).Average / 60 } elseif ($p.Llm) { 3.5 } else { 0.02 }
        $etaMin = ($total - $done) * $avgMin
        if (-not $stop) { $totalEtaMin += $etaMin }
        if ($done -ge $total) { $col = 'Green'; $mark = $Ok } elseif ($current -and $current.P.Name -eq $p.Name) { $col = 'Yellow'; $mark = $Dot } else { $col = 'Gray'; $mark = ' ' }
        if ($x.Api -gt 0) { $col = 'Red' }
        $eta = if ($done -ge $total) { 'done' } else { Fmt $etaMin }
        Row @((Seg ("  $mark " + ('{0,-20}' -f $p.Name)) $col), (Seg (' ' + (Bar $done $total)) $col),
              (Seg ('  {0,4}/{1,-4}  {2,8}  {3,10}' -f $done, $total, $t1s, $eta) $col))
    }
    Rule $ML $MR
    if ($current) {
        $p = $current.P; $rows = $current.Rows
        $lat = @($rows | Where-Object { $_.Latency -gt 0 } | ForEach-Object { $_.Latency })
        $avg = if ($lat.Count -gt 0) { ($lat | Measure-Object -Average).Average / 60 } else { 0 }
        $phaseEl = if ($status -and $status.phase_started -and $status.phase -eq $p.Name) { Fmt ($now - [datetime]::Parse($status.phase_started, [Globalization.CultureInfo]::InvariantCulture)).TotalMinutes } else { '-' }
        Row @((Seg '  RUNNING  ' 'Cyan'), (Seg $p.Name 'Yellow'), (Seg ('   phase elapsed ' + $phaseEl + '   avg ' + ('{0:0.0}' -f $avg) + ' min/scenario') 'Gray'))
        $proj = $elapsedMin + $totalEtaMin
        $pc = if ($proj -gt $CeilingH * 60) { 'Red' } else { 'Green' }
        Row @((Seg '  Projected finish ' 'Gray'), (Seg ((Fmt $proj) + ' of 15h 00m') $pc), (Seg ('   (remaining ' + (Fmt $totalEtaMin) + ')') 'Gray'))
        Row @((Seg '  Last 5 scenarios' 'White'))
        foreach ($r in @($rows | Select-Object -Last 5)) {
            if ($r.Error.StartsWith('llm error')) { $t = "$Bad API failure"; $c = 'Red' }
            elseif ($r.Control) { if ($r.FalseAlarm) { $t = "$Bad false alarm"; $c = 'Red' } else { $t = "$Ok no alarm (correct)"; $c = 'Green' } }
            elseif ($r.Top1 -eq $true) { $t = "$Ok correct"; $c = 'Green' } else { $t = "$Bad wrong"; $c = 'DarkYellow' }
            Row @((Seg ('    {0,-36}' -f $r.Id) 'Gray'), (Seg $t $c))
        }
    } else {
        $ph = if ($status) { $status.phase } else { '' }
        Row @((Seg '  All scenario phases finished or stopped.  Current step: ' 'Green'), (Seg $ph 'Cyan'))
    }
    if ($alerts.Count -gt 0) {
        Rule $ML $MR
        Row @((Seg '  ALERTS' 'Red'))
        foreach ($a in $alerts) { Row @((Seg ('  ! ' + $a.Substring(0, [math]::Min($a.Length, $W - 5))) 'Red')) }
    }
    Rule $BL $BR
    if ($Once) { break }
    Start-Sleep -Seconds 30
}
