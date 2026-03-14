<#
.SYNOPSIS
    Deploy Dyslexia Detection System to AWS EC2 from Windows.
.DESCRIPTION
    PowerShell wrapper that deploys the Docker-based system to AWS.
    Requires: AWS CLI v2, ssh (OpenSSH client).
.EXAMPLE
    .\deploy_windows.ps1
    .\deploy_windows.ps1 -Region us-west-2 -InstanceType t3.small
#>
param(
    [string]$Region      = "us-east-1",
    [string]$InstanceType = "t3.micro",
    [string]$KeyName     = "dyslexia-deploy-key"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$DeployDir   = $PSScriptRoot
$InstanceName       = "dyslexia-detection-server"
$SecurityGroupName  = "dyslexia-cloud-sg"

Write-Host "=============================================="  -ForegroundColor Cyan
Write-Host "  Dyslexia Detection — AWS EC2 Deployment"      -ForegroundColor Cyan
Write-Host "=============================================="  -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Verify AWS CLI ──
Write-Host "[1/8] Checking AWS CLI..." -ForegroundColor Yellow
try {
    $account = aws sts get-caller-identity --query Account --output text 2>&1
    Write-Host "  AWS Account: $account"
    Write-Host "  Region:      $Region"
} catch {
    Write-Host "ERROR: AWS CLI not configured. Run 'aws configure' first." -ForegroundColor Red
    exit 1
}

# ── Step 2: Find AMI ──
Write-Host "[2/8] Finding latest Amazon Linux 2023 AMI..." -ForegroundColor Yellow
$AmiId = aws ec2 describe-images `
    --region $Region `
    --owners amazon `
    --filters "Name=name,Values=al2023-ami-2023.*-x86_64" "Name=state,Values=available" `
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' `
    --output text
Write-Host "  AMI: $AmiId"

# ── Step 3: SSH Key Pair ──
Write-Host "[3/8] Setting up SSH key pair..." -ForegroundColor Yellow
$keyExists = aws ec2 describe-key-pairs --key-names $KeyName --region $Region 2>&1
if ($LASTEXITCODE -ne 0) {
    $keyMaterial = aws ec2 create-key-pair `
        --key-name $KeyName `
        --region $Region `
        --query 'KeyMaterial' `
        --output text
    $keyPath = Join-Path $DeployDir "$KeyName.pem"
    $keyMaterial | Out-File -FilePath $keyPath -Encoding ascii -NoNewline
    Write-Host "  Created key pair -> $keyPath"
} else {
    Write-Host "  Key pair '$KeyName' already exists"
}
$keyPath = Join-Path $DeployDir "$KeyName.pem"

# ── Step 4: Security Group ──
Write-Host "[4/8] Creating security group..." -ForegroundColor Yellow
$VpcId = aws ec2 describe-vpcs --region $Region `
    --filters "Name=isDefault,Values=true" `
    --query 'Vpcs[0].VpcId' --output text

$SgId = aws ec2 describe-security-groups --region $Region `
    --filters "Name=group-name,Values=$SecurityGroupName" `
    --query 'SecurityGroups[0].GroupId' --output text 2>&1

if ($SgId -eq "None" -or $LASTEXITCODE -ne 0) {
    $SgId = aws ec2 create-security-group `
        --group-name $SecurityGroupName `
        --description "Dyslexia Detection Cloud System" `
        --vpc-id $VpcId `
        --region $Region `
        --query 'GroupId' --output text

    foreach ($port in @(22, 80, 8000, 8501)) {
        aws ec2 authorize-security-group-ingress `
            --group-id $SgId `
            --protocol tcp `
            --port $port `
            --cidr "0.0.0.0/0" `
            --region $Region 2>$null
    }
    Write-Host "  Created: $SgId"
} else {
    Write-Host "  Exists: $SgId"
}

# ── Step 5: Launch instance ──
Write-Host "[5/8] Launching EC2 instance ($InstanceType)..." -ForegroundColor Yellow

$ExistingId = aws ec2 describe-instances --region $Region `
    --filters "Name=tag:Name,Values=$InstanceName" "Name=instance-state-name,Values=running,pending" `
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>&1

if ($ExistingId -ne "None" -and $ExistingId -ne "") {
    $InstanceId = $ExistingId
    Write-Host "  Already running: $InstanceId"
} else {
    $userDataPath = Join-Path $DeployDir "user_data.sh"
    $InstanceId = aws ec2 run-instances `
        --image-id $AmiId `
        --instance-type $InstanceType `
        --key-name $KeyName `
        --security-group-ids $SgId `
        --region $Region `
        --block-device-mappings '[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"VolumeSize\":20,\"VolumeType\":\"gp3\"}}]' `
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$InstanceName}]" `
        --user-data "file://$userDataPath" `
        --query 'Instances[0].InstanceId' `
        --output text
    Write-Host "  Launched: $InstanceId"
}

# ── Step 6: Wait ──
Write-Host "[6/8] Waiting for instance..." -ForegroundColor Yellow
aws ec2 wait instance-running --instance-ids $InstanceId --region $Region

$PublicIp = aws ec2 describe-instances `
    --instance-ids $InstanceId `
    --region $Region `
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
Write-Host "  Public IP: $PublicIp" -ForegroundColor Green

# ── Step 7: Wait for SSH ──
Write-Host "[7/8] Waiting for SSH (up to 5 min)..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i $keyPath ec2-user@$PublicIp "echo ok" 2>$null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    } catch {}
    Write-Host "  Waiting... ($($i+1)/30)"
    Start-Sleep -Seconds 10
}
if (-not $ready) {
    Write-Host "WARNING: SSH not ready yet. The instance may still be initializing." -ForegroundColor Yellow
    Write-Host "  Wait a few minutes and run the upload step manually."
}

# ── Step 8: Upload & Deploy ──
Write-Host "[8/8] Deploying application..." -ForegroundColor Yellow

# Create deployment archive
$archivePath = Join-Path $env:TEMP "dyslexia-deploy.tar.gz"
Push-Location $ProjectRoot
tar czf $archivePath `
    --exclude='__pycache__' `
    --exclude='*.pyc' `
    --exclude='.git' `
    --exclude='ds003126_raw' `
    --exclude='ds006239_raw' `
    --exclude='.venv' `
    cloud_system/
Pop-Location

# Upload
Write-Host "  Uploading (~may take a minute)..."
scp -o StrictHostKeyChecking=no -i $keyPath $archivePath "ec2-user@${PublicIp}:~/"

# Start containers
Write-Host "  Starting Docker containers..."
ssh -o StrictHostKeyChecking=no -i $keyPath ec2-user@$PublicIp @"
cd ~
tar xzf dyslexia-deploy.tar.gz
cd cloud_system/docker
sudo docker compose up --build -d
echo 'Done!'
sudo docker compose ps
"@

Write-Host ""
Write-Host "=============================================="  -ForegroundColor Green
Write-Host "  DEPLOYMENT COMPLETE!"                          -ForegroundColor Green
Write-Host "=============================================="  -ForegroundColor Green
Write-Host ""
Write-Host "  Dashboard:  http://$PublicIp"                  -ForegroundColor Cyan
Write-Host "  API Docs:   http://$PublicIp`:8000/docs"       -ForegroundColor Cyan
Write-Host "  Health:     http://$PublicIp/health"           -ForegroundColor Cyan
Write-Host ""
Write-Host "  SSH:        ssh -i $keyPath ec2-user@$PublicIp"
Write-Host "  Destroy:    aws ec2 terminate-instances --instance-ids $InstanceId --region $Region"
