#!/bin/bash
# ============================================================================
# AWS EC2 Deployment Script — Dyslexia Detection Cloud System
# ============================================================================
# This script automates deploying the system to an AWS EC2 instance.
#
# PREREQUISITES:
#   1. AWS CLI installed and configured:  aws configure
#   2. An AWS account with free tier eligibility
#   3. An SSH key pair created in AWS (or use the script to create one)
#
# USAGE:
#   chmod +x deploy_ec2.sh
#   ./deploy_ec2.sh
# ============================================================================

set -euo pipefail

# ── Configuration ──
REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.micro}"        # Free-tier eligible
AMI_ID=""                                          # Auto-detected below
KEY_NAME="${KEY_NAME:-dyslexia-deploy-key}"
SECURITY_GROUP_NAME="dyslexia-cloud-sg"
INSTANCE_NAME="dyslexia-detection-server"
PROJECT_DIR="cloud_system"

echo "=============================================="
echo "  Dyslexia Detection — AWS EC2 Deployment"
echo "=============================================="
echo ""

# ── Step 1: Verify AWS CLI ──
echo "[1/8] Checking AWS CLI..."
if ! command -v aws &> /dev/null; then
    echo "ERROR: AWS CLI not found. Install it first:"
    echo "  https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)
if [ -z "$AWS_ACCOUNT" ]; then
    echo "ERROR: AWS CLI not configured. Run:  aws configure"
    exit 1
fi
echo "  AWS Account: $AWS_ACCOUNT"
echo "  Region:      $REGION"

# ── Step 2: Find latest Amazon Linux 2023 AMI ──
echo "[2/8] Finding latest Amazon Linux 2023 AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023.*-x86_64" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)
echo "  AMI: $AMI_ID"

# ── Step 3: Create/verify SSH key pair ──
echo "[3/8] Setting up SSH key pair '$KEY_NAME'..."
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" &>/dev/null; then
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > "${KEY_NAME}.pem"
    chmod 400 "${KEY_NAME}.pem"
    echo "  Created new key pair → ${KEY_NAME}.pem"
else
    echo "  Key pair '$KEY_NAME' already exists"
fi

# ── Step 4: Create Security Group ──
echo "[4/8] Creating security group '$SECURITY_GROUP_NAME'..."
VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' --output text)

SG_ID=$(aws ec2 describe-security-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [ "$SG_ID" = "None" ] || [ -z "$SG_ID" ]; then
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SECURITY_GROUP_NAME" \
        --description "Dyslexia Detection Cloud System" \
        --vpc-id "$VPC_ID" \
        --region "$REGION" \
        --query 'GroupId' --output text)

    # Allow SSH (22), HTTP (80), API (8000), Dashboard (8501)
    for PORT in 22 80 8000 8501; do
        aws ec2 authorize-security-group-ingress \
            --group-id "$SG_ID" \
            --protocol tcp \
            --port "$PORT" \
            --cidr 0.0.0.0/0 \
            --region "$REGION" 2>/dev/null || true
    done
    echo "  Created security group: $SG_ID"
else
    echo "  Security group already exists: $SG_ID"
fi

# ── Step 5: Launch EC2 Instance ──
echo "[5/8] Launching EC2 instance ($INSTANCE_TYPE)..."

# Check if instance already exists
EXISTING_ID=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
              "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || true)

if [ "$EXISTING_ID" != "None" ] && [ -n "$EXISTING_ID" ]; then
    INSTANCE_ID="$EXISTING_ID"
    echo "  Instance already running: $INSTANCE_ID"
else
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$INSTANCE_TYPE" \
        --key-name "$KEY_NAME" \
        --security-group-ids "$SG_ID" \
        --region "$REGION" \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":20,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME}]" \
        --user-data file://user_data.sh \
        --query 'Instances[0].InstanceId' \
        --output text)
    echo "  Launched instance: $INSTANCE_ID"
fi

# ── Step 6: Wait for instance to be running ──
echo "[6/8] Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids "$INSTANCE_ID" \
    --region "$REGION" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "  Public IP: $PUBLIC_IP"

# ── Step 7: Wait for SSH availability ──
echo "[7/8] Waiting for SSH to be available..."
for i in {1..30}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        -i "${KEY_NAME}.pem" ec2-user@"$PUBLIC_IP" "echo ok" &>/dev/null; then
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 10
done

# ── Step 8: Deploy application ──
echo "[8/8] Deploying application..."

# Create project archive (excluding large data files)
echo "  Creating deployment archive..."
cd "$(dirname "$0")/../../.."
tar czf /tmp/dyslexia-deploy.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='ds003126_raw' \
    --exclude='ds006239_raw' \
    --exclude='.venv' \
    cloud_system/

# Upload to EC2
echo "  Uploading to EC2..."
scp -o StrictHostKeyChecking=no -i "cloud_system/deploy/aws/${KEY_NAME}.pem" \
    /tmp/dyslexia-deploy.tar.gz ec2-user@"$PUBLIC_IP":~/

# Extract and start
echo "  Starting Docker containers..."
ssh -o StrictHostKeyChecking=no -i "cloud_system/deploy/aws/${KEY_NAME}.pem" ec2-user@"$PUBLIC_IP" << 'REMOTE'
cd ~
tar xzf dyslexia-deploy.tar.gz
cd cloud_system/docker
sudo docker compose up --build -d
echo "Containers started!"
sudo docker compose ps
REMOTE

echo ""
echo "=============================================="
echo "  DEPLOYMENT COMPLETE!"
echo "=============================================="
echo ""
echo "  Dashboard:   http://$PUBLIC_IP"
echo "  API Docs:    http://$PUBLIC_IP:8000/docs"
echo "  Health:      http://$PUBLIC_IP/health"
echo ""
echo "  SSH access:  ssh -i ${KEY_NAME}.pem ec2-user@$PUBLIC_IP"
echo ""
echo "  To stop:     ssh -i ${KEY_NAME}.pem ec2-user@$PUBLIC_IP 'cd cloud_system/docker && sudo docker compose down'"
echo "  To destroy:  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
