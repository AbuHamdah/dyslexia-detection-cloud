#!/bin/bash
# ============================================================================
# AWS Teardown Script — Remove all deployed resources
# ============================================================================
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_NAME="dyslexia-detection-server"
SECURITY_GROUP_NAME="dyslexia-cloud-sg"
KEY_NAME="${KEY_NAME:-dyslexia-deploy-key}"

echo "=============================================="
echo "  Tearing down AWS resources..."
echo "=============================================="

# Find and terminate instance
INSTANCE_ID=$(aws ec2 describe-instances --region "$REGION" \
    --filters "Name=tag:Name,Values=$INSTANCE_NAME" \
              "Name=instance-state-name,Values=running,pending,stopped" \
    --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || true)

if [ "$INSTANCE_ID" != "None" ] && [ -n "$INSTANCE_ID" ]; then
    echo "Terminating instance: $INSTANCE_ID"
    aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION"
    echo "Waiting for termination..."
    aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" --region "$REGION"
    echo "  Instance terminated."
fi

# Delete security group
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)

if [ "$SG_ID" != "None" ] && [ -n "$SG_ID" ]; then
    echo "Deleting security group: $SG_ID"
    aws ec2 delete-security-group --group-id "$SG_ID" --region "$REGION" 2>/dev/null || true
fi

# Delete key pair
echo "Deleting key pair: $KEY_NAME"
aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION" 2>/dev/null || true
rm -f "${KEY_NAME}.pem"

echo ""
echo "All resources cleaned up."
