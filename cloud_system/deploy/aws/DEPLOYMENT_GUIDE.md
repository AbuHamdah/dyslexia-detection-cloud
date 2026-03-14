# AWS Deployment Guide — Dyslexia Detection Cloud System

## Recommended: AWS EC2 Free Tier

**Why AWS EC2 for this project:**
- **Free tier**: t3.micro (1 vCPU, 1 GB RAM) free for 12 months
- **Docker-native**: your existing `docker-compose.yml` works as-is
- **Full control**: SSH access, custom ports, persistent storage
- **Scalable**: upgrade instance type later if needed

## Architecture on AWS

```
Internet
   │
   ▼
┌──────────────────────────────────────┐
│  AWS EC2 (t3.micro — Free Tier)     │
│                                      │
│  ┌─────────┐  ┌──────────────────┐  │
│  │  NGINX  │──│  FastAPI (8000)  │  │
│  │  (:80)  │  │  3D-CNN / LSTM   │  │
│  └────┬────┘  └──────────────────┘  │
│       │                              │
│  ┌────▼──────────────────────────┐  │
│  │  Streamlit Dashboard (8501)   │  │
│  └───────────────────────────────┘  │
│                                      │
│  20 GB gp3 EBS volume              │
│  2 GB swap (for TensorFlow)         │
└──────────────────────────────────────┘
```

## Prerequisites

1. **AWS Account** — Sign up at https://aws.amazon.com (free tier eligible)
2. **AWS CLI v2** — Install from https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
3. **Configure CLI**:
   ```powershell
   aws configure
   # Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)
   ```

## Quick Deploy (Windows)

From PowerShell, run the one-click deployment script:

```powershell
cd cloud_system\deploy\aws
.\deploy_windows.ps1
```

This will:
1. Create an SSH key pair
2. Create a security group (ports 22, 80, 8000, 8501)
3. Launch a t3.micro EC2 instance with Amazon Linux 2023
4. Install Docker & Docker Compose on the instance
5. Upload your code + models
6. Build & start all containers

**Total time: ~5-10 minutes**

## Quick Deploy (Linux/Mac)

```bash
cd cloud_system/deploy/aws
chmod +x deploy_ec2.sh user_data.sh
./deploy_ec2.sh
```

## After Deployment

| Service | URL |
|---------|-----|
| **Dashboard** | `http://<PUBLIC_IP>` |
| **API Docs** | `http://<PUBLIC_IP>:8000/docs` |
| **Health Check** | `http://<PUBLIC_IP>/health` |

## SSH into Server

```bash
ssh -i dyslexia-deploy-key.pem ec2-user@<PUBLIC_IP>
```

## Useful Commands on Server

```bash
# View running containers
cd ~/cloud_system/docker
sudo docker compose ps

# View API logs
sudo docker compose logs api -f

# View dashboard logs
sudo docker compose logs dashboard -f

# Restart everything
sudo docker compose restart

# Rebuild after code changes
sudo docker compose up --build -d
```

## CI/CD with GitHub Actions

A workflow file is at `.github/workflows/deploy-aws.yml`. To enable:

1. Push your code to GitHub
2. Go to repo → Settings → Secrets → Actions
3. Add these secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `EC2_SSH_PRIVATE_KEY` (contents of `dyslexia-deploy-key.pem`)
4. Every push to `main` that changes `cloud_system/` will auto-deploy

## Cost Breakdown (Free Tier)

| Resource | Free Tier | After Free Tier |
|----------|-----------|-----------------|
| EC2 t3.micro | 750 hrs/month (12 months) | ~$8.50/month |
| EBS 20 GB gp3 | 30 GB free | ~$1.60/month |
| Data transfer | 100 GB out/month | $0.09/GB |
| **Total** | **$0/month** | **~$10/month** |

## Upgrading for Better Performance

If t3.micro is too slow for TensorFlow inference:

| Instance | vCPU | RAM | Cost/month | Notes |
|----------|------|-----|------------|-------|
| t3.micro | 2 | 1 GB | Free | Basic, uses swap |
| t3.small | 2 | 2 GB | ~$15 | Comfortable for inference |
| t3.medium | 2 | 4 GB | ~$30 | Smooth TF inference |
| g4dn.xlarge | 4 | 16 GB + GPU | ~$380 | If you need GPU |

To upgrade:
```bash
aws ec2 stop-instances --instance-ids <ID>
aws ec2 modify-instance-attribute --instance-id <ID> --instance-type t3.small
aws ec2 start-instances --instance-ids <ID>
```

## Tear Down

Remove all AWS resources:

**Windows:**
```powershell
aws ec2 terminate-instances --instance-ids <INSTANCE_ID> --region us-east-1
```

**Linux/Mac:**
```bash
cd cloud_system/deploy/aws
./teardown.sh
```

## Troubleshooting

### "Cannot allocate memory" during Docker build
The t3.micro has only 1 GB RAM. The user_data script creates a 2 GB swap file. If you still see memory errors:
```bash
sudo fallocate -l 4G /swapfile2
sudo chmod 600 /swapfile2
sudo mkswap /swapfile2
sudo swapon /swapfile2
```

### Models not loading
Ensure the `.keras` files are in `cloud_system/saved_models/`:
```bash
ls -la ~/cloud_system/saved_models/
```

### Dashboard can't reach API
Check that all containers are running:
```bash
cd ~/cloud_system/docker
sudo docker compose ps
sudo docker compose logs api
```

### Port not accessible
Verify security group allows the port:
```bash
aws ec2 describe-security-groups --group-names dyslexia-cloud-sg
```
