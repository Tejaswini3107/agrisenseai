# AgriSense AI

AgriSense AI is a farmer-focused platform with three pieces:

- a FastAPI backend for weather, crop, pest, irrigation, and chat services
- a Next.js admin website for monitoring and operations
- a React Native mobile app for farmer-facing workflows

The repo also includes Docker Compose files and a GitHub Actions pipeline for linting, testing, scanning, building, pushing, and deploying.

## What this repo contains

- [agrisense-backend/](agrisense-backend/) - backend API, data pipeline, ML/rule services, and backend-specific docs
- [agrisense-admin-website/](agrisense-admin-website/) - admin dashboard
- [agrisense-mobile/](agrisense-mobile/) - mobile app
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) - CI/CD workflow
- [docker-compose.yml](docker-compose.yml) - local development stack
- [docker-compose.prod.yml](docker-compose.prod.yml) - production stack for EC2

## Quick Start

### Backend

```bash
cd agrisense-backend
pip install -r requirements.txt
python app/main.py
```

### Admin website

```bash
cd agrisense-admin-website
npm install
npm run dev
```

### Mobile app

```bash
cd agrisense-mobile
npm install
npm run start
```

### Docker Compose

```bash
docker compose up --build
```

## Core capabilities

- Crop recommendation support
- Disease and pest risk detection
- Irrigation planning and advice
- Yield and crop health forecasting
- Climate and weather risk analysis
- Bilingual chatbot support
- Farmer authentication
- Admin dashboard monitoring

## Deployment

Deployment is handled by [.github/workflows/deploy.yml](.github/workflows/deploy.yml) on pushes to `main`.

The workflow does the following:

1. Lints backend and admin code.
2. Runs backend tests.
3. Scans for dependency issues and secrets.
4. Builds Docker images.
5. Pushes images to ECR.
6. Copies `docker-compose.prod.yml` to EC2.
7. Starts the production stack.
8. Runs smoke tests.

## Environment variables

Common variables used across the repo:

- `DB_HOST`
- `DB_USER`
- `DB_PASSWORD`
- `OPENWEATHER_API_KEY`
- `HF_TOKEN`
- `FIREBASE_SERVICE_ACCOUNT_JSON_B64`
- `GOOGLE_OAUTH_WEB_CLIENT_ID`
- `GOOGLE_OAUTH_IOS_CLIENT_ID`
- `ONESIGNAL_APP_ID`
- `ONESIGNAL_API_KEY`
- `EC2_HOST`
- `EC2_SSH_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

Store secrets in local `.env` files for development and in GitHub Secrets for deployment.

## Documentation

- [agrisense-backend/README.md](agrisense-backend/README.md) - full backend documentation
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) - deployment workflow
- [docker-compose.yml](docker-compose.yml) - local compose stack
- [docker-compose.prod.yml](docker-compose.prod.yml) - production compose stack

## Testing

```bash
cd agrisense-backend
pytest tests -v

cd ../agrisense-admin-website
npm run lint
npm run type-check

cd ../agrisense-mobile
npm run test
npm run lint
```
