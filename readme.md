# ECG Labeling System

A full-stack web application for labeling and managing ECG (Electrocardiogram) data. This system allows medical professionals to upload ECG data, label it manually or with AI assistance, and visualize the results.

## Features

- **ECG Data Upload**: Support for CSV and Excel files
- **Manual Labeling**: Human experts can label ECG records
- **AI-Assisted Labeling**: Integrated machine learning models for automated labeling
- **Data Visualization**: Interactive charts using ECharts and Plotly
- **User Management**: Role-based access (Admin, Doctor, Patient)
- **Real-time Processing**: Redis-backed caching for performance
- **Model Management**: Deploy and manage multiple ML models

## Tech Stack

### Backend
- **Django 5.2.5**: Web framework
- **Django REST Framework**: API development
- **SQLite**: Database (default, can be changed to PostgreSQL)
- **Redis**: Caching and session storage
- **PyTorch**: Machine learning framework
- **XGBoost**: Gradient boosting for ML models

### Frontend
- **React 19**: UI framework
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Styling
- **ECharts & Plotly**: Data visualization
- **Axios**: HTTP client

## Prerequisites

- Python 3.11+
- Node.js 18+
- Redis (local installation or Docker)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Kaditya67/ecg-heartcare-ai.git
cd ecg-heartcare-ai
```

### 2. Backend Setup

#### Create Virtual Environment
```bash
cd backend
python -m venv venv
```

#### Activate Virtual Environment
- **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate
  ```
- **Windows (PowerShell)**:
  ```powershell
  venv\Scripts\Activate.ps1
  ```
- **Linux/Mac**:
  ```bash
  source venv/bin/activate
  ```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Database Setup
```bash
python manage.py migrate
```

#### Create Superuser (Optional)
```bash
python manage.py createsuperuser
```

### 3. Frontend Setup

```bash
cd ../frontend
npm install
```

### 4. Redis Setup

#### Option 1: Local Installation
- **Windows**: Download and install Redis from [redis.io](https://redis.io/download)
- **Linux/Mac**: Use package manager or download from redis.io

Start Redis:
```bash
redis-server
```

#### Option 2: Using Docker
```bash
docker run --name ecg-redis -p 6379:6379 -d redis
```

## Running the Application

### Start Backend
```bash
cd backend
# Activate venv if not already
python manage.py runserver
```
Backend runs at: `http://127.0.0.1:8000/`

### Start Frontend
```bash
cd frontend
npm run dev
```
Frontend runs at: `http://localhost:5173/`

### Start Redis (if not using Docker)
Ensure Redis is running on port 6379.

## Configuration

### Environment Variables
Create a `.env` file in the backend directory:

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///db.sqlite3
REDIS_URL=redis://127.0.0.1:6379/1
```

### Django Settings
Key settings in `backend/core/settings.py`:
- `ALLOWED_HOSTS`: Add your domain in production
- `CORS_ALLOWED_ORIGINS`: Configure for frontend URL
- `CACHES`: Redis configuration
- `DATABASES`: Database configuration

## Model Setup

The system includes pre-trained ML models. Place model files in `backend/api/models/`:

- `best_ecg_1dcnn.pth`: PyTorch CNN model
- `best_bert_ecg.pth`: BERT-based model
- `ecg_xgboost.pth`: XGBoost model

Models are automatically loaded on startup.

## API Documentation

### Authentication
The API uses JWT tokens for authentication.

### Key Endpoints
- `POST /api/auth/login/`: User login
- `GET /api/ecg-records/`: List ECG records
- `POST /api/upload/`: Upload ECG data
- `GET /api/models/`: List available models

## Development

### Running Tests
```bash
cd backend
python manage.py test
```

### Linting
```bash
cd frontend
npm run lint
```

### Building for Production
```bash
cd frontend
npm run build
```

## Deployment

### Backend Deployment
1. Set `DEBUG=False` in settings
2. Configure production database
3. Use a WSGI server like Gunicorn
4. Set up reverse proxy (nginx)

### Frontend Deployment
1. Build the application: `npm run build`
2. Serve static files from `dist/` directory
3. Configure routing for SPA

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   - Ensure Redis is running on port 6379
   - Check `REDIS_URL` in settings

2. **Model Loading Errors**
   - Verify model files exist in `api/models/`
   - Check file permissions

3. **CORS Errors**
   - Add frontend URL to `CORS_ALLOWED_ORIGINS`

4. **Database Errors**
   - Run `python manage.py migrate`
   - Check database file permissions

### Logs
- Backend logs: Check Django console output
- Frontend logs: Browser developer tools

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Demo

[▶️ Watch Demo Video](https://drive.google.com/file/d/1giqB2EaELZ_RCoDTJQc_9VJoIJpz4lQr/preview)

## Contact

For questions or support, please open an issue on GitHub.