# ECG Labeling System

This project is a **full-stack web application** for labeling and managing ECG data.
It consists of the following components:

- **Backend**: Django (REST API, data processing, database management)
- **Frontend**: React (user interface for labeling and visualization)

***

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Kaditya67/ecg-heartcare-ai.git
```


### 2. Setup Backend (Django)

Move into the backend folder:

```bash
cd backend
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run migrations and start the server:

```bash
python manage.py migrate
python manage.py runserver
```

By default, the backend runs at:
`http://127.0.0.1:8000/`

***

### 3. Setup Frontend (React)

Move into the frontend folder:

```bash
cd ../frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

By default, the frontend runs at:
`http://localhost:5173/`

*** 

### 4. Starting Redis

#### Option 1: If Redis is installed locally on your system

For Linux/macOS:

```bash
redis-server
```

For Windows (with WSL or Redis installed):

```bash
redis-server.exe
```

Or via WSL:

```bash
wsl redis-server
```


#### Option 2: Using Docker

If you have Docker installed, you can run Redis with:

```bash
docker run --name ecg-redis -p 6379:6379 -d redis
```


**Important**:

- Make sure your Django (backend) settings point to the correct Redis URL, e.g., `redis://127.0.0.1:6379` if running locally.

***

## Full Local Development Workflow

1. **Start Redis**
2. **Start Django backend** (`cd backend && python manage.py runserver`)
3. **Start React frontend** (`cd frontend && npm start`)



## 📂 Project Structure

```
ecg-labeling-system/
│
├── backend/          # Django backend (APIs, models, database)
├── frontend/         # React frontend (UI for labeling, visualization)
├── readme.md         # Project documentation (this file)
```


## Tech Stack

- **Backend**: Django, Django REST Framework
- **Frontend**: React, Axios
- **Database**: SQLite (default, can be switched to PostgreSQL/MySQL)
- **Other Tools**: npm/yarn, Python venv

***

## Future Improvements

- Model Integration 
- Real time collaboration
- More charts

### Demo of the System
[▶️ Watch Demo Video (Google Drive)](https://drive.google.com/file/d/1giqB2EaELZ_RCoDTJQc_9VJoIJpz4lQr/preview)
