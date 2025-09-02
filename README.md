# PetPooja Prompt Optimizer

A tool to transform natural language queries into optimized prompts for PetPooja Agent.

## Project Structure

```
petpooja-prompt-optimizer/
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py       # FastAPI application
│   │   ├── optimizer.py  # Core optimization logic
│   │   └── models.py     # Pydantic models
│   └── requirements.txt  # Python dependencies
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── App.jsx       # Main React component
│   │   └── index.js      # Entry point
│   └── package.json      # Node.js dependencies
└── README.md             # This file
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   cd app
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`
   - API docs: `http://localhost:8000/docs`
   - Redoc: `http://localhost:8000/redoc`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   The app will be available at `http://localhost:3001`

## API Endpoints

- `POST /optimize` - Optimizes a natural language query
  - Request body: `{ "query": "your natural language query", "function_type": "menu|inventory|analytics|support|raw_material" }`
  - Response: Optimized prompt and parameters

## Development

- Backend: Python 3.8+ with FastAPI
- Frontend: React with Material-UI
- No external APIs required

## License

MIT
