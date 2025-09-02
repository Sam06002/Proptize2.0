# PetPooja Prompt Optimizer

A natural language processing service that optimizes user queries for the PetPooja restaurant management system.

## Features

- Intent classification for 5 main categories: Menu, Inventory, Analytics, Support, and Raw Materials
- Entity extraction with confidence scoring
- Fuzzy string matching for better handling of typos and variations
- Performance optimized for production use
- Comprehensive test suite
- Containerized with Docker

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- Redis (for production)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/petpooja-optimizer.git
   cd petpooja-optimizer/backend
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally

1. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

2. The API will be available at `http://localhost:8000`

### Running with Docker

1. Build and start the services:
   ```bash
   docker-compose up --build
   ```

2. The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

### Main Endpoints

- `POST /optimize` - Optimize a natural language query
- `GET /history` - Get optimization history
- `POST /feedback` - Submit feedback on an optimization
- `GET /analytics` - Get optimization analytics
- `GET /health` - Health check endpoint

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=app --cov-report=term-missing

# Run performance tests
pytest -m performance
```

### Test Structure

- `tests/test_optimizer.py` - Unit tests for the optimizer
- `tests/test_performance.py` - Performance tests
- `tests/data_generator.py` - Test data generation
- `tests/conftest.py` - Test fixtures and configuration

## Deployment

### Production Deployment

1. Set environment variables in `.env`:
   ```
   ENVIRONMENT=production
   DEBUG=false
   REDIS_URL=redis://redis:6379/0
   ```

2. Build and deploy with Docker:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d --build
   ```

## Performance

- Average response time: < 50ms
- P95 response time: < 100ms
- Supports 100+ concurrent requests
- In-memory caching for frequently accessed data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with FastAPI
- Uses Redis for caching
- Inspired by modern NLP applications
