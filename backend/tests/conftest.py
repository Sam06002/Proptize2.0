import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.optimizer import PetPoojaOptimizer

@pytest.fixture(scope="module")
def test_client():
    """Fixture for FastAPI test client."""
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="module")
def optimizer():
    """Fixture for PetPoojaOptimizer instance."""
    return PetPoojaOptimizer()

# Test data for parameterized tests
@pytest.fixture
def menu_test_cases():
    return [
        ("add biryani ₹300", {'intent': 'menu', 'entities': ['item', 'price']}),
        ("new dish chicken tikka masala", {'intent': 'menu', 'entities': ['item']}),
        ("add paneer butter masala to main course", {'intent': 'menu', 'entities': ['item', 'category']})
    ]

@pytest.fixture
def inventory_test_cases():
    return [
        ("check rice stock", {'intent': 'inventory', 'entities': ['item']}),
        ("what's the quantity of tomatoes?", {'intent': 'inventory', 'entities': ['item']}),
        ("how much chicken is available?", {'intent': 'inventory', 'entities': ['item']})
    ]

# Performance test configuration
@pytest.fixture
def performance_config():
    return {
        'max_response_time_ms': 100,
        'load_test_iterations': 1000,
        'acceptable_failure_rate': 0.01  # 1% failure rate allowed
    }
