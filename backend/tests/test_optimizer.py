import unittest
import time
from fastapi.testclient import TestClient
from app.main import app
from app.optimizer import PetPoojaOptimizer

class TestPetPoojaOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = PetPoojaOptimizer()
        self.test_cases = {
            'menu': [
                "add biryani ₹300",
                "new dish chicken tikka masala for ₹450",
                "add paneer butter masala to main course for ₹350"
            ],
            'inventory': [
                "check rice stock",
                "what's the quantity of tomatoes?",
                "how much chicken is available?"
            ],
            'analytics': [
                "show sales for today",
                "revenue this month",
                "top selling items last week"
            ],
            'support': [
                "I have a billing problem",
                "need callback about my order",
                "help with payment issue"
            ],
            'raw_material': [
                "add 5kg flour to ingredients",
                "update stock for sugar 10kg",
                "add new ingredient: olive oil 2L"
            ]
        }

    def test_intent_classification(self):
        """Test that the correct intent is classified for various inputs."""
        for intent, test_cases in self.test_cases.items():
            with self.subTest(intent=intent):
                for test_case in test_cases:
                    detected_intent, confidence = self.optimizer.classify_intent(test_case)
                    self.assertEqual(detected_intent, intent)
                    self.assertGreaterEqual(confidence, 0.7)  # Minimum confidence threshold

    def test_entity_extraction(self):
        """Test that entities are correctly extracted from queries."""
        test_cases = [
            ("add biryani ₹300", {'item': 'biryani', 'price': '300'}),
            ("check rice stock", {'item': 'rice'}),
            ("sales for today", {'time_period': 'today'}),
        ]
        
        for query, expected_entities in test_cases:
            with self.subTest(query=query):
                intent, _ = self.optimizer.classify_intent(query)
                entities, _ = self.optimizer.extract_entities(query, intent)
                
                for key, value in expected_entities.items():
                    self.assertIn(key, entities)
                    self.assertEqual(entities[key], value)

    def test_edge_cases(self):
        """Test handling of edge cases."""
        # Empty input
        with self.assertRaises(ValueError):
            self.optimizer.optimize_prompt("")
            
        # Gibberish input
        result = self.optimizer.optimize_prompt("asdf1234!@#$")
        self.assertEqual(result['intent'], 'unknown')
        self.assertLess(result['confidence'], 0.5)

    def test_response_time(self):
        """Test that response time is within acceptable limits."""
        test_query = "add butter chicken for ₹450 to the menu"
        
        # Warm-up
        self.optimizer.optimize_prompt(test_query)
        
        # Measure response time
        start_time = time.time()
        for _ in range(100):
            self.optimizer.optimize_prompt(test_query)
        avg_time = (time.time() - start_time) * 1000 / 100  # in ms
        
        self.assertLess(avg_time, 100)  # Should be under 100ms

class TestAPIEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
    def test_optimize_endpoint(self):
        """Test the main optimization endpoint."""
        test_cases = [
            ("add paneer tikka ₹350", 200, 'menu'),
            ("check stock", 200, 'inventory'),
            ("show me sales", 200, 'analytics'),
            ("help me", 200, 'support'),
            ("add ingredient", 200, 'raw_material')
        ]
        
        for query, status_code, expected_intent in test_cases:
            with self.subTest(query=query):
                response = self.client.post("/optimize", json={"query": query})
                self.assertEqual(response.status_code, status_code)
                data = response.json()
                self.assertEqual(data['intent'], expected_intent)
                self.assertIn('optimized_prompt', data)
    
    def test_analytics_endpoint(self):
        """Test the analytics endpoint."""
        response = self.client.get("/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('total_optimizations', data)
        self.assertIn('intent_distribution', data)
        
    def test_invalid_input(self):
        """Test handling of invalid input."""
        response = self.client.post("/optimize", json={"invalid": "input"})
        self.assertEqual(response.status_code, 422)  # Validation error

if __name__ == '__main__':
    unittest.main()
