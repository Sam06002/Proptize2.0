import random
from faker import Faker
from typing import List, Dict, Any

class TestDataGenerator:
    """Generate test data for PetPooja Optimizer."""
    
    def __init__(self):
        self.fake = Faker()
        self.menu_items = [
            "biryani", "butter chicken", "paneer tikka", "dal makhani",
            "naan", "roti", "gulab jamun", "rasmalai"
        ]
        self.categories = ["main course", "starter", "dessert", "bread"]
        self.ingredients = ["rice", "chicken", "paneer", "flour", "sugar", "milk"]
        self.units = ["kg", "g", "L", "ml", "dozen"]
    
    def generate_menu_query(self) -> Dict[str, str]:
        """Generate a test query for menu operations."""
        item = random.choice(self.menu_items)
        price = random.randint(100, 1000)
        category = random.choice(self.categories)
        
        templates = [
            f"add {item} for ₹{price}",
            f"new dish: {item} in {category} for {price} rupees",
            f"please add {item} to the {category} section priced at ₹{price}"
        ]
        
        return {
            'query': random.choice(templates),
            'expected_intent': 'menu',
            'expected_entities': ['item', 'price', 'category']
        }
    
    def generate_inventory_query(self) -> Dict[str, Any]:
        """Generate a test query for inventory operations."""
        item = random.choice(self.ingredients)
        quantity = random.randint(1, 20)
        unit = random.choice(self.units)
        
        templates = [
            f"check {item} stock",
            f"what's the quantity of {item}?",
            f"how much {item} is available?",
            f"update {item} stock to {quantity} {unit}"
        ]
        
        return {
            'query': random.choice(templates),
            'expected_intent': 'inventory',
            'expected_entities': ['item']
        }
    
    def generate_analytics_query(self) -> Dict[str, Any]:
        """Generate a test query for analytics."""
        metrics = ["sales", "revenue", "orders", "customers"]
        time_periods = ["today", "yesterday", "this week", "this month", "this year"]
        
        templates = [
            f"show {random.choice(metrics)} for {random.choice(time_periods)}",
            f"what are the {random.choice(metrics)} for {random.choice(time_periods)}?",
            f"generate a report on {random.choice(metrics)} {random.choice(time_periods)}"
        ]
        
        return {
            'query': random.choice(templates),
            'expected_intent': 'analytics',
            'expected_entities': ['metric', 'time_period']
        }
    
    def generate_support_query(self) -> Dict[str, Any]:
        """Generate a test query for support."""
        issues = [
            "billing problem", "order issue", "payment failure",
            "technical problem", "refund request"
        ]
        
        templates = [
            f"I have a {random.choice(issues)}",
            f"need help with {random.choice(issues)}",
            f"please assist me with {random.choice(issues)}",
            f"I'm having trouble with {random.choice(issues)}"
        ]
        
        return {
            'query': random.choice(templates),
            'expected_intent': 'support',
            'expected_entities': ['issue_type']
        }
    
    def generate_raw_material_query(self) -> Dict[str, Any]:
        """Generate a test query for raw materials."""
        ingredient = random.choice(self.ingredients)
        quantity = random.randint(1, 50)
        unit = random.choice(self.units)
        
        templates = [
            f"add {quantity}{unit} {ingredient} to inventory",
            f"update {ingredient} stock to {quantity}{unit}",
            f"we need to order {quantity}{unit} of {ingredient}"
        ]
        
        return {
            'query': random.choice(templates),
            'expected_intent': 'raw_material',
            'expected_entities': ['ingredient', 'quantity', 'unit']
        }
    
    def generate_test_cases(self, num_cases: int = 10) -> List[Dict[str, Any]]:
        """Generate a list of test cases."""
        generators = [
            self.generate_menu_query,
            self.generate_inventory_query,
            self.generate_analytics_query,
            self.generate_support_query,
            self.generate_raw_material_query
        ]
        
        test_cases = []
        for _ in range(num_cases):
            generator = random.choice(generators)
            test_cases.append(generator())
        
        return test_cases

if __name__ == "__main__":
    # Example usage
    generator = TestDataGenerator()
    test_cases = generator.generate_test_cases(5)
    
    print("Generated Test Cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Query: {case['query']}")
        print(f"  Expected Intent: {case['expected_intent']}")
        print(f"  Expected Entities: {', '.join(case['expected_entities'])}")
