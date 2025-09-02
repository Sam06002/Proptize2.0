from optimizer import PetPoojaOptimizer

def test_optimizer():
    optimizer = PetPoojaOptimizer()
    
    test_cases = [
        "add chicken curry ₹200",
        "check tomato stock",
        "show today sales",
        "new pasta dish for 350 in main course",
        "inventory level for chicken tikka",
        "sales report for last week",
        "help with login issue",
        "need more tomatoes"
    ]
    
    for query in test_cases:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        result = optimizer.optimize_prompt(query)
        
        print(f"INTENT: {result['intent']}")
        print(f"OPTIMIZED: {result['optimized']}")
        print("ENTITIES:")
        for key, value in result['entities'].items():
            print(f"  - {key}: {value}")
        
        if result['missing_entities']:
            print(f"MISSING: {', '.join(result['missing_entities'])}")
        
        if result['needs_user_input']:
            print("NEEDS USER INPUT: Yes")
        
        print(f"{'='*80}")

if __name__ == "__main__":
    test_optimizer()
