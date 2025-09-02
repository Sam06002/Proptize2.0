import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

class PetPoojaOptimizer:
    def __init__(self):
        # Load templates
        templates_path = Path(__file__).parent / "data" / "templates.json"
        with open(templates_path, 'r') as f:
            self.templates = json.load(f)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        
        # Define intent keywords
        self.intent_keywords = {
            'menu': ['add', 'new', 'menu', 'dish', 'food', 'item', 'price'],
            'inventory': ['stock', 'inventory', 'check', 'level', 'quantity', 'available'],
            'analytics': ['sales', 'revenue', 'analytics', 'report', 'insight', 'trend'],
            'support': ['help', 'problem', 'issue', 'ticket', 'error', 'support'],
            'raw_material': ['ingredient', 'raw', 'material', 'supply', 'purchase']
        }
    
    def classify_intent(self, text: str) -> str:
        """Classify the intent of the user query."""
        text_lower = text.lower()
        scores = {intent: 0 for intent in self.intent_keywords}
        
        # Score each intent based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[intent] += 1
        
        # Get the intent with the highest score
        best_intent = max(scores, key=scores.get)
        
        # If no clear intent, try fuzzy matching
        if scores[best_intent] == 0:
            for intent, keywords in self.intent_keywords.items():
                for keyword in keywords:
                    if fuzz.partial_ratio(keyword, text_lower) > 80:
                        scores[intent] += 1
            best_intent = max(scores, key=scores.get)
        
        return best_intent if scores[best_intent] > 0 else 'support'  # Default to support
    
    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """Extract relevant entities based on the detected intent."""
        # Simple tokenization and cleaning
        words = word_tokenize(text.lower())
        words = [word for word in words if word not in self.stop_words and word not in string.punctuation]
        
        entities = {}
        
        if intent == 'menu':
            # Extract price
            price_match = re.search(r'[₹$]\s*(\d+)', text) or re.search(r'(?:rs\.?|usd)\s*(\d+)', text.lower())
            if price_match:
                entities['price'] = price_match.group(1)
            
            # Extract item name (simple approach - could be enhanced)
            if 'add' in text.lower():
                start_idx = text.lower().find('add') + 3
                end_idx = text.lower().find(' for ', start_idx) if ' for ' in text.lower() else len(text)
                item = text[start_idx:end_idx].strip()
                if item:
                    entities['item'] = item
            
            # Simple category detection
            for category in ['starter', 'main course', 'dessert', 'beverage']:
                if category in text.lower():
                    entities['category'] = category
                    break
        
        elif intent == 'inventory':
            # Look for items after check/stock/quantity
            check_words = ['check', 'stock', 'quantity', 'how much', 'how many']
            for word in check_words:
                if word in text.lower():
                    start_idx = text.lower().find(word) + len(word)
                    item = text[start_idx:].split()[0].strip()
                    if item and len(item) > 2:  # Basic validation
                        entities['item'] = item
                        break
        
        elif intent == 'analytics':
            # Extract time period
            time_terms = ['today', 'yesterday', 'week', 'month', 'year']
            for term in time_terms:
                if term in text.lower():
                    entities['period'] = term
                    break
            
            # Extract metric
            metrics = ['sales', 'revenue', 'profit', 'orders', 'customers']
            for metric in metrics:
                if metric in text.lower():
                    entities['metric'] = metric
                    break
        
        elif intent == 'support':
            # Extract issue description
            issue_phrases = ['help with', 'problem with', 'issue with', 'need help']
            for phrase in issue_phrases:
                if phrase in text.lower():
                    start_idx = text.lower().find(phrase) + len(phrase)
                    issue = text[start_idx:].strip()
                    if issue:
                        entities['issue_description'] = issue
                        break
            if 'issue_description' not in entities:
                entities['issue_description'] = text  # Fallback to full text
        
        elif intent == 'raw_material':
            # Extract quantity and unit
            qty_match = re.search(r'(\d+)\s*(kg|g|ml|l|liters?|kilos?|grams?)', text.lower())
            if qty_match:
                entities['quantity'] = qty_match.group(1)
                entities['unit_type'] = qty_match.group(2)
            
            # Extract material name
            add_words = ['add', 'new', 'order']
            for word in add_words:
                if word in text.lower():
                    start_idx = text.lower().find(word) + len(word)
                    material = text[start_idx:].split(' ')[0].strip()
                    if material and len(material) > 2:
                        entities['material_name'] = material
                        break
        
        return entities
    
    def optimize_prompt(self, text: str) -> Tuple[str, str, Dict[str, str]]:
        """
        Optimize the user's natural language query into a structured prompt.
        
        Args:
            text: The user's natural language query
            
        Returns:
            Tuple of (optimized_prompt, detected_intent, extracted_entities)
        """
        if not text or not text.strip():
            return "Error: Empty input", "error", {}
        
        # Classify intent
        intent = self.classify_intent(text)
        
        # Extract entities
        entities = self.extract_entities(text, intent)
        
        # Get template and fill in entities
        template = self.templates.get(intent, "{}")
        try:
            optimized = template.format(**{k: entities.get(k, f"[MISSING {k.upper()}]") for k in template.split('{') if '}' in k})
        except Exception as e:
            optimized = f"Error generating prompt: {str(e)}"
        
        return optimized, intent, entities
