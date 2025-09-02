import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, NamedTuple
from fuzzywuzzy import fuzz
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import string
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationResult(NamedTuple):
    """Container for optimization results."""
    optimized_prompt: str
    original_intent: str
    detected_entities: List[Dict[str, str]]
    confidence: float
    template_used: Optional[str] = None
    suggestions: Optional[List[str]] = None

class PetPoojaOptimizer:
    def __init__(self):
        # Load templates
        templates_path = Path(__file__).parent / "data" / "templates.json"
        with open(templates_path, 'r') as f:
            self.templates = json.load(f)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading punkt tokenizer...")
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            logger.info("Downloading stopwords...")
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
    
    def extract_entities(self, text: str, intent: str) -> List[Dict[str, str]]:
        """Extract entities from the input text based on the detected intent."""
        entities = []
        
        # Simple tokenization that doesn't rely on punkt_tab
        words = re.findall(r"\b[\w']+\b", text.lower())
        words = [w for w in words if w not in self.stop_words and w not in string.punctuation]
        
        # Basic entity extraction based on intent
        if intent == 'menu':
            # Look for menu items and prices
            for i, word in enumerate(words):
                if word.isdigit() and i > 0 and words[i-1] in ['rs', 'rs.', '₹', '$']:
                    entities.append({
                        'type': 'price',
                        'value': f"{words[i-1]}{word}",
                        'original_text': f"{words[i-1]} {word}"
                    })
            if 'add' in text.lower():
                start_idx = text.lower().find('add') + 3
                end_idx = text.lower().find(' for ', start_idx) if ' for ' in text.lower() else len(text)
                item = text[start_idx:end_idx].strip()
                if item:
                    entities.append({
                        'type': 'item',
                        'value': item,
                        'original_text': item
                    })
            
            # Simple category detection
            for category in ['starter', 'main course', 'dessert', 'beverage']:
                if category in text.lower():
                    entities.append({
                        'type': 'category',
                        'value': category,
                        'original_text': category
                    })
                    break
        
        elif intent == 'inventory':
            # Look for items after check/stock/quantity
            check_words = ['check', 'stock', 'quantity', 'how much', 'how many']
            for word in check_words:
                if word in text.lower():
                    start_idx = text.lower().find(word) + len(word)
                    item = text[start_idx:].split()[0].strip()
                    if item and len(item) > 2:  # Basic validation
                        entities.append({
                            'type': 'item',
                            'value': item,
                            'original_text': item
                        })
                        break
        
        elif intent == 'analytics':
            # Extract time period
            time_terms = ['today', 'yesterday', 'week', 'month', 'year']
            for term in time_terms:
                if term in text.lower():
                    entities.append({
                        'type': 'period',
                        'value': term,
                        'original_text': term
                    })
                    break
            
            # Extract metric
            metrics = ['sales', 'revenue', 'profit', 'orders', 'customers']
            for metric in metrics:
                if metric in text.lower():
                    entities.append({
                        'type': 'metric',
                        'value': metric,
                        'original_text': metric
                    })
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
                entities.append({
                    'type': 'quantity',
                    'value': qty_match.group(1),
                    'original_text': qty_match.group(1)
                })
                entities.append({
                    'type': 'unit_type',
                    'value': qty_match.group(2),
                    'original_text': qty_match.group(2)
                })
            
            # Extract material name
            add_words = ['add', 'new', 'order']
            for word in add_words:
                if word in text.lower():
                    start_idx = text.lower().find(word) + len(word)
                    material = text[start_idx:].split(' ')[0].strip()
                    if material and len(material) > 2:
                        entities.append({
                            'type': 'material_name',
                            'value': material,
                            'original_text': material
                        })
                        break
        
        return entities
    
    def optimize_prompt(self, text: str) -> OptimizationResult:
        """
        Optimize the user's natural language query into a structured prompt.
        
        Args:
            text: The user's natural language query
            
        Returns:
            OptimizationResult: Contains the optimized prompt and metadata
        """
        if not text or not text.strip():
            return OptimizationResult(
                optimized_prompt="Error: Empty input",
                original_intent="error",
                detected_entities=[],
                confidence=0.0
            )
        
        # Classify intent
        intent = self.classify_intent(text)
        
        # Extract entities
        entities = self.extract_entities(text, intent)
        
        # Get the appropriate template based on intent
        if intent not in self.templates:
            # If intent not found, use default template with all entities as a string
            entities_str = ", ".join([f"{e['type']}: {e['value']}" for e in entities])
            optimized = self.templates.get('default', "Process: {intent} with {entities}").format(
                intent=intent,
                entities=entities_str
            )
        else:
            # Format the template with extracted entities
            template = self.templates[intent]
            try:
                # Convert entities to dict for easier formatting
                entity_dict = {e['type']: e['value'] for e in entities}
                optimized = template.format(**entity_dict)
            except KeyError as e:
                return OptimizationResult(
                    optimized_prompt=f"Error: Missing required entity - {str(e)}",
                    original_intent="error",
                    detected_entities=entities,
                    confidence=0.0
                )
        
        # Calculate confidence (simple implementation - can be enhanced)
        confidence = min(1.0, len(entities) / 3.0)  # Scale confidence with number of entities
        
        return OptimizationResult(
            optimized_prompt=optimized,
            original_intent=intent,
            detected_entities=entities,
            confidence=confidence,
            template_used=intent
        )
