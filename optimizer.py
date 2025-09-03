import json
import re
import spacy
import difflib
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import os

class PetPoojaOptimizer:
    def __init__(self, templates_path: str = "data/templates.json"):
        """Initialize the optimizer with templates and load the NLP model."""
        self.templates = self._load_templates(templates_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.analytics = {
            'total_optimizations': 0,
            'intent_distribution': {},
            'response_times': [],
            'feedback': {'positive': 0, 'negative': 0}
        }
        self.history = []

    @staticmethod
    def _load_templates(templates_path: str) -> Dict:
        """Load prompt templates from JSON file."""
        try:
            with open(templates_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading templates: {str(e)}")

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """Classify the intent of the input text."""
        text_lower = text.lower()
        intent_scores = {}
        
        # Simple keyword matching first
        for intent in self.templates.keys():
            score = 0
            if intent in text_lower:
                score += 0.5
            
            # Add more sophisticated matching here if needed
            if intent == 'menu' and any(word in text_lower for word in ['add', 'remove', 'update', 'menu']):
                score += 0.3
            elif intent == 'inventory' and any(word in text_lower for word in ['stock', 'inventory', 'quantity']):
                score += 0.3
            elif intent == 'analytics' and any(word in text_lower for word in ['report', 'analytics', 'show me', 'generate']):
                score += 0.3
            elif intent == 'support' and any(word in text_lower for word in ['help', 'support', 'issue', 'problem']):
                score += 0.3
            elif intent == 'raw_material' and any(word in text_lower for word in ['order', 'supply', 'material']):
                score += 0.3
                
            intent_scores[intent] = score
        
        # Get the intent with highest score
        if not intent_scores:
            return 'support', 0.0  # Default to support if no match
            
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], min(1.0, best_intent[1])

    def extract_entities(self, text: str, intent: str) -> Tuple[Dict[str, Any], List[str]]:
        """Extract entities from text based on the detected intent."""
        doc = self.nlp(text)
        entities = {}
        missing = []
        
        # Extract common entities
        currency_entities = [ent for ent in doc.ents if ent.label_ == 'MONEY']
        if currency_entities:
            entities['price'] = currency_entities[0].text
            
        # Extract quantities and units
        quantity_matches = re.findall(r'(\d+)\s*(kg|g|ml|l|pcs|pieces?|units?)', text, re.IGNORECASE)
        if quantity_matches:
            quantity, unit = quantity_matches[0]
            entities['quantity'] = quantity
            entities['unit'] = unit.lower()
        
        # Extract time periods
        time_matches = re.search(r'(last|next|this|for)\s+(\d+)?\s*(day|week|month|year)s?', text, re.IGNORECASE)
        if time_matches:
            entities['time_period'] = time_matches.group(0)
        
        # Extract items and categories (simplified)
        if 'add' in text.lower() and 'to' in text.lower():
            parts = text.lower().split('to', 1)
            if parts:
                item_part = parts[0].replace('add', '').strip()
                entities['item'] = item_part
        
        # Check for required entities in the template
        template = self.templates.get(intent, {})
        for req_entity in template.get('required', []):
            if req_entity not in entities:
                missing.append(req_entity)
                if req_entity in template.get('defaults', {}):
                    entities[req_entity] = template['defaults'][req_entity]
        
        return entities, missing

    def optimize_prompt(self, text: str, user_entities: Optional[Dict] = None) -> Dict:
        """Generate an optimized prompt from the input text."""
        start_time = datetime.now()
        
        # Classify intent
        intent, confidence = self.classify_intent(text)
        
        # Extract entities
        entities, missing = self.extract_entities(text, intent)
        
        # Update with user-provided entities if any
        if user_entities:
            entities.update(user_entities)
            missing = [m for m in missing if m not in user_entities]
        
        # Generate optimized prompt
        template = self.templates.get(intent, {}).get('template', '{}')
        try:
            optimized_prompt = template.format(**{k: str(v) for k, v in entities.items()})
        except KeyError as e:
            optimized_prompt = f"Error: Missing required entity: {e}"
        
        # Update analytics
        self.analytics['total_optimizations'] += 1
        self.analytics['intent_distribution'][intent] = self.analytics['intent_distribution'].get(intent, 0) + 1
        
        response_time = (datetime.now() - start_time).total_seconds()
        self.analytics['response_times'].append(response_time)
        
        # Add to history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'optimized_prompt': optimized_prompt,
            'response_time': response_time
        }
        self.history.append(history_entry)
        
        return {
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'missing_entities': missing,
            'optimized_prompt': optimized_prompt,
            'template': template
        }
    
    def record_feedback(self, is_positive: bool) -> None:
        """Record user feedback on the last optimized prompt."""
        if is_positive:
            self.analytics['feedback']['positive'] += 1
        else:
            self.analytics['feedback']['negative'] += 1
    
    def get_analytics_summary(self) -> Dict:
        """Get a summary of analytics data."""
        avg_response_time = (
            sum(self.analytics['response_times']) / len(self.analytics['response_times'])
            if self.analytics['response_times'] else 0
        )
        
        return {
            'total_optimizations': self.analytics['total_optimizations'],
            'average_response_time': round(avg_response_time, 3),
            'intent_distribution': self.analytics['intent_distribution'],
            'feedback': self.analytics['feedback']
        }
