"""
PetPooja Prompt Optimizer

This module provides the core functionality for optimizing natural language queries
into structured prompts for the PetPooja Agent (restaurant POS system).
It handles intent classification, entity extraction, and template-based prompt generation.
"""

import json
import re
import spacy
import difflib
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import os

class PetPoojaOptimizer:
    """
    Main class for optimizing natural language queries into structured prompts.
    
    This class handles the entire pipeline from natural language understanding
    to optimized prompt generation using spaCy for NLP and template-based
    prompt construction.
    
    Attributes:
        templates (dict): Loaded prompt templates from JSON
        nlp: spaCy NLP pipeline for text processing
        analytics (dict): Tracking metrics for optimization performance
        history (list): Log of previous optimizations
    """
    
    def __init__(self, templates_path: str = "data/templates.json"):
        """
        Initialize the PetPoojaOptimizer with templates and NLP model.
        
        Args:
            templates_path (str): Path to the JSON file containing prompt templates.
                                Defaults to "data/templates.json".
        """
        self.templates = self._load_templates(templates_path)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize analytics tracking
        self.analytics = {
            'total_optimizations': 0,  # Total number of optimizations performed
            'intent_distribution': {},  # Count of each intent type encountered
            'response_times': [],       # List of response times in seconds
            'feedback': {               # User feedback counts
                'positive': 0,         # Positive feedback count
                'negative': 0          # Negative feedback count
            }
        }
        self.history = []  # Store history of optimizations

    @staticmethod
    def _load_templates(templates_path: str) -> Dict[str, Any]:
        """
        Load and parse the JSON file containing prompt templates.
        
        Args:
            templates_path (str): Path to the JSON templates file
            
        Returns:
            Dict[str, Any]: Parsed template data
            
        Raises:
            Exception: If there's an error loading or parsing the templates file
        """
        try:
            with open(templates_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in templates file: {str(e)}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Templates file not found at: {templates_path}")
        except Exception as e:
            raise Exception(f"Error loading templates: {str(e)}")

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text using keyword matching and NLP techniques.
        
        This method analyzes the input text to determine the most likely intent
        by matching against predefined patterns and calculating similarity scores.
        
        Args:
            text (str): The input text to classify
            
        Returns:
            Tuple[str, float]: A tuple containing:
                - str: The classified intent (e.g., 'menu', 'inventory', 'analytics')
                - float: Confidence score between 0 and 1
                
        Example:
            >>> optimizer = PetPoojaOptimizer()
            >>> intent, confidence = optimizer.classify_intent("Add paneer tikka to the menu")
            >>> print(f"Intent: {intent}, Confidence: {confidence:.2f}")
        """
        if not text.strip():
            return "unknown", 0.0
            
        text_lower = text.lower()
        intent_scores = {}
        doc = self.nlp(text_lower)
        
        # Extract key terms and lemmas for better matching
        key_terms = set()
        for token in doc:
            # Skip stop words and punctuation
            if not token.is_stop and not token.is_punct:
                key_terms.add(token.lemma_)
                key_terms.add(token.text)
        
        # Calculate intent scores based on template matching
        for intent, templates in self.templates.items():
            # Initialize score for this intent
            intent_scores[intent] = 0
            
            # Check against all templates for this intent
            for template in templates.get("patterns", []):
                # Simple keyword matching
                if any(term in text_lower for term in template.get("keywords", [])):
                    intent_scores[intent] += 2  # Higher weight for direct keyword matches
                    
                # Check for required terms
                if all(req_term in key_terms for req_term in template.get("required_terms", [])):
                    intent_scores[intent] += 1.5
                    
                # Check for optional terms
                for opt_term in template.get("optional_terms", []):
                    if opt_term in key_terms:
                        intent_scores[intent] += 0.5
        
        # Find the intent with the highest score
        if not intent_scores:
            return "unknown", 0.0
            
        max_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Normalize score to 0-1 range
        total_possible = 3.5 * len(self.templates)  # Max possible score per intent
        confidence = min(max_intent[1] / total_possible, 1.0) if total_possible > 0 else 0.0
        
        # Update analytics
        self.analytics['intent_distribution'][max_intent[0]] = \
            self.analytics['intent_distribution'].get(max_intent[0], 0) + 1
            
        return max_intent[0], confidence

    def extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """
        Extract relevant entities from the input text based on the detected intent.
        
        Args:
            text (str): The input text to extract entities from
            intent (str): The classified intent
            
        Returns:
            Dict[str, Any]: A dictionary of extracted entities and their values
        """
        entities = {}
        doc = self.nlp(text.lower())
        
        # Extract entities based on intent
        if intent == 'menu':
            # Look for menu items, prices, categories, etc.
            entities['items'] = self._extract_menu_items(doc)
            entities['prices'] = self._extract_prices(doc)
            entities['categories'] = self._extract_categories(doc)
            
        elif intent == 'inventory':
            # Look for inventory items, quantities, etc.
            entities['items'] = self._extract_inventory_items(doc)
            entities['quantities'] = self._extract_quantities(doc)
            
        elif intent == 'analytics':
            # Look for date ranges, metrics, etc.
            entities['time_period'] = self._extract_time_period(doc)
            entities['metrics'] = self._extract_metrics(doc)
            
        return entities
        
    def _extract_menu_items(self, doc) -> List[str]:
        """
        Extract menu items from the processed document.
        
        Args:
            doc: A spaCy Doc object containing the processed text
            
        Returns:
            List[str]: A list of extracted menu items
            
        Note:
            This is a placeholder implementation. In a production environment,
            this would use spaCy's NER or custom patterns to identify menu items.
        """
        # Implementation would use spaCy's NER or custom patterns
        return []
        
    def _extract_prices(self, doc):
        """
        Extract price values from the processed document.
        
        Args:
            doc: A spaCy Doc object containing the processed text
            
        Returns:
            List[float]: A list of extracted price values
            
        Note:
            This implementation looks for currency patterns in the text.
            It handles various formats like $10, 10 dollars, INR 100, etc.
        """
        prices = []
        # Look for currency patterns
        currency_patterns = [
            r'\$\s*(\d+(?:\.\d{1,2})?)',  # $10 or $ 10.99
            r'(?:\d+(?:\.\d{1,2})?)\s*(?:dollars?|USD|INR|Rs\.?)',  # 10 dollars, 10.99 USD
            r'(?:Rs\.?|INR)\s*(\d+(?:\.\d{1,2})?)'  # Rs 100 or INR 100.50
        ]
        
        for pattern in currency_patterns:
            matches = re.finditer(pattern, doc.text, re.IGNORECASE)
            for match in matches:
                try:
                    price = float(match.group(1) if match.groups() else match.group(0).replace('$', '').strip())
                    prices.append(price)
                except (ValueError, AttributeError):
                    continue
                    
        return prices
        
    def _extract_categories(self, doc) -> list:
        """
        Extract menu categories from the processed document.
        
        Args:
            doc: A spaCy Doc object containing the processed text
            
        Returns:
            list: A list of extracted categories
            
        Note:
            This implementation looks for common category indicators like 'category:', 'type:',
            or specific category names in the text.
        """
        categories = []
        
        # Look for category indicators
        category_indicators = ['category', 'type', 'section', 'cuisine']
        
        # Check for patterns like "category: appetizers" or "type: main course"
        for indicator in category_indicators:
            pattern = fr'{indicator}\s*[:=]?\s*([\w\s]+)'
            matches = re.finditer(pattern, doc.text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    categories.append(match.group(1).strip())
        
        # If no explicit categories found, try to infer from common category names
        if not categories:
            common_categories = [
                'appetizer', 'starter', 'main course', 'main', 'entree',
                'dessert', 'beverage', 'drink', 'side', 'soup', 'salad',
                'breakfast', 'lunch', 'dinner', 'snack', 'combo', 'meal'
            ]
            
            for token in doc:
                if token.text.lower() in common_categories:
                    categories.append(token.text.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        return [cat for cat in categories if not (cat in seen or seen.add(cat))]
        
    def _extract_inventory_items(self, doc) -> list:
        """
        Extract inventory items from the processed document.
        
        Args:
            doc: A spaCy Doc object containing the processed text
            
        Returns:
            list: A list of extracted inventory items
            
        Note:
            This implementation looks for noun phrases that are likely to be inventory items.
        """
        items = []
        
        # Look for patterns like "item: X" or "ingredient: Y"
        item_indicators = ['item', 'ingredient', 'product', 'material']
        
        for indicator in item_indicators:
            pattern = fr'{indicator}\s*[:=]?\s*([\w\s]+)'
            matches = re.finditer(pattern, doc.text, re.IGNORECASE)
            for match in matches:
                if match.group(1):
                    items.append(match.group(1).strip())
        
        # If no explicit items found, look for noun phrases
        if not items:
            for chunk in doc.noun_chunks:
                # Skip short or common words that are unlikely to be items
                if len(chunk.text) > 2 and chunk.text.lower() not in ['the', 'and', 'or', 'for', 'with']:
                    items.append(chunk.text)
        
        return items
        
    def _extract_quantities(self, doc) -> list:
        """
        Extract quantities from the processed document.
        
        Args:
            doc: A spaCy Doc object containing the processed text
            
        Returns:
            list: A list of extracted quantities with their units
            
        Note:
            This implementation looks for patterns like "5 kg", "10 pieces", etc.
        """
        quantities = []
        
        # Look for quantity patterns like "5 kg", "10 pieces", etc.
        quantity_patterns = [
            r'(\d+(\.\d+)?)\s*(kg|g|ml|l|pcs|pieces?|units?|liters?|kilos?|grams?|milliliters?)',
            r'(\d+(\.\d+)?)\s*(?:of\s+)?(\w+)',
            r'(\d+(\.\d+)?)\s*([a-z]+)'
        ]
        
        for pattern in quantity_patterns:
            matches = re.finditer(pattern, doc.text, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    unit = match.group(3) if len(match.groups()) >= 3 else 'units'
                    quantities.append(f"{value} {unit}")
                except (ValueError, IndexError):
                    continue
        
        return quantities
        
    def generate_prompt(self, text: str, intent: str, entities: Dict[str, Any]) -> str:
        """
        Generate an optimized prompt based on the intent and extracted entities.
        
        Args:
            text (str): The original input text
            intent (str): The classified intent
            entities (Dict[str, Any]): Extracted entities
            
        Returns:
            str: The optimized prompt
        """
        # Get the appropriate template for the intent
        template = self.templates.get(intent, {}).get('template', '')
        
        # Fill in the template with extracted entities
        try:
            return template.format(**entities)
        except KeyError as e:
            # Handle missing entity values
            return f"Error: Missing required entity {str(e)} in template"
    
    def optimize_prompt(self, text: str, user_entities: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main method to optimize a natural language query into a structured prompt.
        
        This is the primary entry point for the optimization pipeline that:
        1. Classifies the intent of the input text
        2. Extracts relevant entities
        3. Generates an optimized prompt using the appropriate template
        4. Tracks performance metrics
        
        Args:
            text (str): The input text to optimize
            user_entities (Dict, optional): Additional user-provided entities
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'original_text': The original input text
                - 'intent': The classified intent
                - 'confidence': Confidence score (0-1)
                - 'entities': Extracted entities
                - 'optimized_prompt': The generated prompt
                - 'timestamp': ISO format timestamp of when optimization occurred
                - 'processing_time_seconds': Time taken to process the request
                
        Example:
            >>> optimizer = PetPoojaOptimizer()
            >>> result = optimizer.optimize_prompt("Add paneer tikka for $10 to the menu")
            >>> print(result['optimized_prompt'])
        """
        start_time = datetime.now()
        
        # Classify intent
        intent, confidence = self.classify_intent(text)
        
        # Extract entities
        entities = self.extract_entities(text, intent)
        if user_entities:
            entities.update(user_entities)
            
        # Generate optimized prompt
        prompt = self.generate_prompt(text, intent, entities)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update analytics
        self.analytics['total_optimizations'] += 1
        self.analytics['response_times'].append(processing_time)
        
        # Log to history
        result = {
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'optimized_prompt': prompt,
            'processing_time_seconds': processing_time
        }
        self.history.append(result)
        
        return result

    def extract_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """
        Extract relevant entities from the input text based on the detected intent.
        
        This method serves as a dispatcher that calls the appropriate entity extraction
        methods based on the detected intent. It handles the entity extraction pipeline
        and ensures consistent return format.
        
        Args:
            text (str): The input text to extract entities from
            intent (str): The classified intent
            
        Returns:
            Dict[str, Any]: A dictionary of extracted entities where keys are entity types
                          and values are the extracted values or lists of values
                          
        Example:
            >>> optimizer = PetPoojaOptimizer()
            >>> entities = optimizer.extract_entities("Add paneer tikka for $10", "menu")
            >>> print(entities)
            {'items': ['paneer tikka'], 'prices': [10.0]}
        """
        if not text.strip():
            return {}
            
        doc = self.nlp(text.lower())
        entities = {}
        
        # Extract entities based on intent
        if intent == 'menu':
            # Look for menu items, prices, categories, etc.
            entities['items'] = self._extract_menu_items(doc)
            entities['prices'] = self._extract_prices(doc)
            entities['categories'] = self._extract_categories(doc)
            
        elif intent == 'inventory':
            # Look for inventory items, quantities, etc.
            entities['items'] = self._extract_inventory_items(doc)
            entities['quantities'] = self._extract_quantities(doc)
            
        elif intent == 'analytics':
            # Look for date ranges, metrics, etc.
            entities['time_period'] = self._extract_time_period(doc)
            entities['metrics'] = self._extract_metrics(doc)
            
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
        missing = []
        for req_entity in template.get('required', []):
            if req_entity not in entities:
                missing.append(req_entity)
                if req_entity in template.get('defaults', {}):
                    entities[req_entity] = template['defaults'][req_entity]
        
        if missing and 'defaults' in template:
            for req in missing:
                if req in template['defaults']:
                    entities[req] = template['defaults'][req]
        
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
