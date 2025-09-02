import re
import json
import spacy
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime
from difflib import get_close_matches
from collections import defaultdict, deque, Counter
from functools import lru_cache
import logging
from pathlib import Path
import streamlit as st
from time import perf_counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Precompile regex patterns
PRICE_RE = re.compile(r'₹\d+|Rs\.?\d+|\d+/-')
CATEGORIES = ["appetizer", "main course", "dessert", "beverage", "soup", "salad"]

# Menu items and categories (moved to module level for better caching)
MENU_ITEMS = {
    'biryani': {'category': 'main course', 'variations': ['chicken biryani', 'veg biryani', 'mutton biryani']},
    'paneer': {'category': 'main course', 'variations': ['paneer tikka', 'paneer butter masala', 'kadai paneer']},
    'butter chicken': {'category': 'main course'},
    'naan': {'category': 'bread', 'variations': ['butter naan', 'garlic naan']},
    'gulab jamun': {'category': 'dessert'},
    'momos': {'category': 'starter', 'variations': ['veg momos', 'chicken momos']},
}

MENU_CATEGORIES = {
    'appetizer': ['starter', 'snack', 'finger food'],
    'main course': ['main', 'thali', 'curry', 'gravy'],
    'dessert': ['sweet', 'mithai'],
    'beverage': ['drink', 'juice', 'soda', 'mocktail'],
    'soup': ['soup', 'stew', 'broth'],
    'salad': ['salad', 'raita'],
    'bread': ['roti', 'naan', 'paratha', 'kulcha']
}

@dataclass
class EntityExtraction:
    value: str
    confidence: float = 1.0
    needs_user_input: bool = False
    suggested_correction: Optional[str] = None
    is_corrected: bool = False
    source: str = "auto"  # 'auto', 'user', 'suggestion'

@dataclass
class OptimizationResult:
    original_query: str
    optimized_prompt: str
    intent: str
    confidence: float = 1.0
    entities: Dict[str, Any] = field(default_factory=dict)
    entity_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    needs_user_input: bool = False
    missing_entities: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    feedback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result
        
    def get_missing_entities(self) -> List[str]:
        """Return a list of missing entity names that need user input."""
        return [k for k, v in self.entity_details.items() 
               if v.get('needs_user_input', False) and not v.get('value')]

@st.cache_resource
def load_nlp():
    """Load and cache the spaCy NLP model."""
    try:
        nlp = spacy.load("en_core_web_sm")
        # Add custom pipeline components
        if not nlp.has_pipe("entity_ruler"):
            ruler = nlp.add_pipe("entity_ruler")
            # Add custom patterns for menu items, categories, etc.
            patterns = [
                {"label": "MENU_ITEM", "pattern": [{"LOWER": {"IN": list(MENU_ITEMS.keys())}}]},
                {"label": "CATEGORY", "pattern": [{"LOWER": {"IN": CATEGORIES}}]},
            ]
            ruler.add_patterns(patterns)
        return nlp
    except OSError:
        logger.error("spaCy model 'en_core_web_sm' not found. Please install it with 'python -m spacy download en_core_web_sm'")
        raise

@st.cache_data
def load_templates():
    """Load and cache the prompt templates."""
    templates_path = Path(__file__).parent / "data" / "templates.json"
    try:
        with open(templates_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Templates file not found at {templates_path}")
        return {}

@st.cache_data
def load_samples():
    """Load and cache the sample queries."""
    samples_path = Path(__file__).parent / "data" / "samples.json"
    try:
        with open(samples_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"Samples file not found at {samples_path}")
        return []

class PetPoojaOptimizer:
    """
    Optimizes natural language queries for PetPooja Agent with performance optimizations.
    Handles menu management, inventory queries, analytics, and support.
    
    Features:
    - Fast intent classification using keyword matching with weights
    - Entity extraction with spaCy for accurate parsing
    - Interactive feedback for missing information
    - Caching for improved performance
    - Support for multiple intents and entities
    """
    
    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the input text using keyword matching with weights.
        Returns a tuple of (intent, confidence_score).
        """
        text_lower = text.lower()
        scores = defaultdict(float)
        
        # Calculate scores for each intent based on keyword matches
        for intent, keywords in self.intent_keywords.items():
            for keyword, weight in keywords.items():
                if keyword in text_lower:
                    scores[intent] += weight
        
        if not scores:
            return ('unknown', 0.0)
            
        # Get the intent with the highest score
        best_intent = max(scores.items(), key=lambda x: x[1])
        max_score = best_intent[1]
        total_score = sum(scores.values())
        
        # Normalize confidence score between 0 and 1
        confidence = (max_score / total_score) if total_score > 0 else 0
        
        return (best_intent[0], confidence)
    
    def extract_entities(self, text: str, intent: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extract entities from the input text based on the detected intent.
        Returns a tuple of (entities, missing_entities).
        """
        doc = self.nlp(text)
        entities = {}
        missing_entities = []
        
        # Extract common entities
        entities['price'] = self._extract_price(text)
        entities['category'] = self._extract_category(text)
        
        # Intent-specific entity extraction
        if intent == 'menu':
            entities['item'] = self._extract_menu_item(text, doc)
            if not entities['item']:
                missing_entities.append('item')
            if not entities['price']:
                missing_entities.append('price')
                
        elif intent == 'inventory':
            entities['item'] = self._extract_inventory_item(text, doc)
            if not entities['item']:
                missing_entities.append('item')
                
        elif intent == 'analytics':
            entities['metric'] = self._extract_metric(text, doc)
            entities['period'] = self._extract_time_period(text)
            
        elif intent == 'support':
            entities['issue'] = text  # Use full text as issue description
            
        elif intent == 'raw_material':
            entities['material'] = self._extract_material(text, doc)
            if not entities['material']:
                missing_entities.append('material')
        
        return entities, missing_entities
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text using regex."""
        match = self.price_re.search(text)
        if match:
            price_str = match.group().replace('₹', '').replace('Rs', '').replace('/', '').strip()
            try:
                return float(price_str)
            except (ValueError, TypeError):
                pass
        return None
    
    def _extract_category(self, text: str) -> Optional[str]:
        """Extract menu category from text."""
        text_lower = text.lower()
        for category, aliases in self.menu_categories.items():
            if category in text_lower:
                return category
            for alias in aliases:
                if alias in text_lower:
                    return category
        return None
    
    def _extract_menu_item(self, text: str, doc) -> Optional[str]:
        """Extract menu item from text using spaCy NER and custom patterns."""
        # Check for exact matches first
        text_lower = text.lower()
        for item in self.menu_items:
            if item in text_lower:
                return item
                
        # Use spaCy NER to find food items
        for ent in doc.ents:
            if ent.label_ == 'FOOD' or ent.label_ == 'MENU_ITEM':
                return ent.text
                
        # Fallback to noun chunks
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(word in chunk_text for word in ['pizza', 'burger', 'pasta']):  # Example items
                return chunk.text
                
        return None
    
    def _extract_inventory_item(self, text: str, doc) -> Optional[str]:
        """Extract inventory item from text."""
        # First check for menu items
        item = self._extract_menu_item(text, doc)
        if item:
            return item
            
        # Then check for raw materials
        return self._extract_material(text, doc)
    
    def _extract_metric(self, text: str, doc) -> str:
        """Extract metric from analytics query."""
        text_lower = text.lower()
        for metric in self.analytics_metrics:
            if metric in text_lower:
                return metric
        return 'sales'  # Default metric
    
    def _extract_time_period(self, text: str) -> str:
        """Extract time period from text."""
        text_lower = text.lower()
        for period in self.time_periods:
            if period in text_lower:
                return period
        return 'this week'  # Default period
    
    def _extract_material(self, text: str, doc) -> Optional[str]:
        """Extract raw material from text."""
        # Simple implementation - can be enhanced with a materials database
        materials = ['flour', 'sugar', 'rice', 'oil', 'spices', 'vegetables', 'meat']
        text_lower = text.lower()
        for material in materials:
            if material in text_lower:
                return material
                
        # Try to find a noun that's not a stop word
        for token in doc:
            if token.pos_ == 'NOUN' and not token.is_stop:
                return token.text
                
        return None
    
    def optimize_prompt(self, text: str) -> OptimizationResult:
        """
        Optimize a natural language query into a structured prompt.
        Returns an OptimizationResult object with the results.
        """
        start_time = perf_counter()
        
        # Classify intent
        intent, confidence = self.classify_intent(text)
        
        # Extract entities
        entities, missing_entities = self.extract_entities(text, intent)
        
        # Get the appropriate template
        template = self.templates.get(intent, {})
        template_str = template.get('template', '{query}')
        
        # Prepare context with defaults
        context = {
            'query': text,
            'intent': intent,
            **template.get('defaults', {}),
            **{k: v for k, v in entities.items() if v is not None}
        }
        
        # Check for missing required fields
        required_fields = template.get('required', [])
        missing_required = [f for f in required_fields if f not in context or context[f] is None]
        
        # Mark missing entities for user input
        needs_user_input = bool(missing_required)
        
        # Generate the optimized prompt
        try:
            optimized_prompt = template_str.format(**context)
        except KeyError as e:
            logger.warning(f"Missing key in template: {e}")
            optimized_prompt = text  # Fallback to original text
        
        # Record analytics
        processing_time = perf_counter() - start_time
        self.analytics['total_optimizations'] += 1
        self.analytics['intent_distribution'][intent] += 1
        self.analytics['confidence_scores'].append(confidence)
        self.analytics['response_times'].append(processing_time)
        
        # Update missing entities tracking
        for entity in missing_required:
            self.analytics['missing_entities'][entity] += 1
        
        # Create and return the result
        return OptimizationResult(
            original_query=text,
            optimized_prompt=optimized_prompt,
            intent=intent,
            confidence=confidence,
            entities=context,
            entity_details={
                k: {
                    'value': v,
                    'needs_user_input': k in missing_required,
                    'source': 'extracted' if k in entities else 'default'
                }
                for k, v in context.items()
            },
            needs_user_input=needs_user_input,
            missing_entities=missing_required
        )
    
    def optimize_with_entities(self, text: str, entities: Dict[str, Any]) -> OptimizationResult:
        """
        Optimize a prompt with pre-filled entity values.
        Useful for when the user provides missing information.
        """
        # Get the base optimization result
        result = self.optimize_prompt(text)
        
        # Update with user-provided entities
        for key, value in entities.items():
            if value:  # Only update with non-empty values
                result.entities[key] = value
                if key in result.entity_details:
                    result.entity_details[key].update({
                        'value': value,
                        'needs_user_input': False,
                        'source': 'user',
                        'is_corrected': True
                    })
        
        # Re-optimize with the updated entities
        return self.optimize_prompt_with_entities(text, result.entities, result.intent)
    
    def optimize_prompt_with_entities(self, text: str, entities: Dict[str, Any], 
                                    intent: str = None) -> OptimizationResult:
        """
        Optimize a prompt with pre-extracted entities and optional intent.
        This is more efficient than reprocessing from scratch.
        """
        if intent is None:
            intent, _ = self.classify_intent(text)
            
        template = self.templates.get(intent, {})
        template_str = template.get('template', '{query}')
        
        # Prepare context with defaults and provided entities
        context = {
            'query': text,
            'intent': intent,
            **template.get('defaults', {}),
            **{k: v for k, v in entities.items() if v is not None}
        }
        
        # Check for missing required fields
        required_fields = template.get('required', [])
        missing_required = [f for f in required_fields if f not in context or context[f] is None]
        
        # Generate the optimized prompt
        try:
            optimized_prompt = template_str.format(**context)
        except KeyError as e:
            logger.warning(f"Missing key in template: {e}")
            optimized_prompt = text  # Fallback to original text
        
        # Create and return the result
        return OptimizationResult(
            original_query=text,
            optimized_prompt=optimized_prompt,
            intent=intent,
            confidence=1.0,  # Higher confidence when using pre-validated entities
            entities=context,
            entity_details={
                k: {
                    'value': v,
                    'needs_user_input': k in missing_required,
                    'source': 'user' if k in entities else 'default'
                }
                for k, v in context.items()
            },
            needs_user_input=bool(missing_required),
            missing_entities=missing_required
        )
    
    def __init__(self, nlp=None, templates=None):
        # Load models and data with caching
        self.nlp = nlp or load_nlp()
        self.templates = templates or load_templates()
        
        # Initialize in-memory storage with reasonable defaults
        self.optimization_history = deque(maxlen=100)  # Increased history size
        self.feedback_history = []
        self.analytics = {
            'total_optimizations': 0,
            'intent_distribution': defaultdict(int),
            'confidence_scores': [],
            'success_rate': 0,
            'feedback': {'positive': 0, 'negative': 0, 'total': 0},
            'response_times': [],
            'missing_entities': defaultdict(int)
        }
        
        # Intent keywords with weights (optimized for quick lookup)
        self.intent_keywords = {
            'menu': {'add': 1.0, 'new': 0.9, 'item': 0.8, 'dish': 0.9, 'menu': 0.7},
            'inventory': {'stock': 1.0, 'inventory': 0.9, 'check': 0.8, 'level': 0.7},
            'analytics': {'sales': 1.0, 'revenue': 0.9, 'report': 0.8, 'insight': 0.7},
            'support': {'help': 1.0, 'problem': 0.9, 'issue': 0.9, 'ticket': 0.8},
            'raw_material': {'ingredient': 1.0, 'raw': 0.9, 'material': 0.9, 'supply': 0.8}
        }
        
        # Menu items and categories (moved to module level for better caching)
        self.menu_items = MENU_ITEMS
        self.menu_categories = MENU_CATEGORIES
        
        # Pre-compile regex patterns
        self.price_re = PRICE_RE
        
        # Common metrics for analytics
        self.analytics_metrics = {
            'sales', 'revenue', 'profit', 'orders', 'customers', 
            'items', 'average', 'total', 'count', 'growth'
        }
        
        # Common time periods
        self.time_periods = {
            'today', 'yesterday', 'week', 'month', 'year',
            'this week', 'last week', 'this month', 'last month',
            'this year', 'last year', 'quarter', 'q1', 'q2', 'q3', 'q4',
            'daily', 'weekly', 'monthly', 'yearly'
        }
        
        # Define prompt templates with placeholders and requirements
        self.templates = {
            'menu': {
                'template': "Add new {item} to {category} with price {price}",
                'required': ['item', 'price'],
                'defaults': {'category': 'main course'}
            },
            'inventory': {
                'template': "Check inventory for {item}",
                'required': ['item'],
                'defaults': {}
            },
            'analytics': {
                'template': "Show {metric} for {period}",
                'required': ['metric'],
                'defaults': {'period': 'today'}
            },
            'support': {
                'template': "Support request: {issue}",
                'required': ['issue'],
                'defaults': {}
            },
            'raw_material': {
                'template': "Process {material} request",
                'required': ['material'],
                'defaults': {}
            }
        }
    
    def get_analytics(self, time_period: str = 'all') -> Dict[str, Any]:
        """
        Get analytics about optimizations.
        
        Args:
            time_period: Time period to filter analytics ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            Dict containing analytics data
        """
        now = datetime.now()
        filtered_history = []
        
        # Filter history based on time period
        if time_period != 'all':
            for opt in self.optimization_history:
                opt_time = datetime.fromtimestamp(opt.timestamp)
                time_diff = now - opt_time
                
                if time_period == 'day' and time_diff.days < 1:
                    filtered_history.append(opt)
                elif time_period == 'week' and time_diff.days < 7:
                    filtered_history.append(opt)
                elif time_period == 'month' and time_diff.days < 30:
                    filtered_history.append(opt)
                elif time_period == 'year' and time_diff.days < 365:
                    filtered_history.append(opt)
        else:
            filtered_history = list(self.optimization_history)
        
        # Calculate metrics
        intent_dist = {}
        conf_scores = []
        
        for opt in filtered_history:
            intent_dist[opt.intent] = intent_dist.get(opt.intent, 0) + 1
            conf_scores.append(opt.confidence)
        
        avg_confidence = sum(conf_scores) / len(conf_scores) if conf_scores else 0
        
        return {
            'time_period': time_period,
            'total_optimizations': len(filtered_history),
            'intent_distribution': intent_dist,
            'average_confidence': round(avg_confidence, 2),
            'success_rate': round(self.analytics.get('success_rate', 0) * 100, 1),
            'feedback': self.analytics['feedback'],
            'top_queries': [opt.original_query for opt in filtered_history[:5]]
        }

    def _calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words using Levenshtein distance."""
        if not word1 or not word2:
            return 0.0
            
        word1 = word1.lower()
        word2 = word2.lower()
        
        # Simple character-based similarity
        common = set(word1) & set(word2)
        return len(common) / max(len(word1), len(word2))
    
    def _find_closest_match(self, word: str, word_list: List[str], threshold: float = 0.6) -> Optional[str]:
        """Find the closest match for a word in a list of words."""
        if not word or not word_list:
            return None
            
        matches = get_close_matches(word, word_list, n=1, cutoff=0.5)
        if matches and self._calculate_similarity(word, matches[0]) >= threshold:
            return matches[0]
        return None

    def classify_intent(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of the user query with confidence score.
        
        Returns:
            Tuple of (intent, confidence_score) where confidence_score is between 0 and 1
        """
        text = text.lower()
        words = set(re.findall(r'\b\w+\b', text))
        scores = {intent: 0.0 for intent in self.intent_keywords}
        
        for intent, keywords in self.intent_keywords.items():
            for word in words:
                # Check for exact matches first
                if word in keywords:
                    scores[intent] += keywords[word]
                else:
                    # Try fuzzy matching for similar words
                    closest = self._find_closest_match(word, list(keywords.keys()))
                    if closest:
                        similarity = self._calculate_similarity(word, closest)
                        scores[intent] += keywords[closest] * similarity * 0.7  # Penalize fuzzy matches
        
        # Calculate confidence score
        total_score = sum(scores.values())
        if total_score == 0:
            return 'unknown', 0.0
            
        # Normalize scores to 0-1 range
        max_score = max(scores.values())
        confidence = min(1.0, max_score / 3.0)  # Cap confidence at 1.0
        
        # Get the best matching intent
        best_intent = max(scores, key=scores.get)
        
        # If confidence is too low, mark as unknown
        if confidence < 0.3:
            return 'unknown', confidence
            
        return best_intent, confidence

    def extract_entities(self, text: str, intent: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Extract relevant entities based on the detected intent.
        
        Returns:
            Tuple of (entities, entity_details) where:
            - entities: Dict of entity name to value
            - entity_details: Dict of entity name to metadata (confidence, needs_input, etc.)
        """
        entities = {}
        entity_details = {}
        text_lower = text.lower()
        
        def add_entity(name: str, value: str, confidence: float = 1.0, needs_input: bool = False, 
                      suggested_correction: Optional[str] = None, is_corrected: bool = False):
            """Helper to add an entity with metadata."""
            entities[name] = value
            entity_details[name] = {
                'confidence': min(1.0, max(0.0, confidence)),
                'needs_user_input': needs_input,
                'suggested_correction': suggested_correction,
                'is_corrected': is_corrected
            }
        
        if intent == 'menu':
            # Extract price with currency
            price_match = re.search(r'[₹$]\s*(\d+(?:\.\d{1,2})?)', text) or \
                         re.search(r'(?:rs\.?|usd)\s*(\d+(?:\.\d{1,2})?)', text_lower)
            if price_match:
                price = price_match.group(1)
                add_entity('price', price, confidence=0.9)
            
            # Extract category with fuzzy matching
            found_category = False
            for category, variations in self.menu_categories.items():
                variations = [category] + variations if isinstance(variations, list) else [category]
                for variant in variations:
                    if variant in text_lower:
                        add_entity('category', category, confidence=0.95)
                        found_category = True
                        break
                if found_category:
                    break
            
            if not found_category:
                # Try to infer category from item name
                for item, data in self.menu_items.items():
                    if item in text_lower:
                        add_entity('category', data['category'], confidence=0.8, 
                                 needs_input=True, suggested_correction=f"Is this a {data['category']}?")
                        break
            
            # Extract item name with fuzzy matching
            words = re.findall(r'\b\w+\b', text_lower)
            item_candidates = []
            
            # Look for known menu items
            for item in self.menu_items:
                if item in text_lower:
                    item_candidates.append((item, 1.0))  # Exact match
                else:
                    # Try fuzzy matching
                    closest = self._find_closest_match(item, words)
                    if closest:
                        similarity = self._calculate_similarity(item, closest)
                        if similarity > 0.7:  # Only consider good matches
                            item_candidates.append((item, similarity))
            
            if item_candidates:
                # Sort by match quality
                item_candidates.sort(key=lambda x: x[1], reverse=True)
                best_item, confidence = item_candidates[0]
                is_corrected = best_item.lower() not in text_lower
                
                add_entity('item', best_item, 
                          confidence=confidence,
                          suggested_correction=best_item if is_corrected else None,
                          is_corrected=is_corrected)
        
        # Add more entity extraction logic for other intents...
        
        return entities, entity_details

    def _store_optimization(self, result: OptimizationResult) -> None:
        """Store the optimization result in history."""
        self.optimization_history.append(result)
        self.analytics['total_optimizations'] += 1
        
        # Update intent distribution
        intent = result.intent
        self.analytics['intent_distribution'][intent] = self.analytics['intent_distribution'].get(intent, 0) + 1
        
        # Update confidence scores
        self.analytics['confidence_scores'].append(result.confidence)
        
        # Update success rate (simplified - in practice, you'd track actual success)
        if result.confidence > 0.7:
            self.analytics['success_rate'] = (
                (self.analytics.get('success_rate', 0) * (self.analytics['total_optimizations'] - 1) + 1) 
                / self.analytics['total_optimizations']
            )
    
    def add_feedback(self, optimization_id: str, is_helpful: bool, comment: str = '') -> None:
        """Add user feedback for an optimization."""
        feedback = {
            'optimization_id': optimization_id,
            'timestamp': datetime.now().timestamp(),
            'is_helpful': is_helpful,
            'comment': comment
        }
        self.feedback_history.append(feedback)
        
        # Update analytics
        if is_helpful:
            self.analytics['feedback']['positive'] += 1
        else:
            self.analytics['feedback']['negative'] += 1
        self.analytics['feedback']['total'] += 1
    
    def find_similar_optimizations(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find similar past optimizations for the given query."""
        if not self.optimization_history:
            return []
            
        # Simple similarity check - in practice, you'd use embeddings or more sophisticated matching
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        scored_results = []
        
        for opt in self.optimization_history:
            opt_terms = set(re.findall(r'\b\w+\b', opt.original_query.lower()))
            common_terms = query_terms & opt_terms
            similarity = len(common_terms) / max(len(query_terms), len(opt_terms))
            
            if similarity > 0.3:  # Only include somewhat similar results
                scored_results.append((similarity, opt))
        
        # Sort by similarity and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [opt.to_dict() for _, opt in scored_results[:limit]]
    
    def optimize_prompt(self, query: str) -> Dict[str, Any]:
        """
        Optimize the given natural language query into a structured prompt.
        
        Args:
            query: The natural language query
            
        Returns:
            Dict containing optimized prompt and metadata
        """
        # Step 1: Classify intent with confidence score
        intent, confidence = self.classify_intent(query)
        
        # Step 2: Extract entities with confidence scores
        entities, entity_details = self.extract_entities(query, intent)
        
        # Step 3: Check for missing required entities
        required_entities = self.templates.get(intent, {}).get('required', [])
        missing_entities = [e for e in required_entities if e not in entities]
        
        # Step 4: Generate optimized prompt using template
        template = self.templates.get(intent, {}).get('template', '{query}')
        
        try:
            optimized = template.format(query=query, **entities)
        except KeyError as e:
            # Handle missing entities
            missing = str(e).strip("'")
            if missing not in missing_entities:
                missing_entities.append(missing)
            optimized = f"{template} [MISSING: {missing}]"
        
        # Create result object
        result = OptimizationResult(
            original_query=query,
            optimized_prompt=optimized,
            intent=intent,
            confidence=confidence,
            entities=entities,
            entity_details=entity_details,
            needs_user_input=len(missing_entities) > 0 or confidence < 0.7,
            missing_entities=missing_entities
        )
        
        # Store the optimization in history
        self._store_optimization(result)
        
        # Find similar past optimizations
        similar_optimizations = self.find_similar_optimizations(query)
        
        # Convert to dict and add similar optimizations
        result_dict = result.to_dict()
        if similar_optimizations:
            result_dict['similar_optimizations'] = similar_optimizations
        
        return result_dict
