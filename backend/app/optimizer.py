import re
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
from difflib import get_close_matches
from collections import deque
from math import exp
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EntityExtraction:
    value: str
    confidence: float = 1.0
    needs_user_input: bool = False
    suggested_correction: Optional[str] = None
    is_corrected: bool = False

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

class PetPoojaOptimizer:
    """
    Optimizes natural language queries for PetPooja Agent.
    Handles menu management, inventory queries, analytics, and support.
    """
    
    def __init__(self, history_size: int = 10):
        # In-memory storage for optimization history
        self.optimization_history: deque[OptimizationResult] = deque(maxlen=history_size)
        self.feedback_history: List[Dict[str, Any]] = []
        self.analytics = {
            'total_optimizations': 0,
            'intent_distribution': {},
            'confidence_scores': [],
            'success_rate': 0,
            'feedback': {'positive': 0, 'negative': 0, 'total': 0}
        }
        # Define intent keywords with weights
        self.intent_keywords = {
            'menu': {
                'add': 1.0, 'new': 0.9, 'item': 0.8, 'dish': 0.9, 'food': 0.8, 
                'menu': 0.7, 'category': 0.6, 'price': 0.7, 'update': 0.8, 'remove': 0.7
            },
            'inventory': {
                'stock': 1.0, 'inventory': 0.9, 'check': 0.8, 'level': 0.7, 
                'quantity': 0.8, 'available': 0.7, 'out of stock': 0.9, 'low': 0.6
            },
            'analytics': {
                'sales': 1.0, 'revenue': 0.9, 'analytics': 0.8, 'report': 0.8, 
                'insight': 0.7, 'trend': 0.7, 'metric': 0.6, 'stat': 0.6, 'graph': 0.5
            },
            'support': {
                'help': 1.0, 'problem': 0.9, 'issue': 0.9, 'ticket': 0.8, 
                'error': 0.9, 'not working': 0.8, 'question': 0.7, 'assist': 0.7
            },
            'raw_material': {
                'ingredient': 1.0, 'raw': 0.9, 'material': 0.9, 'supply': 0.8, 
                'purchase': 0.7, 'order': 0.7, 'stock': 0.6
            }
        }
        
        # Common menu items and categories with variations
        self.menu_items = {
            'biryani': {'category': 'main course', 'variations': ['chicken biryani', 'veg biryani', 'mutton biryani']},
            'paneer': {'category': 'main course', 'variations': ['paneer tikka', 'paneer butter masala', 'kadai paneer']},
            'butter chicken': {'category': 'main course'},
            'naan': {'category': 'bread', 'variations': ['butter naan', 'garlic naan']},
            'gulab jamun': {'category': 'dessert'},
            'momos': {'category': 'starter', 'variations': ['veg momos', 'chicken momos']},
            # Add more items as needed
        }
        
        # Menu categories with common variations
        self.menu_categories = {
            'appetizer': ['starter', 'snack', 'finger food'],
            'main course': ['main', 'thali', 'curry', 'gravy'],
            'dessert': ['sweet', 'mithai'],
            'beverage': ['drink', 'juice', 'soda', 'mocktail'],
            'soup': ['soup', 'stew', 'broth'],
            'salad': ['salad', 'raita'],
            'bread': ['roti', 'naan', 'paratha', 'kulcha']
        }
        
        # Common metrics for analytics
        self.analytics_metrics = {
            'sales', 'revenue', 'profit', 'orders', 'customers', 
            'items', 'average', 'total', 'count', 'growth'
        }
        
        # Common time periods
        self.time_periods = {
            'today', 'yesterday', 'week', 'month', 'year',
            'daily', 'weekly', 'monthly', 'yearly',
            'last week', 'last month', 'last year'
        }
        
        # Define prompt templates with placeholders
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
