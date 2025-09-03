import streamlit as st
import json
import os
from typing import Dict, Any, List, Optional
from optimizer import PetPoojaOptimizer
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="PetPooja Prompt Optimizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .stTextArea>div>div>textarea {
        min-height: 150px;
    }
    .entity-badge {
        display: inline-block;
        padding: 0.4em 0.8em;
        font-size: 85%;
        font-weight: 700;
        line-height: 1.2;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 4px;
        background-color: #2c5282;  /* Darker blue for better contrast */
        color: #ffffff;
        margin: 3px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .intent-badge {
        background-color: #2f855a;  /* Darker green for better contrast */
    }
    .success-box {
        padding: 1.2rem;
        background-color: #f0fdf4;  /* Lighter background for better contrast */
        color: #1a2e05;  /* Darker text color */
        border-left: 4px solid #38a169;  /* Slightly darker green border */
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_optimizer():
    """Load the PetPoojaOptimizer with caching."""
    return PetPoojaOptimizer("data/templates.json")

@st.cache_data
def load_sample_queries() -> List[str]:
    """Load sample queries from JSON file."""
    try:
        with open("data/samples.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading sample queries: {e}")
        return []

def display_entity_badges(entities: Dict[str, Any]) -> None:
    """Display extracted entities as badges."""
    if not entities:
        st.info("No entities extracted.")
        return
    
    cols = st.columns(4)
    for i, (key, value) in enumerate(entities.items()):
        with cols[i % 4]:
            st.markdown(f'<span class="entity-badge">{key}: {value}</span>', unsafe_allow_html=True)

def display_missing_entities_form(missing_entities: List[str], intent: str) -> Dict[str, Any]:
    """Display a form for user to provide missing entities."""
    if not missing_entities:
        return {}
    
    st.warning("Some required information is missing. Please provide the following details:")
    user_entities = {}
    
    # Create a form for missing entities
    with st.form("missing_entities_form"):
        for entity in missing_entities:
            user_entities[entity] = st.text_input(
                f"{entity.replace('_', ' ').title()}",
                placeholder=f"Enter {entity.replace('_', ' ')}"
            )
        
        submitted = st.form_submit_button("Re-optimize with provided details")
        if submitted:
            return {k: v for k, v in user_entities.items() if v}
    
    return {}

def main():
    """Main Streamlit application."""
    st.title("✨ PetPooja Prompt Optimizer")
    st.markdown("Transform natural language queries into optimized prompts for PetPooja Agent.")
    
    # Initialize session state
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = load_optimizer()
    if 'optimized_result' not in st.session_state:
        st.session_state.optimized_result = None
    if 'user_entities' not in st.session_state:
        st.session_state.user_entities = {}
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar with sample queries
    with st.sidebar:
        st.header("Sample Queries")
        sample_queries = load_sample_queries()
        
        if st.button("Random Query"):
            import random
            st.session_state.sample_query = random.choice(sample_queries)
        
        for query in sample_queries[:10]:  # Show first 10 samples
            if st.button(query, key=f"sample_{query[:20]}"):
                st.session_state.sample_query = query
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input text area
        query = st.text_area(
            "Enter your query:",
            value=st.session_state.get('sample_query', ''),
            placeholder="e.g., Add butter chicken to the menu for ₹350"
        )
        
        # Optimize button
        if st.button("Optimize Prompt"):
            if not query.strip():
                st.error("Please enter a query to optimize.")
            else:
                with st.spinner("Optimizing your prompt..."):
                    st.session_state.optimized_result = st.session_state.optimizer.optimize_prompt(
                        query, 
                        st.session_state.get('user_entities', {})
                    )
                    st.session_state.user_entities = {}
                    st.rerun()
        
        # Display results if available
        if st.session_state.get('optimized_result'):
            result = st.session_state.optimized_result
            
            # Show success message
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Optimized Prompt Generated</h4>
                <p>Intent: <span class="entity-badge intent-badge">{result['intent'].title()}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show original and optimized prompt
            with st.expander("Original Query", expanded=False):
                st.write(query)
            
            st.subheader("Optimized Prompt")
            st.code(result['optimized_prompt'], language="text")
            
            # Show extracted entities
            st.subheader("Extracted Entities")
            display_entity_badges(result['entities'])
            
            # Show missing entities form if any
            if result['missing_entities']:
                st.session_state.user_entities = display_missing_entities_form(
                    result['missing_entities'], 
                    result['intent']
                )
            
            # Feedback buttons
            st.subheader("Was this helpful?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Yes"):
                    st.session_state.optimizer.record_feedback(True)
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎 No"):
                    st.session_state.optimizer.record_feedback(False)
                    st.error("We'll try to improve. Thanks for your feedback!")
    
    with col2:
        # Show analytics in the sidebar
        st.header("Analytics")
        analytics = st.session_state.optimizer.get_analytics_summary()
        
        st.metric("Total Optimizations", analytics['total_optimizations'])
        st.metric("Avg. Response Time", f"{analytics['average_response_time']:.2f}s")
        
        st.subheader("Intent Distribution")
        for intent, count in analytics['intent_distribution'].items():
            st.progress(
                count / max(analytics['intent_distribution'].values(), default=1),
                text=f"{intent.title()}: {count}"
            )
        
        # Show history
        if st.session_state.optimizer.history:
            st.subheader("Recent Queries")
            for item in st.session_state.optimizer.history[-5:]:
                with st.expander(f"{item['intent'].title()}: {item['original_text'][:30]}..."):
                    st.caption(f"Confidence: {item['confidence']:.2f}")
                    st.code(item['optimized_prompt'])

if __name__ == "__main__":
    main()
