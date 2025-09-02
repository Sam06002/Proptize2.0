import streamlit as st
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from optimizer import PetPoojaOptimizer, OptimizationResult

# Set page config
st.set_page_config(
    page_title="PetPooja Prompt Optimizer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache resources
@st.cache_resource
def get_optimizer() -> PetPoojaOptimizer:
    """Get a cached instance of the PetPoojaOptimizer."""
    return PetPoojaOptimizer()

@st.cache_data
def load_samples() -> List[str]:
    """Load sample queries from JSON file."""
    samples_path = Path(__file__).parent / "data" / "samples.json"
    try:
        with open(samples_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Samples file not found. Please ensure data/samples.json exists.")
        return []

def display_optimization_result(result: OptimizationResult) -> None:
    """Display the optimization result in a user-friendly format."""
    # Display optimized prompt
    st.subheader("Optimized Prompt")
    st.code(result.optimized_prompt, language="text")
    
    # Copy to clipboard button
    st.download_button(
        label="📋 Copy to Clipboard",
        data=result.optimized_prompt,
        file_name="optimized_prompt.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # Display details in expanders
    with st.expander("🔍 Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Detected Intent", f"{result.intent.capitalize()}")
            st.metric("Confidence", f"{result.confidence:.0%}")
        
        with col2:
            if result.needs_user_input:
                st.warning("⚠️ Additional information needed")
                for entity in result.missing_entities:
                    st.info(f"Please provide: {entity}")
    
    # Show extracted entities
    with st.expander("📋 Extracted Entities"):
        if result.entities:
            for key, value in result.entities.items():
                st.write(f"**{key.replace('_', ' ').title()}:** {value}")
        else:
            st.info("No entities extracted.")

def main():
    """Main application function."""
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    
    # Sidebar with examples and info
    with st.sidebar:
        st.title("Examples")
        samples = load_samples()
        selected_example = st.selectbox(
            "Try an example:",
            ["-- Select an example --"] + samples,
            key="example_selector"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool helps you optimize natural language queries for the PetPooja Agent.
        
        **Supported Intents:**
        - 🍽️ Menu Management
        - 📦 Inventory
        - 📊 Analytics
        - 🆘 Support
        - 🥫 Raw Materials
        """)
    
    # Main interface
    st.title("🍽️ PetPooja Prompt Optimizer")
    st.markdown("""
    Transform natural language queries into optimized prompts for the PetPooja Agent.
    Enter your query below and click 'Optimize' to see the results.
    """)
    
    # Text input with example
    user_input = st.text_area(
        "Enter your query:",
        value=st.session_state.get('last_input', ''),
        height=120,
        placeholder="e.g., Add butter chicken to the menu for ₹350",
        key="user_input"
    )
    
    # Optimize button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🚀 Optimize", use_container_width=True, type="primary"):
            if not user_input or user_input.strip() == "":
                st.warning("Please enter a query or select an example.")
            else:
                with st.spinner("Optimizing your prompt..."):
                    start_time = time.time()
                    optimizer = get_optimizer()
                    
                    # Process the query
                    result = optimizer.optimize_prompt(user_input)
                    
                    # Store in history
                    st.session_state.history.insert(0, {
                        'query': user_input,
                        'optimized': result.optimized_prompt,
                        'intent': result.intent,
                        'entities': result.entities,
                        'timestamp': time.time(),
                        'processing_time': time.time() - start_time
                    })
                    
                    # Store current result for further interaction
                    st.session_state.current_result = result
                    st.session_state.last_input = user_input
                    
                    # Rerun to update the UI
                    st.rerun()
    
    # Display current result if available
    if st.session_state.current_result:
        result = st.session_state.current_result
        display_optimization_result(result)
        
        # Handle missing entities
        if result.needs_user_input and result.missing_entities:
            st.markdown("---")
            st.subheader("🔍 Additional Information Required")
            
            # Create a form for missing entities
            with st.form(key="missing_entities_form"):
                entity_values = {}
                for entity in result.missing_entities:
                    entity_values[entity] = st.text_input(
                        f"Please provide {entity.replace('_', ' ').title()}:",
                        key=f"missing_{entity}"
                    )
                
                if st.form_submit_button("✅ Update and Re-optimize", type="primary"):
                    if all(entity_values.values()):
                        # Update the result with user-provided values
                        updated_entities = {
                            **result.entities,
                            **{k: v for k, v in entity_values.items() if v}
                        }
                        
                        # Re-optimize with the new entities
                        optimizer = get_optimizer()
                        new_result = optimizer.optimize_with_entities(
                            result.original_query,
                            updated_entities
                        )
                        
                        # Update the current result
                        st.session_state.current_result = new_result
                        st.session_state.history.insert(0, {
                            'query': result.original_query,
                            'optimized': new_result.optimized_prompt,
                            'intent': new_result.intent,
                            'entities': new_result.entities,
                            'timestamp': time.time(),
                            'processing_time': 0  # Not measured for re-optimization
                        })
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields.")
    
    # Display history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 History")
        
        for i, item in enumerate(st.session_state.history[:5]):  # Show last 5 items
            with st.expander(f"{item['intent'].capitalize()}: {item['query'][:50]}..."):
                st.code(item['optimized'], language="text")
                st.caption(f"Processed in {item.get('processing_time', 0):.2f}s")
                
                if st.button(f"Use this again", key=f"use_again_{i}"):
                    st.session_state.user_input = item['query']
                    st.rerun()

# Run the app
if __name__ == "__main__":
    main()
