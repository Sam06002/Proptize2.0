import streamlit as st
import json
from pathlib import Path
from typing import List, Dict, Any
from optimizer import PetPoojaOptimizer

# Set page config
st.set_page_config(
    page_title="PetPooja Prompt Optimizer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load sample queries
def load_samples() -> List[str]:
    samples_path = Path(__file__).parent / "data" / "samples.json"
    with open(samples_path, 'r') as f:
        return json.load(f)

# Cache the optimizer for better performance
@st.cache_resource
def get_optimizer():
    return PetPoojaOptimizer()

def main():
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Sidebar with examples
    st.sidebar.title("Examples")
    samples = load_samples()
    selected_example = st.sidebar.selectbox(
        "Try an example:",
        ["-- Select an example --"] + samples
    )
    
    # Main interface
    st.title("🍽️ PetPooja Prompt Optimizer")
    st.markdown("""
    Transform natural language queries into optimized prompts for the PetPooja Agent.
    Enter your query below and click 'Optimize' to see the results.
    """)
    
    # Text input with example
    user_input = st.text_area(
        "Enter your query:",
        value=selected_example if selected_example != "-- Select an example --" else "",
        height=100,
        placeholder="e.g., Add butter chicken to the menu for ₹350"
    )
    
    # Optimize button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🚀 Optimize", use_container_width=True):
            if not user_input or user_input.strip() == "":
                st.warning("Please enter a query or select an example.")
            else:
                with st.spinner("Optimizing your prompt..."):
                    # Get the optimizer
                    optimizer = get_optimizer()
                    
                    # Process the query
                    optimized, intent, entities = optimizer.optimize_prompt(user_input)
                    
                    # Store in history
                    st.session_state.history.insert(0, {
                        'query': user_input,
                        'optimized': optimized,
                        'intent': intent,
                        'entities': entities
                    })
                    
                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["Optimized Prompt", "Details", "Raw Data"])
                    
                    with tab1:
                        st.subheader("Optimized Prompt")
                        st.code(optimized, language="text")
                        
                        # Copy to clipboard
                        st.download_button(
                            label="📋 Copy to Clipboard",
                            data=optimized,
                            file_name="optimized_prompt.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with tab2:
                        st.subheader("Detected Intent")
                        intent_emoji = {
                            'menu': '📋',
                            'inventory': '📦',
                            'analytics': '📊',
                            'support': '🆘',
                            'raw_material': '🥫'
                        }.get(intent, '❓')
                        
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                            <span style="font-size: 24px;">{intent_emoji}</span>
                            <h3 style="margin: 0;">{intent.replace('_', ' ').title()}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.subheader("Extracted Entities")
                        if entities:
                            cols = st.columns(2)
                            for i, (key, value) in enumerate(entities.items()):
                                with cols[i % 2]:
                                    st.markdown(f"**{key.replace('_', ' ').title()}**")
                                    st.info(value)
                        else:
                            st.info("No entities were extracted from this query.")
                    
                    with tab3:
                        st.subheader("Raw Data")
                        st.json({
                            'original_query': user_input,
                            'optimized_prompt': optimized,
                            'detected_intent': intent,
                            'extracted_entities': entities
                        })
    
    # History section
    if st.session_state.history:
        st.sidebar.title("History")
        for i, item in enumerate(st.session_state.history[:5]):  # Show last 5 items
            with st.sidebar.expander(f"{item['query'][:30]}..." if len(item['query']) > 30 else item['query']):
                st.text(f"Intent: {item['intent']}")
                st.text_area("Optimized", value=item['optimized'], height=50, key=f"opt_{i}")
                if st.button("🔁 Use this", key=f"use_{i}", use_container_width=True):
                    st.experimental_rerun()
        
        if st.sidebar.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
            st.experimental_rerun()
    
    # Add some styling
    st.markdown("""
    <style>
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    .stTextArea>div>div>textarea {
        min-height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
