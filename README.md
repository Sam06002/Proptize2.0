# PetPooja Prompt Optimizer

A Streamlit-based application that transforms natural language queries into optimized prompts for the PetPooja Agent (restaurant POS system).

## Features

- **Natural Language Processing**: Understands various types of restaurant-related queries
- **Intent Classification**: Identifies the purpose of the query (menu, inventory, analytics, etc.)
- **Entity Extraction**: Extracts key information like items, quantities, prices, and time periods
- **Smart Defaults**: Provides sensible defaults for missing information
- **Interactive UI**: User-friendly interface with sample queries and feedback system
- **Analytics**: Tracks usage patterns and optimization performance

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd petpooja-prompt-optimizer
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your query in the text area or select a sample query from the sidebar

4. Click "Optimize Prompt" to generate the optimized prompt

5. If any required information is missing, provide it in the form that appears

6. Provide feedback using the thumbs up/down buttons to help improve the system

## Project Structure

```
petpooja-prompt-optimizer/
├── app.py                 # Main Streamlit application
├── optimizer.py           # Core NLP and optimization logic
├── data/
│   ├── templates.json     # Prompt templates for different intents
│   └── samples.json       # Sample natural language queries
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Customization

### Adding New Templates

1. Edit `data/templates.json` to add or modify prompt templates:
   ```json
   "your_new_intent": {
       "template": "Your template with {placeholders}",
       "required": ["list", "of", "required", "entities"],
       "defaults": {"optional": "default values"}
   }
   ```

### Adding Sample Queries

1. Edit `data/samples.json` to add or modify sample queries:
   ```json
   [
       "Your new sample query",
       "Another example query"
   ]
   ```

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Port already in use**:
   ```bash
   streamlit run app.py --server.port 8502
   ```

## Performance

The application is optimized for performance with:
- Cached NLP model loading
- Efficient entity extraction
- Asynchronous processing for better responsiveness

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- NLP powered by [spaCy](https://spacy.io/)
- Icons by [Font Awesome](https://fontawesome.com/)
