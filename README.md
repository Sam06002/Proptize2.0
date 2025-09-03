# PetPooja Prompt Optimizer

A Streamlit-based application that transforms natural language queries into optimized prompts for the PetPooja Agent, a restaurant POS system. This tool helps restaurant staff interact with the POS system using natural language, making it more intuitive and efficient.

## 🚀 Features

- **Natural Language Understanding**: Processes restaurant-related queries using spaCy's NLP pipeline
- **Intent Recognition**: Classifies queries into categories like menu management, inventory, and analytics
- **Entity Extraction**: Identifies key entities such as menu items, quantities, prices, and time periods
- **Template-based Optimization**: Uses predefined templates to structure optimized prompts
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience
- **Query History**: Maintains a history of previous optimizations for reference
- **Analytics Dashboard**: Tracks usage patterns and optimization performance

## 🛠️ Prerequisites

- Python 3.10 (recommended) or 3.8+
- pip (Python package manager)
- Git (for version control)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sam06002/Proptize2.0.git
   cd Proptize2.0
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## 🏃‍♂️ Quick Start

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`
   - Enter your natural language query in the text area
   - Click "Optimize Prompt" to see the optimized version

## 🧩 Code Structure

```
Proptize2.0/
├── app.py              # Streamlit web application
├── optimizer.py        # Core optimization logic
├── data/               # Data files
│   ├── templates.json  # Prompt templates
│   └── samples.json    # Sample queries
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 🧠 How It Works

1. **Input Processing**: Takes natural language input from the user
2. **Intent Classification**: Determines the purpose of the query
3. **Entity Extraction**: Identifies key information in the query
4. **Template Selection**: Chooses the most appropriate template
5. **Prompt Generation**: Fills the template with extracted entities
6. **Output**: Returns the optimized prompt

## 📊 Features in Detail

### Natural Language Understanding
- Processes restaurant-specific terminology
- Handles variations in phrasing and synonyms
- Supports multiple languages (with appropriate spaCy models)

### Intent Classification
- Categorizes queries into predefined intents:
  - Menu management (add, update, remove items)
  - Inventory tracking
  - Sales analytics
  - Staff management
  - Customer orders

### Entity Recognition
- Extracts various entity types:
  - Menu items
  - Quantities and measurements
  - Prices and costs
  - Time periods and dates
  - Staff roles and names

## 🧪 Testing

To run the test suite:

```bash
python -m pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- NLP powered by [spaCy](https://spacy.io/)
- Template-based approach inspired by modern prompt engineering practices

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
