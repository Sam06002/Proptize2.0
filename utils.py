import spacy

def get_spacy_model():
    """
    Load the spaCy model.
    
    Returns:
        spacy.Language: Loaded spaCy model
        
    Raises:
        OSError: If the model is not found. Please install it using:
                python -m spacy download en_core_web_sm
    """
    try:
        return spacy.load("en_core_web_sm")
    except OSError as e:
        raise OSError(
            "The 'en_core_web_sm' model is not installed. "
            "Please install it manually using:\n"
            "    python -m spacy download en_core_web_sm"
        ) from e
