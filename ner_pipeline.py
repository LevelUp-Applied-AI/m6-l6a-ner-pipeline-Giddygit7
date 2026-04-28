"""
Module 6 Week A — Lab: NER Pipeline

Build and compare Named Entity Recognition pipelines using spaCy
and Hugging Face on climate-related text data.

Run: python ner_pipeline.py
"""

import pandas as pd
import numpy as np
import spacy
from transformers import pipeline as hf_pipeline
import unicodedata
from collections import defaultdict


def load_data(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with columns: id, text, source, language, category.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None


def explore_data(df):
    """Summarize basic corpus statistics.

    Args:
        df: DataFrame returned by load_data.

    Returns:
        Dictionary with keys:
          'shape': tuple (n_rows, n_cols)
          'lang_counts': dict mapping language code -> row count
          'category_counts': dict mapping category -> row count
          'text_length_stats': dict with 'mean', 'min', 'max' word counts
    """
    # Calculate shape
    shape = df.shape
    
    # Count languages
    lang_counts = df['language'].value_counts().to_dict()
    
    # Count categories
    category_counts = df['category'].value_counts().to_dict()
    
    # Calculate word counts for each text
    word_counts = df['text'].str.split().str.len()
    
    text_length_stats = {
        'mean': float(word_counts.mean()),
        'min': int(word_counts.min()),
        'max': int(word_counts.max())
    }
    
    return {
        'shape': shape,
        'lang_counts': lang_counts,
        'category_counts': category_counts,
        'text_length_stats': text_length_stats
    }


def preprocess_text(text, nlp):
    """Preprocess a single text string for NLP analysis.

    Normalize Unicode, lowercase, remove punctuation, tokenize,
    and lemmatize using the injected spaCy pipeline.

    Args:
        text: Raw text string.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        List of cleaned, lemmatized token strings.
    """
    # Normalize to NFC
    normalized_text = unicodedata.normalize('NFC', text)
    
    # Process with spaCy
    doc = nlp(normalized_text)
    
    # Extract lemmas, exclude punctuation and whitespace
    tokens = [
        token.lemma_.lower() 
        for token in doc 
        if not token.is_punct and not token.is_space
    ]
    
    return tokens


def extract_spacy_entities(df, nlp):
    """Extract named entities from English texts using spaCy NER.

    Args:
        df: DataFrame with columns id, text, language, ...
        nlp: A loaded spaCy Language object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # Filter to English only
    english_df = df[df['language'] == 'en'].copy()
    
    entities = []
    
    for _, row in english_df.iterrows():
        doc = nlp(row['text'])
        
        for ent in doc.ents:
            entities.append({
                'text_id': row['id'],
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
    
    if not entities:
        return pd.DataFrame(columns=['text_id', 'entity_text', 'entity_label', 'start_char', 'end_char'])
    
    return pd.DataFrame(entities)


def extract_hf_entities(df, ner_pipeline):
    """Extract named entities from English texts using Hugging Face NER.

    Uses the injected HF pipeline (expected: dslim/bert-base-NER).

    Args:
        df: DataFrame with columns id, text, language, ...
        ner_pipeline: A loaded Hugging Face `pipeline('ner', ...)` object.

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    # Filter to English only
    english_df = df[df['language'] == 'en'].copy()
    
    all_entities = []
    
    for _, row in english_df.iterrows():
        raw_entities = ner_pipeline(row['text'])
        
        if not raw_entities:
            continue
        
        # Merge subword tokens (those starting with ##)
        merged_entities = []
        current_entity = None
        
        for ent in raw_entities:
            word = ent['word']
            
            # Handle subword tokens (##)
            if word.startswith('##'):
                if current_entity:
                    current_entity['word'] += word[2:]  # Remove ##
                    current_entity['end'] = ent['end']
                continue
            else:
                # Save previous entity if exists
                if current_entity:
                    merged_entities.append(current_entity)
                
                # Start new entity
                current_entity = {
                    'word': word,
                    'entity': ent['entity'],
                    'start': ent['start'],
                    'end': ent['end']
                }
        
        # Append last entity
        if current_entity:
            merged_entities.append(current_entity)
        
        # Strip IOB prefixes (B- and I-)
        for ent in merged_entities:
            label = ent['entity']
            if label.startswith('B-') or label.startswith('I-'):
                label = label[2:]  # Remove first two characters
            
            all_entities.append({
                'text_id': row['id'],
                'entity_text': ent['word'],
                'entity_label': label,
                'start_char': ent['start'],
                'end_char': ent['end']
            })
    
    if not all_entities:
        return pd.DataFrame(columns=['text_id', 'entity_text', 'entity_label', 'start_char', 'end_char'])
    
    return pd.DataFrame(all_entities)


def compare_ner_outputs(spacy_df, hf_df):
    """Compare entity extraction results from spaCy and Hugging Face.

    Args:
        spacy_df: DataFrame of spaCy entities (from extract_spacy_entities).
        hf_df: DataFrame of HF entities (from extract_hf_entities).

    Returns:
        Dictionary with keys:
          'spacy_counts': dict of entity_label -> count for spaCy
          'hf_counts': dict of entity_label -> count for HF
          'total_spacy': int total entities from spaCy
          'total_hf': int total entities from HF
          'both': set of (text_id, entity_text) tuples found by both systems
          'spacy_only': set of (text_id, entity_text) tuples found only by spaCy
          'hf_only': set of (text_id, entity_text) tuples found only by HF
    """
    # Count entities per label for spaCy
    spacy_counts = spacy_df['entity_label'].value_counts().to_dict()
    total_spacy = len(spacy_df)
    
    # Count entities per label for HF
    hf_counts = hf_df['entity_label'].value_counts().to_dict()
    total_hf = len(hf_df)
    
    # Create tuples for matching
    spacy_tuples = set(zip(spacy_df['text_id'], spacy_df['entity_text']))
    hf_tuples = set(zip(hf_df['text_id'], hf_df['entity_text']))
    
    # Calculate overlaps
    both = spacy_tuples & hf_tuples
    spacy_only = spacy_tuples - hf_tuples
    hf_only = hf_tuples - spacy_tuples
    
    # Print summary
    print("\n" + "="*50)
    print("NER COMPARISON SUMMARY")
    print("="*50)
    print(f"spaCy total entities: {total_spacy}")
    print(f"HF total entities: {total_hf}")
    print(f"\nspaCy counts by label: {spacy_counts}")
    print(f"HF counts by label: {hf_counts}")
    print(f"\nAgreement: {len(both)} entities found by both")
    print(f"spaCy-only: {len(spacy_only)}")
    print(f"HF-only: {len(hf_only)}")
    
    return {
        'spacy_counts': spacy_counts,
        'hf_counts': hf_counts,
        'total_spacy': total_spacy,
        'total_hf': total_hf,
        'both': both,
        'spacy_only': spacy_only,
        'hf_only': hf_only
    }


def evaluate_ner(predicted_df, gold_df):
    """Evaluate NER predictions against gold-standard annotations.

    Computes entity-level precision, recall, and F1. An entity is a
    true positive if both the entity text and label match a gold entry
    for the same text_id.

    Args:
        predicted_df: DataFrame with columns text_id, entity_text,
                      entity_label.
        gold_df: DataFrame with columns text_id, entity_text,
                 entity_label.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1' (floats 0-1).
    """
    # Filter predictions to only text_ids present in gold
    gold_text_ids = set(gold_df['text_id'])
    filtered_pred = predicted_df[predicted_df['text_id'].isin(gold_text_ids)]
    
    # Create sets of (text_id, entity_text, entity_label) for gold and predictions
    gold_set = set(zip(gold_df['text_id'], gold_df['entity_text'], gold_df['entity_label']))
    pred_set = set(zip(filtered_pred['text_id'], filtered_pred['entity_text'], filtered_pred['entity_label']))
    
    # Calculate true positives
    true_positives = len(gold_set & pred_set)
    
    # Calculate false positives and false negatives
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    # Compute precision, recall, f1 (avoid division by zero)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }


if __name__ == "__main__":
    # Load spaCy and HF models once, reuse across functions
    nlp = spacy.load("en_core_web_sm")
    hf_ner = hf_pipeline("ner", model="dslim/bert-base-NER")
    
    # Load multilingual model for Task 7
    multilingual_nlp = spacy.load("xx_ent_wiki_sm")
    
    # Load and explore
    df = load_data()
    if df is not None:
        summary = explore_data(df)
        if summary is not None:
            print(f"Shape: {summary['shape']}")
            print(f"Languages: {summary['lang_counts']}")
            print(f"Categories: {summary['category_counts']}")
            print(f"Text length (words): {summary['text_length_stats']}")
        
        # Preprocess a sample to verify your function
        sample_row = df[df["language"] == "en"].iloc[0]
        sample_tokens = preprocess_text(sample_row["text"], nlp)
        if sample_tokens is not None:
            print(f"\nSample preprocessed tokens: {sample_tokens[:10]}")
        
        # spaCy NER across the English corpus
        spacy_entities = extract_spacy_entities(df, nlp)
        if spacy_entities is not None:
            print(f"\nspaCy entities: {len(spacy_entities)} total")
        
        # HF NER across the English corpus
        hf_entities = extract_hf_entities(df, hf_ner)
        if hf_entities is not None:
            print(f"HF entities: {len(hf_entities)} total")
        
        # Compare the two systems
        if spacy_entities is not None and hf_entities is not None:
            comparison = compare_ner_outputs(spacy_entities, hf_entities)
            if comparison is not None:
                print(f"\nBoth systems agreed on {len(comparison['both'])} entities")
                print(f"spaCy-only: {len(comparison['spacy_only'])}")
                print(f"HF-only: {len(comparison['hf_only'])}")
        
        # Evaluate against gold standard
        gold = pd.read_csv("data/gold_entities.csv")
        if spacy_entities is not None:
            spacy_metrics = evaluate_ner(spacy_entities, gold)
            print(f"\nspaCy evaluation: {spacy_metrics}")
        
        if hf_entities is not None:
            hf_metrics = evaluate_ner(hf_entities, gold)
            print(f"HF evaluation: {hf_metrics}")
        
        # Task 7: Multilingual NER on Arabic
        print("\n" + "="*50)
        print("TASK 7: MULTILINGUAL NER ON ARABIC TEXTS")
        print("="*50)
        
        # Filter to Arabic rows
        arabic_df = df[df['language'] == 'ar'].copy()
        print(f"Arabic texts found: {len(arabic_df)}")
        
        # Extract Arabic entities using multilingual model
        arabic_entities = []
        for _, row in arabic_df.iterrows():
            doc = multilingual_nlp(row['text'])
            for ent in doc.ents:
                arabic_entities.append({
                    'text_id': row['id'],
                    'entity_text': ent.text,
                    'entity_label': ent.label_,
                    'start_char': ent.start_char,
                    'end_char': ent.end_char
                })
        
        if arabic_entities:
            arabic_entities_df = pd.DataFrame(arabic_entities)
            print(f"\nArabic entities extracted: {len(arabic_entities_df)}")
            print(f"Entity label distribution: {arabic_entities_df['entity_label'].value_counts().to_dict()}")
            print(f"\nSample Arabic entities (first 5):")
            print(arabic_entities_df[['entity_text', 'entity_label']].head())
        else:
            print("No Arabic entities found.")
        
        print("\nNote: Arabic NER using xx_ent_wiki_sm shows different labels (PER, LOC, ORG, MISC)")
        print("compared to English-only models (PERSON, GPE, ORG, DATE, etc.). Multilingual")
        print("models are essential for non-English text - single-language pipelines fail.")