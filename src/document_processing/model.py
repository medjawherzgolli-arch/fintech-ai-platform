"""
Document Processing Pipeline

NLP and text extraction pipeline for financial documents including:
- Text preprocessing
- Named Entity Recognition (NER)
- Field extraction (amounts, dates, names, account numbers)
- Document classification
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import joblib


class DocumentProcessor:
    """Process and extract information from financial documents."""

    def __init__(self):
        """Initialize the document processor."""
        self.document_types = ['invoice', 'bank_statement', 'tax_form', 'contract']

        # Regex patterns for extraction
        self.patterns = {
            'amount': r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            'date': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'invoice_number': r'(?:Invoice|INV)[\s#:-]*([A-Z0-9-]+)',
            'account_number': r'(?:Account|Acct)[\s#:-]*(\d{7,})',
            'tax_id': r'(\d{2}-\d{7})',
            'contract_number': r'(?:Contract|CTR)[\s#:-]*([A-Z0-9-]+)',
            'email': r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'phone': r'(\d{3}[-.]?\d{3}[-.]?\d{4})'
        }

    def extract_fields(self, text: str, document_type: str) -> Dict[str, Any]:
        """
        Extract structured fields from document text.

        Args:
            text: Document text
            document_type: Type of document

        Returns:
            Dictionary of extracted fields
        """
        extracted = {}

        # Extract amounts
        amounts = self._extract_pattern(text, 'amount')
        if amounts:
            # Clean and convert to float
            amounts_float = [float(a.replace(',', '')) for a in amounts]
            extracted['amounts'] = amounts_float
            extracted['total_amount'] = max(amounts_float) if amounts_float else None
            extracted['amount_count'] = len(amounts_float)

        # Extract dates
        dates = self._extract_pattern(text, 'date')
        extracted['dates'] = dates
        extracted['date_count'] = len(dates)

        # Document-specific extraction
        if document_type == 'invoice':
            invoice_num = self._extract_pattern(text, 'invoice_number')
            extracted['invoice_number'] = invoice_num[0] if invoice_num else None

        elif document_type == 'bank_statement':
            account_num = self._extract_pattern(text, 'account_number')
            extracted['account_number'] = account_num[0] if account_num else None

        elif document_type == 'tax_form':
            tax_id = self._extract_pattern(text, 'tax_id')
            extracted['tax_id'] = tax_id[0] if tax_id else None

        elif document_type == 'contract':
            contract_num = self._extract_pattern(text, 'contract_number')
            extracted['contract_number'] = contract_num[0] if contract_num else None

        # Extract contact information
        emails = self._extract_pattern(text, 'email')
        phones = self._extract_pattern(text, 'phone')
        extracted['emails'] = emails
        extracted['phones'] = phones

        # Extract names (simple heuristic - capitalized words)
        names = self._extract_names(text)
        extracted['potential_names'] = names[:5]  # Top 5 potential names

        # Text statistics
        extracted['word_count'] = len(text.split())
        extracted['char_count'] = len(text)
        extracted['line_count'] = len(text.split('\n'))

        return extracted

    def _extract_pattern(self, text: str, pattern_name: str) -> List[str]:
        """Extract all matches for a given pattern."""
        pattern = self.patterns.get(pattern_name)
        if not pattern:
            return []

        matches = re.findall(pattern, text, re.IGNORECASE)
        return matches

    def _extract_names(self, text: str) -> List[str]:
        """
        Extract potential person/company names.
        Simple heuristic: sequences of 2-4 capitalized words.
        """
        # Pattern for capitalized words
        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b'
        matches = re.findall(pattern, text)

        # Filter out common non-names
        exclude = {'Invoice', 'Date', 'Total', 'Amount', 'Account', 'Balance', 'Statement', 'Tax', 'Form'}
        names = [m for m in matches if not any(word in m for word in exclude)]

        # Count occurrences and return most common
        from collections import Counter
        name_counts = Counter(names)
        return [name for name, count in name_counts.most_common(10)]

    def classify_document(self, text: str) -> Tuple[str, float]:
        """
        Classify document type using keyword matching.

        Args:
            text: Document text

        Returns:
            Tuple of (document_type, confidence)
        """
        text_lower = text.lower()

        # Keyword scores for each document type
        scores = {
            'invoice': 0,
            'bank_statement': 0,
            'tax_form': 0,
            'contract': 0
        }

        # Invoice keywords
        invoice_keywords = ['invoice', 'bill to', 'from:', 'due date', 'payment terms', 'subtotal']
        scores['invoice'] = sum(1 for kw in invoice_keywords if kw in text_lower)

        # Bank statement keywords
        statement_keywords = ['bank statement', 'account number', 'opening balance', 'closing balance', 'transaction']
        scores['bank_statement'] = sum(1 for kw in statement_keywords if kw in text_lower)

        # Tax form keywords
        tax_keywords = ['tax return', 'taxable income', 'deduction', 'irs', 'form 1040', 'social security']
        scores['tax_form'] = sum(1 for kw in tax_keywords if kw in text_lower)

        # Contract keywords
        contract_keywords = ['contract', 'agreement', 'parties', 'terms and conditions', 'hereby', 'whereas']
        scores['contract'] = sum(1 for kw in contract_keywords if kw in text_lower)

        # Determine best match
        if max(scores.values()) == 0:
            return 'unknown', 0.0

        doc_type = max(scores, key=scores.get)
        confidence = scores[doc_type] / sum(scores.values())

        return doc_type, confidence

    def process_document(self, text: str, document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document end-to-end.

        Args:
            text: Document text
            document_type: Known document type (if None, will classify)

        Returns:
            Dictionary with classification and extracted fields
        """
        # Classify if type not provided
        if document_type is None:
            document_type, confidence = self.classify_document(text)
        else:
            confidence = 1.0

        # Extract fields
        extracted_fields = self.extract_fields(text, document_type)

        # Combine results
        result = {
            'document_type': document_type,
            'classification_confidence': confidence,
            **extracted_fields
        }

        return result

    def process_batch(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Process multiple documents in batch.

        Args:
            df: DataFrame with documents
            text_column: Name of column containing document text

        Returns:
            DataFrame with extracted information
        """
        results = []

        for idx, row in df.iterrows():
            text = row[text_column]
            doc_type = row.get('document_type', None)

            result = self.process_document(text, doc_type)
            result['document_id'] = row.get('document_id', idx)

            results.append(result)

        return pd.DataFrame(results)

    def save(self, filepath: str):
        """Save the processor."""
        joblib.dump({
            'patterns': self.patterns,
            'document_types': self.document_types
        }, filepath)

    def load(self, filepath: str):
        """Load the processor."""
        data = joblib.load(filepath)
        self.patterns = data['patterns']
        self.document_types = data['document_types']


def main():
    """Example usage of the document processor."""
    from src.data_generators.document_processing_generator import DocumentProcessingDataGenerator

    # Generate data
    print("Generating synthetic documents...")
    generator = DocumentProcessingDataGenerator(random_state=42)
    df = generator.generate(n_samples=100)

    # Initialize processor
    processor = DocumentProcessor()

    # Process documents
    print("\nProcessing documents...")
    results = processor.process_batch(df)

    print("\nProcessing Results:")
    print("=" * 50)
    print(f"Total documents processed: {len(results)}")
    print(f"\nDocument type distribution:")
    print(results['document_type'].value_counts())

    print("\nSample extracted fields:")
    print(results[['document_id', 'document_type', 'total_amount', 'date_count', 'word_count']].head(10))

    # Test single document processing
    print("\n" + "=" * 50)
    print("Single Document Processing Example:")
    print("=" * 50)
    sample_doc = df.iloc[0]
    print(f"\nDocument Type: {sample_doc['document_type']}")
    print(f"\nDocument Text (first 500 chars):")
    print(sample_doc['text'][:500])

    result = processor.process_document(sample_doc['text'], sample_doc['document_type'])
    print(f"\nExtracted Information:")
    for key, value in result.items():
        if key not in ['amounts', 'dates', 'potential_names', 'emails', 'phones']:
            print(f"  {key}: {value}")

    # Save processor
    processor.save("models/document_processing/document_processor.pkl")
    print("\nProcessor saved to models/document_processing/document_processor.pkl")


if __name__ == "__main__":
    main()
