"""
Document Processing Synthetic Data Generator

Generates realistic synthetic financial document data including:
- Invoices
- Bank statements
- Tax forms
- Contracts

Creates structured text data that can be used for NLP and OCR training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random


class DocumentProcessingDataGenerator:
    """Generate synthetic financial document data."""

    def __init__(self, random_state: Optional[int] = 42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        # Sample data
        self.company_names = [
            "Acme Corporation", "Global Tech Solutions", "Premier Services LLC",
            "Advanced Systems Inc", "Metro Business Group", "First National Suppliers",
            "Central Trading Company", "Elite Professional Services", "Summit Industries",
            "Pacific Ventures Ltd"
        ]

        self.customer_names = [
            "John Smith", "Jane Doe", "Michael Johnson", "Emily Williams", "David Brown",
            "Sarah Davis", "Robert Miller", "Jennifer Wilson", "William Moore", "Lisa Taylor"
        ]

        self.product_services = [
            "Consulting Services", "Software License", "Hardware Equipment", "Maintenance Contract",
            "Professional Services", "Technical Support", "Training Program", "Cloud Services",
            "Data Analysis", "Project Management", "Marketing Services", "Legal Services"
        ]

        self.bank_names = [
            "First National Bank", "Global Trust Bank", "Metro Credit Union",
            "Pacific Bank", "Central Federal Bank", "United Community Bank"
        ]

    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic document dataset.

        Args:
            n_samples: Number of documents to generate

        Returns:
            DataFrame with document metadata and extracted fields
        """
        documents = []

        # Distribute document types
        n_invoices = int(n_samples * 0.4)
        n_statements = int(n_samples * 0.3)
        n_tax_forms = int(n_samples * 0.2)
        n_contracts = n_samples - n_invoices - n_statements - n_tax_forms

        # Generate each document type
        documents.extend(self._generate_invoices(n_invoices))
        documents.extend(self._generate_bank_statements(n_statements))
        documents.extend(self._generate_tax_forms(n_tax_forms))
        documents.extend(self._generate_contracts(n_contracts))

        df = pd.DataFrame(documents)
        return df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

    def _generate_invoices(self, n: int) -> List[Dict]:
        """Generate invoice documents."""
        invoices = []

        for i in range(n):
            invoice_number = f"INV-{random.randint(10000, 99999)}"
            vendor = random.choice(self.company_names)
            customer = random.choice(self.customer_names)
            invoice_date = datetime.now() - timedelta(days=random.randint(0, 365))
            due_date = invoice_date + timedelta(days=random.choice([15, 30, 45, 60]))

            # Generate line items
            n_items = random.randint(1, 5)
            line_items = []
            subtotal = 0

            for _ in range(n_items):
                item = random.choice(self.product_services)
                quantity = random.randint(1, 10)
                unit_price = round(random.uniform(50, 5000), 2)
                line_total = round(quantity * unit_price, 2)
                subtotal += line_total

                line_items.append({
                    "description": item,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "total": line_total
                })

            tax_rate = 0.08
            tax = round(subtotal * tax_rate, 2)
            total = round(subtotal + tax, 2)

            # Create document text
            text = f"""
INVOICE

Invoice Number: {invoice_number}
Invoice Date: {invoice_date.strftime('%m/%d/%Y')}
Due Date: {due_date.strftime('%m/%d/%Y')}

BILL TO:
{customer}

FROM:
{vendor}

ITEMS:
"""
            for item in line_items:
                text += f"{item['description']} - Qty: {item['quantity']} x ${item['unit_price']:.2f} = ${item['total']:.2f}\n"

            text += f"""
Subtotal: ${subtotal:.2f}
Tax (8%): ${tax:.2f}
TOTAL DUE: ${total:.2f}

Payment Terms: Net {(due_date - invoice_date).days} days
            """

            invoices.append({
                "document_id": f"DOC_{i:06d}",
                "document_type": "invoice",
                "text": text.strip(),
                "vendor": vendor,
                "customer": customer,
                "invoice_number": invoice_number,
                "invoice_date": invoice_date.strftime('%Y-%m-%d'),
                "due_date": due_date.strftime('%Y-%m-%d'),
                "amount": total,
                "tax_amount": tax,
                "subtotal": subtotal
            })

        return invoices

    def _generate_bank_statements(self, n: int) -> List[Dict]:
        """Generate bank statement documents."""
        statements = []

        for i in range(n):
            account_number = f"{random.randint(1000000, 9999999)}"
            account_holder = random.choice(self.customer_names)
            bank_name = random.choice(self.bank_names)
            statement_date = datetime.now() - timedelta(days=random.randint(0, 365))
            statement_month = statement_date.strftime('%B %Y')

            # Generate transactions
            n_transactions = random.randint(10, 30)
            transactions = []
            balance = round(random.uniform(5000, 50000), 2)
            opening_balance = balance

            for _ in range(n_transactions):
                trans_date = statement_date - timedelta(days=random.randint(0, 30))
                trans_type = random.choice(["Deposit", "Withdrawal", "Transfer", "Payment"])
                description = random.choice([
                    "Direct Deposit", "ATM Withdrawal", "Wire Transfer", "Check Payment",
                    "Debit Card Purchase", "Online Transfer", "Bill Payment"
                ])
                amount = round(random.uniform(10, 2000), 2)

                if trans_type in ["Deposit", "Transfer"]:
                    balance += amount
                else:
                    balance -= amount

                transactions.append({
                    "date": trans_date.strftime('%m/%d/%Y'),
                    "description": description,
                    "type": trans_type,
                    "amount": amount,
                    "balance": round(balance, 2)
                })

            closing_balance = balance

            # Create document text
            text = f"""
{bank_name}
BANK STATEMENT

Account Holder: {account_holder}
Account Number: {account_number}
Statement Period: {statement_month}
Statement Date: {statement_date.strftime('%m/%d/%Y')}

ACCOUNT SUMMARY:
Opening Balance: ${opening_balance:.2f}
Closing Balance: ${closing_balance:.2f}

TRANSACTIONS:
"""
            for trans in transactions:
                text += f"{trans['date']} - {trans['description']} ({trans['type']}) - ${trans['amount']:.2f} - Balance: ${trans['balance']:.2f}\n"

            statements.append({
                "document_id": f"DOC_{i:06d}",
                "document_type": "bank_statement",
                "text": text.strip(),
                "bank_name": bank_name,
                "account_number": account_number,
                "account_holder": account_holder,
                "statement_date": statement_date.strftime('%Y-%m-%d'),
                "opening_balance": opening_balance,
                "closing_balance": closing_balance,
                "n_transactions": n_transactions
            })

        return statements

    def _generate_tax_forms(self, n: int) -> List[Dict]:
        """Generate tax form documents."""
        tax_forms = []

        for i in range(n):
            tax_id = f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}"
            taxpayer_name = random.choice(self.customer_names)
            tax_year = random.randint(2020, 2024)

            # Generate income and deductions
            wages = round(random.uniform(30000, 150000), 2)
            interest = round(random.uniform(100, 5000), 2)
            dividends = round(random.uniform(0, 10000), 2)
            total_income = round(wages + interest + dividends, 2)

            standard_deduction = 12950
            taxable_income = round(max(0, total_income - standard_deduction), 2)

            # Simple tax calculation
            if taxable_income <= 10275:
                tax = round(taxable_income * 0.10, 2)
            elif taxable_income <= 41775:
                tax = round(1027.50 + (taxable_income - 10275) * 0.12, 2)
            else:
                tax = round(4807.50 + (taxable_income - 41775) * 0.22, 2)

            refund_or_owed = round(random.uniform(-5000, 5000), 2)

            # Create document text
            text = f"""
U.S. INDIVIDUAL INCOME TAX RETURN
FORM 1040

Tax Year: {tax_year}
Taxpayer Name: {taxpayer_name}
Social Security Number: {tax_id}

INCOME:
Wages, salaries, tips: ${wages:.2f}
Interest income: ${interest:.2f}
Dividend income: ${dividends:.2f}
Total Income: ${total_income:.2f}

DEDUCTIONS:
Standard Deduction: ${standard_deduction:.2f}

TAXABLE INCOME: ${taxable_income:.2f}

TAX LIABILITY: ${tax:.2f}

{"REFUND" if refund_or_owed > 0 else "AMOUNT OWED"}: ${abs(refund_or_owed):.2f}
            """

            tax_forms.append({
                "document_id": f"DOC_{i:06d}",
                "document_type": "tax_form",
                "text": text.strip(),
                "taxpayer_name": taxpayer_name,
                "tax_id": tax_id,
                "tax_year": tax_year,
                "total_income": total_income,
                "taxable_income": taxable_income,
                "tax_liability": tax,
                "refund_or_owed": refund_or_owed
            })

        return tax_forms

    def _generate_contracts(self, n: int) -> List[Dict]:
        """Generate contract documents."""
        contracts = []

        for i in range(n):
            contract_number = f"CTR-{random.randint(1000, 9999)}"
            party_a = random.choice(self.company_names)
            party_b = random.choice(self.customer_names)
            contract_date = datetime.now() - timedelta(days=random.randint(0, 365))
            start_date = contract_date
            end_date = start_date + timedelta(days=random.choice([365, 730, 1095]))
            contract_value = round(random.uniform(10000, 500000), 2)

            service = random.choice(self.product_services)

            # Create document text
            text = f"""
SERVICE CONTRACT

Contract Number: {contract_number}
Contract Date: {contract_date.strftime('%m/%d/%Y')}

PARTIES:
Party A (Provider): {party_a}
Party B (Client): {party_b}

TERMS:
Service: {service}
Start Date: {start_date.strftime('%m/%d/%Y')}
End Date: {end_date.strftime('%m/%d/%Y')}
Contract Value: ${contract_value:.2f}

This agreement outlines the terms and conditions for the provision of
{service} services. The provider agrees to deliver services according
to the specifications outlined in Exhibit A. Payment terms are net 30 days
from invoice date.

Both parties agree to the terms and conditions set forth in this contract.
            """

            contracts.append({
                "document_id": f"DOC_{i:06d}",
                "document_type": "contract",
                "text": text.strip(),
                "contract_number": contract_number,
                "party_a": party_a,
                "party_b": party_b,
                "contract_date": contract_date.strftime('%Y-%m-%d'),
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d'),
                "contract_value": contract_value,
                "service": service
            })

        return contracts


def main():
    """Example usage of the document processing data generator."""
    generator = DocumentProcessingDataGenerator(random_state=42)
    df = generator.generate(n_samples=1000)

    print("Generated Document Processing Dataset")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nDocument Type Distribution:")
    print(df['document_type'].value_counts())
    print(f"\nFirst few rows:")
    print(df[['document_id', 'document_type', 'amount']].head(10))
    print(f"\nSample Invoice Text:")
    invoice_sample = df[df['document_type'] == 'invoice'].iloc[0]
    print(invoice_sample['text'])

    # Save to CSV
    df.to_csv("data/raw/document_processing_synthetic.csv", index=False)
    print(f"\nData saved to data/raw/document_processing_synthetic.csv")


if __name__ == "__main__":
    main()
