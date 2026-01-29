"""
Credit Risk Synthetic Data Generator

Generates realistic synthetic data for credit risk modeling including:
- Borrower demographics
- Financial metrics
- Loan characteristics
- Default labels
"""

import numpy as np
import pandas as pd
from typing import Optional


class CreditRiskDataGenerator:
    """Generate synthetic credit risk data for modeling."""

    def __init__(self, random_state: Optional[int] = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate(self, n_samples: int = 10000, default_rate: float = 0.15) -> pd.DataFrame:
        """
        Generate synthetic credit risk dataset.

        Args:
            n_samples: Number of samples to generate
            default_rate: Target default rate (0-1)

        Returns:
            DataFrame with borrower features and default labels
        """
        # Borrower demographics
        age = np.random.normal(40, 12, n_samples).clip(18, 75)
        employment_length = np.random.exponential(5, n_samples).clip(0, 40)

        # Financial metrics
        income = np.random.lognormal(10.5, 0.6, n_samples).clip(20000, 500000)
        credit_score = np.random.normal(680, 80, n_samples).clip(300, 850)

        # Loan characteristics
        loan_amount = np.random.lognormal(10, 0.8, n_samples).clip(1000, 100000)
        loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15])
        interest_rate = np.random.normal(0.12, 0.05, n_samples).clip(0.03, 0.30)

        # Home ownership
        home_ownership = np.random.choice(
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            n_samples,
            p=[0.35, 0.15, 0.45, 0.05]
        )

        # Loan purpose
        loan_purpose = np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business", "other"],
            n_samples,
            p=[0.35, 0.20, 0.15, 0.15, 0.10, 0.05]
        )

        # Calculate debt-to-income ratio
        monthly_income = income / 12
        monthly_payment = (loan_amount * (interest_rate / 12)) / (1 - (1 + interest_rate / 12) ** -loan_term)
        existing_debt = np.random.uniform(0.1, 0.4, n_samples) * monthly_income
        debt_to_income = (monthly_payment + existing_debt) / monthly_income

        # Calculate number of delinquencies
        delinquencies_2yrs = np.random.poisson(0.5, n_samples).clip(0, 10)

        # Number of open accounts
        open_accounts = np.random.poisson(8, n_samples).clip(1, 30)

        # Total credit lines
        total_credit_lines = np.random.poisson(15, n_samples).clip(1, 50)

        # Revolving balance
        revolving_balance = np.random.lognormal(8, 1.5, n_samples).clip(0, 100000)

        # Revolving utilization
        revolving_utilization = np.random.beta(2, 5, n_samples).clip(0, 1)

        # Calculate default probability based on risk factors
        default_prob = self._calculate_default_probability(
            age, income, credit_score, debt_to_income, employment_length,
            delinquencies_2yrs, revolving_utilization
        )

        # Adjust to match target default rate
        threshold = np.percentile(default_prob, (1 - default_rate) * 100)
        default = (default_prob > threshold).astype(int)

        # Create DataFrame
        df = pd.DataFrame({
            "age": age.round(0).astype(int),
            "employment_length": employment_length.round(1),
            "annual_income": income.round(2),
            "credit_score": credit_score.round(0).astype(int),
            "loan_amount": loan_amount.round(2),
            "loan_term": loan_term,
            "interest_rate": interest_rate.round(4),
            "debt_to_income": debt_to_income.round(4),
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "delinquencies_2yrs": delinquencies_2yrs,
            "open_accounts": open_accounts,
            "total_credit_lines": total_credit_lines,
            "revolving_balance": revolving_balance.round(2),
            "revolving_utilization": revolving_utilization.round(4),
            "default": default
        })

        return df

    def _calculate_default_probability(
        self,
        age: np.ndarray,
        income: np.ndarray,
        credit_score: np.ndarray,
        debt_to_income: np.ndarray,
        employment_length: np.ndarray,
        delinquencies_2yrs: np.ndarray,
        revolving_utilization: np.ndarray
    ) -> np.ndarray:
        """
        Calculate default probability based on risk factors.

        Uses a logistic function with weighted features.
        """
        # Normalize features
        age_norm = (age - 40) / 12
        income_norm = (np.log(income) - 10.5) / 0.6
        credit_score_norm = (credit_score - 680) / 80
        dti_norm = (debt_to_income - 0.3) / 0.15
        employment_norm = (employment_length - 5) / 5
        delinq_norm = delinquencies_2yrs / 2
        revol_util_norm = (revolving_utilization - 0.3) / 0.2

        # Calculate risk score (higher = more risky)
        risk_score = (
            -0.3 * age_norm +
            -0.5 * income_norm +
            -0.8 * credit_score_norm +
            0.6 * dti_norm +
            -0.2 * employment_norm +
            0.4 * delinq_norm +
            0.3 * revol_util_norm
        )

        # Add some noise
        risk_score += np.random.normal(0, 0.5, len(age))

        # Convert to probability using sigmoid
        prob = 1 / (1 + np.exp(-risk_score))

        return prob


def main():
    """Example usage of the credit risk data generator."""
    generator = CreditRiskDataGenerator(random_state=42)
    df = generator.generate(n_samples=10000, default_rate=0.15)

    print("Generated Credit Risk Dataset")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nDefault Rate: {df['default'].mean():.2%}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())
    print(f"\nSummary Statistics:")
    print(df.describe())

    # Save to CSV
    df.to_csv("data/raw/credit_risk_synthetic.csv", index=False)
    print(f"\nData saved to data/raw/credit_risk_synthetic.csv")


if __name__ == "__main__":
    main()
