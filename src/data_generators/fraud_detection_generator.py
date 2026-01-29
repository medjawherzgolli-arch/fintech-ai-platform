"""
Fraud Detection Synthetic Data Generator

Generates realistic synthetic transaction data for fraud detection including:
- Transaction details (amount, time, merchant)
- Customer behavior patterns
- Fraud labels with realistic patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional


class FraudDetectionDataGenerator:
    """Generate synthetic fraud detection data for modeling."""

    def __init__(self, random_state: Optional[int] = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate(self, n_samples: int = 50000, fraud_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate synthetic fraud detection dataset.

        Args:
            n_samples: Number of transactions to generate
            fraud_rate: Target fraud rate (0-1)

        Returns:
            DataFrame with transaction features and fraud labels
        """
        # Generate transaction IDs
        transaction_ids = [f"TXN_{str(i).zfill(8)}" for i in range(n_samples)]

        # Generate customer IDs (fewer customers than transactions)
        n_customers = int(n_samples * 0.3)
        customer_ids = np.random.choice([f"CUST_{str(i).zfill(6)}" for i in range(n_customers)], n_samples)

        # Generate timestamps (last 90 days)
        start_date = datetime.now() - timedelta(days=90)
        timestamps = [start_date + timedelta(
            days=np.random.uniform(0, 90),
            hours=np.random.uniform(0, 24),
            minutes=np.random.uniform(0, 60)
        ) for _ in range(n_samples)]

        # Extract time features
        transaction_hour = np.array([t.hour for t in timestamps])
        transaction_day = np.array([t.weekday() for t in timestamps])
        transaction_month = np.array([t.month for t in timestamps])

        # Transaction amounts (lognormal distribution)
        transaction_amount = np.random.lognormal(4, 1.5, n_samples).clip(1, 10000).round(2)

        # Merchant categories
        merchant_categories = np.random.choice(
            ["grocery", "gas_station", "restaurant", "online_retail", "electronics", "travel", "entertainment", "healthcare", "other"],
            n_samples,
            p=[0.20, 0.15, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05, 0.05]
        )

        # Card present vs card not present
        card_present = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])

        # Distance from home (miles)
        distance_from_home = np.random.exponential(10, n_samples).clip(0, 500).round(1)

        # Transaction frequency (transactions in last 24 hours)
        transaction_frequency = np.random.poisson(2, n_samples).clip(0, 20)

        # Average transaction amount for customer
        avg_transaction_amount = np.random.lognormal(3.5, 1, n_samples).clip(10, 1000).round(2)

        # Days since last transaction
        days_since_last_transaction = np.random.exponential(3, n_samples).clip(0, 30).round(1)

        # Number of transactions in last 7 days
        transactions_last_7days = np.random.poisson(5, n_samples).clip(0, 50)

        # Number of failed transactions in last 24 hours
        failed_transactions_24h = np.random.poisson(0.2, n_samples).clip(0, 10)

        # Merchant risk score
        merchant_risk_score = np.random.beta(2, 8, n_samples).clip(0, 1).round(3)

        # International transaction
        international = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

        # Device ID changes
        device_id_change = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])

        # Calculate fraud probability based on risk factors
        fraud_prob = self._calculate_fraud_probability(
            transaction_amount, transaction_hour, card_present, distance_from_home,
            transaction_frequency, avg_transaction_amount, failed_transactions_24h,
            merchant_risk_score, international, device_id_change, merchant_categories
        )

        # Adjust to match target fraud rate
        threshold = np.percentile(fraud_prob, (1 - fraud_rate) * 100)
        is_fraud = (fraud_prob > threshold).astype(int)

        # For fraud cases, adjust features to make them more suspicious
        fraud_indices = np.where(is_fraud == 1)[0]
        if len(fraud_indices) > 0:
            # Increase transaction amounts for fraud
            transaction_amount[fraud_indices] *= np.random.uniform(1.5, 3, len(fraud_indices))
            # More likely to be card-not-present
            card_present[fraud_indices] = np.random.choice([0, 1], len(fraud_indices), p=[0.8, 0.2])
            # Higher distances from home
            distance_from_home[fraud_indices] *= np.random.uniform(2, 5, len(fraud_indices))
            # More international transactions
            international[fraud_indices] = np.random.choice([0, 1], len(fraud_indices), p=[0.3, 0.7])

        # Create DataFrame
        df = pd.DataFrame({
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "timestamp": timestamps,
            "transaction_amount": transaction_amount.round(2),
            "transaction_hour": transaction_hour,
            "transaction_day": transaction_day,
            "transaction_month": transaction_month,
            "merchant_category": merchant_categories,
            "card_present": card_present,
            "distance_from_home": distance_from_home.round(1),
            "transaction_frequency_24h": transaction_frequency,
            "avg_transaction_amount": avg_transaction_amount.round(2),
            "days_since_last_transaction": days_since_last_transaction,
            "transactions_last_7days": transactions_last_7days,
            "failed_transactions_24h": failed_transactions_24h,
            "merchant_risk_score": merchant_risk_score,
            "international": international,
            "device_id_change": device_id_change,
            "is_fraud": is_fraud
        })

        return df.sort_values("timestamp").reset_index(drop=True)

    def _calculate_fraud_probability(
        self,
        amount: np.ndarray,
        hour: np.ndarray,
        card_present: np.ndarray,
        distance: np.ndarray,
        frequency: np.ndarray,
        avg_amount: np.ndarray,
        failed_txns: np.ndarray,
        merchant_risk: np.ndarray,
        international: np.ndarray,
        device_change: np.ndarray,
        categories: np.ndarray
    ) -> np.ndarray:
        """
        Calculate fraud probability based on risk factors.
        """
        # Normalize features
        amount_ratio = amount / (avg_amount + 1)
        amount_norm = np.log1p(amount_ratio)

        # Unusual hours (late night/early morning more risky)
        hour_risk = np.where((hour >= 0) & (hour <= 6), 1.5,
                    np.where((hour >= 22) & (hour <= 23), 1.2, 1.0))

        # Card not present is riskier
        card_risk = np.where(card_present == 0, 1.3, 1.0)

        # High distance is risky
        distance_risk = np.log1p(distance) / 3

        # High frequency is risky
        frequency_risk = np.log1p(frequency) / 2

        # Failed transactions are risky
        failed_risk = failed_txns * 0.5

        # Calculate risk score
        risk_score = (
            0.4 * amount_norm +
            0.2 * hour_risk +
            0.2 * card_risk +
            0.3 * distance_risk +
            0.2 * frequency_risk +
            0.3 * failed_risk +
            0.4 * merchant_risk +
            0.3 * international +
            0.3 * device_change
        )

        # Add some noise
        risk_score += np.random.normal(0, 0.3, len(amount))

        # Convert to probability using sigmoid
        prob = 1 / (1 + np.exp(-risk_score + 3))  # Shift to make fraud rare

        return prob


def main():
    """Example usage of the fraud detection data generator."""
    generator = FraudDetectionDataGenerator(random_state=42)
    df = generator.generate(n_samples=50000, fraud_rate=0.02)

    print("Generated Fraud Detection Dataset")
    print("=" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nFraud Rate: {df['is_fraud'].mean():.2%}")
    print(f"\nFraud vs Non-Fraud Transaction Amounts:")
    print(df.groupby('is_fraud')['transaction_amount'].describe())
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset Info:")
    print(df.info())

    # Save to CSV
    df.to_csv("data/raw/fraud_detection_synthetic.csv", index=False)
    print(f"\nData saved to data/raw/fraud_detection_synthetic.csv")


if __name__ == "__main__":
    main()
