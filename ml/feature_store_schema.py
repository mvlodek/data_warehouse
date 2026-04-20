"""Feature store schema creation and management."""

from sqlalchemy import text


def create_feature_store_schema(engine):
    """Create feature store tables for technical indicators and macro ratios."""
    
    with engine.connect() as conn:
        # Technical indicators feature table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feat_technical_indicators (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                close FLOAT,
                sma_20 FLOAT,
                sma_50 FLOAT,
                sma_200 FLOAT,
                rsi_14 FLOAT,
                macd FLOAT,
                macd_signal FLOAT,
                macd_diff FLOAT,
                atr_14 FLOAT,
                momentum_10 FLOAT,
                momentum_20 FLOAT,
                volume_sma_20 FLOAT,
                volatility_20 FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date),
                FOREIGN KEY(ticker) REFERENCES dim_company(ticker)
            );
        """))
        conn.commit()

        # Macro aggregation features table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feat_macro_ratios (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                interest_rate FLOAT,
                interest_rate_momentum_1m FLOAT,
                interest_rate_momentum_3m FLOAT,
                inflation FLOAT,
                inflation_momentum_1m FLOAT,
                inflation_momentum_3m FLOAT,
                interest_rate_volatility_20d FLOAT,
                inflation_volatility_20d FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date),
                FOREIGN KEY(ticker) REFERENCES dim_company(ticker)
            );
        """))
        conn.commit()

        # Company fundamentals features table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feat_company_fundamentals (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                revenue_growth_yoy FLOAT,
                eps_growth_yoy FLOAT,
                net_income_growth_yoy FLOAT,
                revenue_momentum_ttm FLOAT,
                earnings_momentum_ttm FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date),
                FOREIGN KEY(ticker) REFERENCES dim_company(ticker)
            );
        """))
        conn.commit()

        # Training labels table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feat_labels (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                date DATE NOT NULL,
                forward_return_30d FLOAT,
                label VARCHAR(10) DEFAULT 'HOLD',
                is_valid BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date),
                FOREIGN KEY(ticker) REFERENCES dim_company(ticker)
            );
        """))
        conn.commit()

        # Predictions table for historical tracking
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fact_predictions (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) NOT NULL,
                prediction_date DATE NOT NULL,
                prediction_horizon_days INT DEFAULT 30,
                predicted_label VARCHAR(10),
                predicted_probs_buy FLOAT,
                predicted_probs_hold FLOAT,
                predicted_probs_sell FLOAT,
                confidence FLOAT,
                actual_label VARCHAR(10),
                actual_return FLOAT,
                is_correct BOOLEAN,
                model_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, prediction_date, model_version),
                FOREIGN KEY(ticker) REFERENCES dim_company(ticker)
            );
        """))
        conn.commit()

        print("✓ Feature store schema created successfully")


if __name__ == "__main__":
    from db import get_engine
    engine = get_engine()
    create_feature_store_schema(engine)
