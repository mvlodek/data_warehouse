"""
Database initialization script for the data warehouse.
Sets up PostgreSQL, MySQL, and MongoDB with proper schemas.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_multi import DatabaseFactory, load_db_env, PostgreSQLDB, MySQLDB, MongoDB


# =============================================================================
# PostgreSQL Schema
# =============================================================================

POSTGRES_SCHEMA = """
-- Dimension Tables
CREATE TABLE IF NOT EXISTS dim_company (
    ticker VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_date (
    date DATE PRIMARY KEY,
    year INTEGER,
    quarter INTEGER,
    month INTEGER,
    day_of_week INTEGER,
    is_weekend BOOLEAN
);

-- Fact Tables
CREATE TABLE IF NOT EXISTS fact_stock_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS fact_predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    prediction_date DATE,
    predicted_label VARCHAR(10),
    confidence DECIMAL(5,4),
    predicted_probs_buy DECIMAL(5,4),
    predicted_probs_hold DECIMAL(5,4),
    predicted_probs_sell DECIMAL(5,4),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature Tables
CREATE TABLE IF NOT EXISTS feat_technical_indicators (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    close DECIMAL(10,2),
    sma_20 DECIMAL(10,2),
    sma_50 DECIMAL(10,2),
    sma_200 DECIMAL(10,2),
    rsi_14 DECIMAL(5,2),
    macd DECIMAL(10,4),
    macd_signal DECIMAL(10,4),
    macd_diff DECIMAL(10,4),
    atr_14 DECIMAL(10,4),
    momentum_10 DECIMAL(10,4),
    momentum_20 DECIMAL(10,4),
    volume_sma_20 BIGINT,
    volatility_20 DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS feat_macro_ratios (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    interest_rate DECIMAL(5,2),
    interest_rate_momentum_1m DECIMAL(5,4),
    interest_rate_momentum_3m DECIMAL(5,4),
    inflation DECIMAL(5,2),
    inflation_momentum_1m DECIMAL(5,4),
    inflation_momentum_3m DECIMAL(5,4),
    interest_rate_volatility_20d DECIMAL(5,4),
    inflation_volatility_20d DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS feat_company_fundamentals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    revenue_growth_yoy DECIMAL(10,4),
    eps_growth_yoy DECIMAL(10,4),
    net_income_growth_yoy DECIMAL(10,4),
    revenue_momentum_ttm DECIMAL(10,4),
    earnings_momentum_ttm DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS feat_labels (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    label VARCHAR(10),
    forward_return_30d DECIMAL(10,4),
    is_valid BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE TABLE IF NOT EXISTS feat_fmp_metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    pe_ratio DECIMAL(10,2),
    eps DECIMAL(10,4),
    market_cap BIGINT,
    dividend_yield DECIMAL(5,4),
    beta DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_fact_stock_ticker_date ON fact_stock_prices(ticker, date);
CREATE INDEX IF NOT EXISTS idx_fact_predictions_ticker_date ON fact_predictions(ticker, prediction_date);
CREATE INDEX IF NOT EXISTS idx_feat_technical_ticker_date ON feat_technical_indicators(ticker, date);
CREATE INDEX IF NOT EXISTS idx_feat_macro_ticker_date ON feat_macro_ratios(ticker, date);
CREATE INDEX IF NOT EXISTS idx_feat_labels_ticker_date ON feat_labels(ticker, date);
"""


# =============================================================================
# MySQL Schema
# =============================================================================

MYSQL_SCHEMA = """
-- Dimension Tables
CREATE TABLE IF NOT EXISTS dim_company (
    ticker VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fact Tables
CREATE TABLE IF NOT EXISTS fact_stock_prices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_ticker_date (ticker, date)
);

CREATE TABLE IF NOT EXISTS fact_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10),
    prediction_date DATE,
    predicted_label VARCHAR(10),
    confidence DECIMAL(5,4),
    predicted_probs_buy DECIMAL(5,4),
    predicted_probs_hold DECIMAL(5,4),
    predicted_probs_sell DECIMAL(5,4),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature Tables
CREATE TABLE IF NOT EXISTS feat_technical_indicators (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    close DECIMAL(10,2),
    sma_20 DECIMAL(10,2),
    sma_50 DECIMAL(10,2),
    sma_200 DECIMAL(10,2),
    rsi_14 DECIMAL(5,2),
    macd DECIMAL(10,4),
    macd_signal DECIMAL(10,4),
    macd_diff DECIMAL(10,4),
    atr_14 DECIMAL(10,4),
    momentum_10 DECIMAL(10,4),
    momentum_20 DECIMAL(10,4),
    volume_sma_20 BIGINT,
    volatility_20 DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_tech_ticker_date (ticker, date)
);

CREATE TABLE IF NOT EXISTS feat_macro_ratios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    interest_rate DECIMAL(5,2),
    interest_rate_momentum_1m DECIMAL(5,4),
    interest_rate_momentum_3m DECIMAL(5,4),
    inflation DECIMAL(5,2),
    inflation_momentum_1m DECIMAL(5,4),
    inflation_momentum_3m DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_macro_ticker_date (ticker, date)
);

CREATE TABLE IF NOT EXISTS feat_labels (
    id INT AUTO_INCREMENT PRIMARY KEY,
    ticker VARCHAR(10),
    date DATE,
    label VARCHAR(10),
    forward_return_30d DECIMAL(10,4),
    is_valid TINYINT(1) DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_labels_ticker_date (ticker, date)
);
"""


# =============================================================================
# MongoDB Collections
# =============================================================================

MONGO_COLLECTIONS = {
    'companies': {
        'ticker': 'string',
        'name': 'string',
        'sector': 'string',
        'industry': 'string',
        'market_cap': 'number'
    },
    'stock_prices': {
        'ticker': 'string',
        'date': 'date',
        'open': 'number',
        'high': 'number',
        'low': 'number',
        'close': 'number',
        'volume': 'number'
    },
    'predictions': {
        'ticker': 'string',
        'prediction_date': 'date',
        'predicted_label': 'string',
        'confidence': 'number',
        'model_version': 'string'
    },
    'technical_indicators': {
        'ticker': 'string',
        'date': 'date',
        'close': 'number',
        'sma_20': 'number',
        'sma_50': 'number',
        'rsi_14': 'number',
        'macd': 'number'
    },
    'macro_data': {
        'indicator': 'string',
        'date': 'date',
        'value': 'number'
    }
}


# =============================================================================
# Initialization Functions
# =============================================================================

def init_postgres():
    """Initialize PostgreSQL database."""
    print("\n" + "="*50)
    print("Initializing PostgreSQL...")
    print("="*50)
    
    try:
        db = PostgreSQLDB()
        db.connect()
        
        # Execute schema
        for statement in POSTGRES_SCHEMA.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    db.execute_write(statement)
                except Exception as e:
                    # Table might already exist
                    if 'already exists' not in str(e).lower():
                        print(f"  Note: {e}")
        
        print("✓ PostgreSQL initialized successfully")
        print(f"  Tables: {db.get_tables()}")
        db.close()
        return True
    except Exception as e:
        print(f"✗ PostgreSQL initialization failed: {e}")
        return False


def init_mysql():
    """Initialize MySQL database."""
    print("\n" + "="*50)
    print("Initializing MySQL...")
    print("="*50)
    
    try:
        db = MySQLDB()
        db.connect()
        
        # Execute schema
        for statement in MYSQL_SCHEMA.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    db.execute_write(statement)
                except Exception as e:
                    if 'already exists' not in str(e).lower():
                        print(f"  Note: {e}")
        
        print("✓ MySQL initialized successfully")
        print(f"  Tables: {db.get_tables()}")
        db.close()
        return True
    except Exception as e:
        print(f"✗ MySQL initialization failed: {e}")
        return False


def init_mongodb():
    """Initialize MongoDB collections."""
    print("\n" + "="*50)
    print("Initializing MongoDB...")
    print("="*50)
    
    try:
        mongo = MongoDB()
        mongo.connect()
        
        # Create collections with sample documents
        for collection_name, schema in MONGO_COLLECTIONS.items():
            # Insert a sample document to create collection
            sample_doc = {
                '_init': True,
                'created_at': datetime.now().isoformat()
            }
            try:
                mongo.insert_documents(collection_name, [sample_doc])
                # Remove the sample
                mongo.delete_documents(collection_name, {'_init': True})
            except:
                pass
        
        print("✓ MongoDB initialized successfully")
        print(f"  Collections: {mongo.get_collections()}")
        mongo.close()
        return True
    except Exception as e:
        print(f"✗ MongoDB initialization failed: {e}")
        return False


def seed_sample_data(db_type: str = 'postgresql'):
    """Seed sample data for testing."""
    print("\n" + "="*50)
    print(f"Seeding sample data for {db_type}...")
    print("="*50)
    
    load_db_env()
    db = DatabaseFactory.get_database(db_type)
    db.connect()
    
    # Sample companies
    companies = [
        {'ticker': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'market_cap': 3000000000000},
        {'ticker': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'Technology', 'industry': 'Software', 'market_cap': 2800000000000},
        {'ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'sector': 'Technology', 'industry': 'Internet', 'market_cap': 1800000000000},
        {'ticker': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'Consumer Cyclical', 'industry': 'E-commerce', 'market_cap': 1600000000000},
        {'ticker': 'META', 'name': 'Meta Platforms Inc.', 'sector': 'Technology', 'industry': 'Social Media', 'market_cap': 1200000000000},
    ]
    
    try:
        for company in companies:
            try:
                db.load_dataframe(pd.DataFrame([company]), 'dim_company', 'append')
            except:
                pass
        print(f"✓ Seeded {len(companies)} companies")
    except Exception as e:
        print(f"  Note: {e}")
    
    db.close()


def main():
    """Main initialization function."""
    print("\n" + "="*60)
    print("DATABASE INITIALIZATION")
    print("="*60)
    
    load_db_env()
    
    results = {}
    
    # Initialize each database
    results['postgresql'] = init_postgres()
    results['mysql'] = init_mysql()
    results['mongodb'] = init_mongodb()
    
    # Summary
    print("\n" + "="*60)
    print("INITIALIZATION SUMMARY")
    print("="*60)
    
    for db_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {db_type.upper()}: {status}")
    
    # Seed sample data if any database is available
    if any(results.values()):
        available_db = next((k for k, v in results.items() if v), None)
        if available_db:
            seed_sample_data(available_db)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Update .env with your database credentials")
    print("2. Run: streamlit run app_v2.py")
    print("3. Navigate to the Data Warehouse tab to explore data")


if __name__ == "__main__":
    main()