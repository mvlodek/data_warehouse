from sqlalchemy import create_engine

def get_engine():
    engine = create_engine(
        "postgresql+psycopg2://postgres:87654321@localhost:5432/financial_data_warehouse"
    )
    return engine