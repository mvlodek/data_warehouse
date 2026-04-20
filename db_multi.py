"""
Multi-database support: PostgreSQL, MongoDB, and MySQL
Unified interface for the data warehouse.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, Any
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool


# =============================================================================
# Abstract Base Database Interface
# =============================================================================

class DatabaseInterface(ABC):
    """Abstract interface for all database backends."""
    
    @abstractmethod
    def connect(self) -> Any:
        """Establish database connection."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        pass
    
    @abstractmethod
    def execute_write(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute a write operation (INSERT/UPDATE/DELETE)."""
        pass
    
    @abstractmethod
    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> int:
        """Load a DataFrame into a table."""
        pass
    
    @abstractmethod
    def get_tables(self) -> list:
        """List all tables in the database."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass


# =============================================================================
# PostgreSQL Implementation
# =============================================================================

class PostgreSQLDB(DatabaseInterface):
    """PostgreSQL database implementation."""
    
    def __init__(self, 
                 host: str = None,
                 port: int = 5432,
                 database: str = None,
                 user: str = None,
                 password: str = None):
        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port
        self.database = database or os.getenv('POSTGRES_DB', 'financial_data_warehouse')
        self.user = user or os.getenv('POSTGRES_USER', 'postgres')
        self.password = password or os.getenv('POSTGRES_PASSWORD', '87654321')
        self._engine = None
    
    def connect(self):
        connection_string = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self._engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10
        )
        return self._engine
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        if not self._engine:
            self.connect()
        return pd.read_sql(text(query), self._engine, params=params)
    
    def execute_write(self, query: str, params: Optional[tuple] = None) -> int:
        if not self._engine:
            self.connect()
        with self._engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result.rowcount
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> int:
        if not self._engine:
            self.connect()
        df.to_sql(table_name, self._engine, if_exists=if_exists, index=False)
        return len(df)
    
    def get_tables(self) -> list:
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        return self.execute_query(query)['table_name'].tolist()
    
    def close(self) -> None:
        if self._engine:
            self._engine.dispose()


# =============================================================================
# MySQL Implementation
# =============================================================================

class MySQLDB(DatabaseInterface):
    """MySQL database implementation."""
    
    def __init__(self,
                 host: str = None,
                 port: int = 3306,
                 database: str = None,
                 user: str = None,
                 password: str = None):
        self.host = host or os.getenv('MYSQL_HOST', 'localhost')
        self.port = port
        self.database = database or os.getenv('MYSQL_DB', 'financial_data_warehouse')
        self.user = user or os.getenv('MYSQL_USER', 'root')
        self.password = password or os.getenv('MYSQL_PASSWORD', '')
        self._engine = None
    
    def connect(self):
        connection_string = f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        self._engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        return self._engine
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        if not self._engine:
            self.connect()
        return pd.read_sql(text(query), self._engine, params=params)
    
    def execute_write(self, query: str, params: Optional[tuple] = None) -> int:
        if not self._engine:
            self.connect()
        with self._engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            return result.rowcount
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append') -> int:
        if not self._engine:
            self.connect()
        # MySQL requires lowercase table names and handles datetime differently
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['datetime64']).columns:
            df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_copy.to_sql(table_name.lower(), self._engine, if_exists=if_exists, index=False)
        return len(df)
    
    def get_tables(self) -> list:
        query = "SHOW TABLES"
        result = self.execute_query(query)
        return result.iloc[:, 0].tolist()
    
    def close(self) -> None:
        if self._engine:
            self._engine.dispose()


# =============================================================================
# MongoDB Implementation
# =============================================================================

class MongoDB:
    """MongoDB database implementation for document storage."""
    
    def __init__(self,
                 host: str = None,
                 port: int = 27017,
                 database: str = None,
                 user: str = None,
                 password: str = None):
        self.host = host or os.getenv('MONGO_HOST', 'localhost')
        self.port = port
        self.database = database or os.getenv('MONGO_DB', 'financial_data_warehouse')
        self.user = user or os.getenv('MONGO_USER', '')
        self.password = password or os.getenv('MONGO_PASSWORD', '')
        self._client = None
        self._db = None
    
    def connect(self):
        """Connect to MongoDB."""
        try:
            from pymongo import MongoClient
            if self.user and self.password:
                connection_string = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}"
            else:
                connection_string = f"mongodb://{self.host}:{self.port}"
            self._client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            self._db = self._client[self.database]
            # Test connection
            self._client.server_info()
            return self._db
        except ImportError:
            raise ImportError("pymongo is required. Install with: pip install pymongo")
    
    def get_collection(self, collection_name: str):
        """Get a collection by name."""
        if not self._db:
            self.connect()
        return self._db[collection_name]
    
    def insert_documents(self, collection_name: str, documents: list) -> int:
        """Insert multiple documents into a collection."""
        if not self._db:
            self.connect()
        collection = self._db[collection_name]
        if isinstance(documents, list):
            result = collection.insert_many(documents)
            return len(result.inserted_ids)
        else:
            result = collection.insert_one(documents)
            return 1
    
    def find(self, collection_name: str, query: dict = None, projection: dict = None, limit: int = 0) -> list:
        """Find documents in a collection."""
        if not self._db:
            self.connect()
        collection = self._db[collection_name]
        cursor = collection.find(query or {}, projection)
        if limit > 0:
            cursor = cursor.limit(limit)
        return list(cursor)
    
    def find_dataframe(self, collection_name: str, query: dict = None, projection: dict = None, limit: int = 0) -> pd.DataFrame:
        """Find documents and return as DataFrame."""
        documents = self.find(collection_name, query, projection, limit)
        if documents:
            return pd.DataFrame(documents)
        return pd.DataFrame()
    
    def update_documents(self, collection_name: str, filter_query: dict, update: dict, many: bool = True) -> int:
        """Update documents in a collection."""
        if not self._db:
            self.connect()
        collection = self._db[collection_name]
        if many:
            result = collection.update_many(filter_query, update)
            return result.modified_count
        else:
            result = collection.update_one(filter_query, update)
            return result.modified_count
    
    def delete_documents(self, collection_name: str, filter_query: dict, many: bool = True) -> int:
        """Delete documents from a collection."""
        if not self._db:
            self.connect()
        collection = self._db[collection_name]
        if many:
            result = collection.delete_many(filter_query)
            return result.deleted_count
        else:
            result = collection.delete_one(filter_query)
            return result.deleted_count
    
    def aggregate(self, collection_name: str, pipeline: list) -> list:
        """Run aggregation pipeline on a collection."""
        if not self._db:
            self.connect()
        collection = self._db[collection_name]
        return list(collection.aggregate(pipeline))
    
    def get_collections(self) -> list:
        """List all collections in the database."""
        if not self._db:
            self.connect()
        return self._db.list_collection_names()
    
    def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            self._client.close()


# =============================================================================
# Database Factory
# =============================================================================

class DatabaseFactory:
    """Factory for creating database connections."""
    
    _instances = {}
    
    @classmethod
    def get_database(cls, db_type: str, **kwargs) -> DatabaseInterface:
        """
        Get a database instance by type.
        
        Args:
            db_type: 'postgresql', 'mysql', or 'mongodb'
            **kwargs: Additional connection parameters
        
        Returns:
            Database interface instance
        """
        db_type = db_type.lower()
        
        if db_type == 'postgresql':
            if 'postgresql' not in cls._instances:
                cls._instances['postgresql'] = PostgreSQLDB(**kwargs)
            return cls._instances['postgresql']
        
        elif db_type == 'mysql':
            if 'mysql' not in cls._instances:
                cls._instances['mysql'] = MySQLDB(**kwargs)
            return cls._instances['mysql']
        
        elif db_type == 'mongodb':
            if 'mongodb' not in cls._instances:
                cls._instances['mongodb'] = MongoDB(**kwargs)
            return cls._instances['mongodb']
        
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    @classmethod
    def get_all_databases(cls) -> dict:
        """Get all configured database instances."""
        return cls._instances.copy()
    
    @classmethod
    def close_all(cls) -> None:
        """Close all database connections."""
        for db in cls._instances.values():
            db.close()
        cls._instances.clear()


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================

def get_postgres_engine():
    """Get PostgreSQL engine (backward compatible)."""
    return PostgreSQLDB().connect()


def get_mysql_engine():
    """Get MySQL engine."""
    return MySQLDB().connect()


def get_mongo_client():
    """Get MongoDB client."""
    return MongoDB().connect()


# =============================================================================
# Environment Variables Setup
# =============================================================================

def load_db_env():
    """Load database configuration from environment variables."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # PostgreSQL
    os.getenv('POSTGRES_HOST', 'localhost')
    os.getenv('POSTGRES_DB', 'financial_data_warehouse')
    os.getenv('POSTGRES_USER', 'postgres')
    os.getenv('POSTGRES_PASSWORD', '87654321')
    
    # MySQL
    os.getenv('MYSQL_HOST', 'localhost')
    os.getenv('MYSQL_DB', 'financial_data_warehouse')
    os.getenv('MYSQL_USER', 'root')
    os.getenv('MYSQL_PASSWORD', '')
    
    # MongoDB
    os.getenv('MONGO_HOST', 'localhost')
    os.getenv('MONGO_DB', 'financial_data_warehouse')
    os.getenv('MONGO_USER', '')
    os.getenv('MONGO_PASSWORD', '')


if __name__ == "__main__":
    # Test connections
    load_db_env()
    
    print("Testing PostgreSQL...")
    try:
        pg = DatabaseFactory.get_database('postgresql')
        pg.connect()
        print(f"  ✓ Connected. Tables: {pg.get_tables()}")
        pg.close()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\nTesting MySQL...")
    try:
        mysql = DatabaseFactory.get_database('mysql')
        mysql.connect()
        print(f"  ✓ Connected. Tables: {mysql.get_tables()}")
        mysql.close()
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    
    print("\nTesting MongoDB...")
    try:
        mongo = DatabaseFactory.get_database('mongodb')
        mongo.connect()
        print(f"  ✓ Connected. Collections: {mongo.get_collections()}")
        mongo.close()
    except Exception as e:
        print(f"  ✗ Failed: {e}")