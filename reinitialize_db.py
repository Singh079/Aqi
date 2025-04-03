"""
Script to reinitialize database with new city and state information
"""
from database import init_db, Base, engine, City, AQIReading, PollutantReading
import db_operations as db_ops

print("Reinitializing database...")

# Drop all tables
print("Dropping existing tables...")
Base.metadata.drop_all(bind=engine)

# Create tables and initialize with new city and state data
print("Creating tables and initializing cities...")
init_db()

# Seed with test data
print("Seeding database with test data...")
db_ops.seed_database_with_test_data()

print("Database reinitialization complete!")