"""
Database module for AQI Monitoring Application
"""
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Get database connection from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL')

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define database models
class City(Base):
    """City model for storing city information"""
    __tablename__ = "cities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    state = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    aqi_readings = relationship("AQIReading", back_populates="city")
    pollutant_readings = relationship("PollutantReading", back_populates="city")

class AQIReading(Base):
    """AQI Reading model for storing AQI values"""
    __tablename__ = "aqi_readings"

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    value = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    city = relationship("City", back_populates="aqi_readings")

class PollutantReading(Base):
    """Pollutant Reading model for storing pollutant values"""
    __tablename__ = "pollutant_readings"

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    pm25 = Column(Float)
    pm10 = Column(Float)
    no2 = Column(Float)
    so2 = Column(Float)
    co = Column(Float)
    o3 = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    city = relationship("City", back_populates="pollutant_readings")

class PredictionModel(Base):
    """Prediction Model for storing trained model information"""
    __tablename__ = "prediction_models"

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    model_type = Column(String)  # e.g., "RandomForest", "LinearRegression"
    accuracy = Column(Float)
    features = Column(String)  # JSON string of features used
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

class AQIPrediction(Base):
    """AQI Prediction model for storing predicted AQI values"""
    __tablename__ = "aqi_predictions"

    id = Column(Integer, primary_key=True, index=True)
    city_id = Column(Integer, ForeignKey("cities.id"))
    value = Column(Float)
    prediction_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.now)

# Function to get a database session
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables
def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

# Initialize database with city data
def initialize_cities():
    """Initialize cities in the database"""
    from constants import MAJOR_CITIES
    db = SessionLocal()
    
    try:
        # First, check if we need to update by counting cities
        existing_count = db.query(City).count()
        if existing_count >= len(MAJOR_CITIES):
            print(f"Database already has {existing_count} cities, no update needed")
            db.close()
            return
            
        # We need to update cities - first clear any existing related data
        print(f"Updating cities database with {len(MAJOR_CITIES)} cities")
        db.query(AQIPrediction).delete()
        db.query(PollutantReading).delete()
        db.query(AQIReading).delete()
        db.query(PredictionModel).delete()
        db.query(City).delete()
        db.commit()
        
        # Now add the new cities from constants
        for city_name, data in MAJOR_CITIES.items():
            city = City(
                name=city_name,
                latitude=data["lat"],
                longitude=data["lon"],
                state=data["state"]
            )
            db.add(city)
        
        db.commit()
        print(f"Successfully initialized database with {len(MAJOR_CITIES)} cities")
    except Exception as e:
        print(f"Error initializing cities: {str(e)}")
        db.rollback()
    finally:
        db.close()

# Initialize the database
def init_db():
    """Initialize the database"""
    create_tables()
    initialize_cities()