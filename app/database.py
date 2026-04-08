import os
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Boolean, DateTime, Text, ForeignKey, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
from dotenv import load_dotenv
import uuid, bcrypt

load_dotenv()

DB_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

engine = create_engine(DB_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id      = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username     = Column(String(50),  unique=True, nullable=False)
    email        = Column(String(120), unique=True, nullable=False)
    password_hash= Column(String(255), nullable=False)
    full_name    = Column(String(100))
    phone        = Column(String(20))
    created_at   = Column(DateTime, default=func.now())
    last_login   = Column(DateTime)
    is_active    = Column(Boolean, default=True)

    transactions = relationship("Transaction", back_populates="user",
                                cascade="all, delete-orphan")
    sessions     = relationship("UserSession", back_populates="user",
                                cascade="all, delete-orphan")
    fraud_results= relationship("FraudResult", back_populates="user",
                                cascade="all, delete-orphan")


class UserSession(Base):
    __tablename__ = "user_sessions"

    session_id  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    login_at    = Column(DateTime, default=func.now())
    logout_at   = Column(DateTime)
    ip_address  = Column(String(50))
    device_info = Column(String(200))
    is_active   = Column(Boolean, default=True)

    user = relationship("User", back_populates="sessions")


class Transaction(Base):
    __tablename__ = "transactions"

    txn_id               = Column(String(20), primary_key=True)
    user_id              = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    amount               = Column(Float)
    merchant_cat         = Column(String(50))
    merchant_risk_score  = Column(Float)
    hour                 = Column(Integer)
    day_of_week          = Column(Integer)
    is_new_merchant      = Column(Integer)
    txn_per_day          = Column(Integer)
    avg_amount_7d        = Column(Float)
    device_change        = Column(Integer)
    location_change      = Column(Integer)
    failed_txn_count     = Column(Integer)
    city                 = Column(String(50))
    bank                 = Column(String(50))
    is_weekend           = Column(Integer)
    amount_to_avg_ratio  = Column(Float)
    txn_date             = Column(String(20))
    uploaded_at          = Column(DateTime, default=func.now())

    user         = relationship("User", back_populates="transactions")
    fraud_result = relationship("FraudResult", back_populates="transaction",
                                uselist=False, cascade="all, delete-orphan")


class FraudResult(Base):
    __tablename__ = "fraud_results"

    result_id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    txn_id             = Column(String(20), ForeignKey("transactions.txn_id"), nullable=False)
    user_id            = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    fraud_probability  = Column(Float)
    is_fraud_predicted = Column(Integer)
    rf_flag            = Column(Integer)
    iso_flag           = Column(Integer)
    shap_reason        = Column(Text)
    alert_text         = Column(Text)
    analysed_at        = Column(DateTime, default=func.now())
    user_feedback      = Column(String(20))

    user        = relationship("User", back_populates="fraud_results")
    transaction = relationship("Transaction", back_populates="fraud_result")


def init_db():
    Base.metadata.create_all(bind=engine)
    print("All tables created.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_user(db, username, email, password, full_name="", phone=""):
    user = User(
        username=username,
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        phone=phone
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_username(db, username: str):
    return db.query(User).filter(User.username == username).first()


def log_session(db, user_id, ip=None, device=None):
    from datetime import datetime
    db.query(UserSession).filter(
        UserSession.user_id == user_id,
        UserSession.is_active == True
    ).update({"is_active": False, "logout_at": datetime.now()})
    session = UserSession(user_id=user_id, ip_address=ip, device_info=device)
    db.add(session)
    db.commit()
    return session


def save_transactions(db, df, user_id):
    from datetime import datetime
    saved = 0
    for _, row in df.iterrows():
        existing = db.query(Transaction).filter(
            Transaction.txn_id == row['txn_id'],
            Transaction.user_id == user_id
        ).first()
        if existing:
            continue
        txn = Transaction(
            txn_id              = row['txn_id'],
            user_id             = user_id,
            amount              = float(row.get('amount', 0)),
            merchant_cat        = str(row.get('merchant_cat', '')),
            merchant_risk_score = float(row.get('merchant_risk_score', 0)),
            hour                = int(row.get('hour', 0)),
            day_of_week         = int(row.get('day_of_week', 0)),
            is_new_merchant     = int(row.get('is_new_merchant', 0)),
            txn_per_day         = int(row.get('txn_per_day', 1)),
            avg_amount_7d       = float(row.get('avg_amount_7d', 0)),
            device_change       = int(row.get('device_change', 0)),
            location_change     = int(row.get('location_change', 0)),
            failed_txn_count    = int(row.get('failed_txn_count', 0)),
            city                = str(row.get('city', '')),
            bank                = str(row.get('bank', '')),
            is_weekend          = int(row.get('is_weekend', 0)),
            amount_to_avg_ratio = float(row.get('amount_to_avg_ratio', 1)),
            txn_date            = str(row.get('txn_date', '')),
            uploaded_at         = datetime.now()
        )
        db.add(txn)
        saved += 1
    db.commit()
    return saved


def save_fraud_results(db, results_df, user_id):
    for _, row in results_df.iterrows():
        existing = db.query(FraudResult).filter(
            FraudResult.txn_id  == row['txn_id'],
            FraudResult.user_id == user_id
        ).first()
        if existing:
            existing.fraud_probability  = float(row.get('fraud_probability', 0))
            existing.is_fraud_predicted = int(row.get('is_fraud_predicted', 0))
            existing.rf_flag            = int(row.get('rf_flag', 0))
            existing.iso_flag           = int(row.get('iso_flag', 0))
        else:
            fr = FraudResult(
                txn_id             = row['txn_id'],
                user_id            = user_id,
                fraud_probability  = float(row.get('fraud_probability', 0)),
                is_fraud_predicted = int(row.get('is_fraud_predicted', 0)),
                rf_flag            = int(row.get('rf_flag', 0)),
                iso_flag           = int(row.get('iso_flag', 0)),
            )
            db.add(fr)
    db.commit()


def get_user_transactions(db, user_id):
    import pandas as pd
    txns = db.query(Transaction).filter(Transaction.user_id == user_id).all()
    if not txns:
        return pd.DataFrame()
    return pd.DataFrame([{
        'txn_id': t.txn_id, 'amount': t.amount, 'merchant_cat': t.merchant_cat,
        'merchant_risk_score': t.merchant_risk_score, 'hour': t.hour,
        'day_of_week': t.day_of_week, 'is_new_merchant': t.is_new_merchant,
        'txn_per_day': t.txn_per_day, 'avg_amount_7d': t.avg_amount_7d,
        'device_change': t.device_change, 'location_change': t.location_change,
        'failed_txn_count': t.failed_txn_count, 'city': t.city, 'bank': t.bank,
        'is_weekend': t.is_weekend, 'amount_to_avg_ratio': t.amount_to_avg_ratio,
        'txn_date': t.txn_date
    } for t in txns])


def get_user_fraud_results(db, user_id):
    import pandas as pd
    rows = db.query(FraudResult).filter(FraudResult.user_id == user_id).all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        'txn_id': r.txn_id, 'fraud_probability': r.fraud_probability,
        'is_fraud_predicted': r.is_fraud_predicted, 'rf_flag': r.rf_flag,
        'iso_flag': r.iso_flag, 'alert_text': r.alert_text,
        'user_feedback': r.user_feedback, 'analysed_at': r.analysed_at
    } for r in rows])