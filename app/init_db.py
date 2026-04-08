import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from database import init_db, create_user, SessionLocal

if __name__ == '__main__':
    init_db()
    db = SessionLocal()
    try:
        create_user(db, "admin", "admin@bg.com",  "guard123", "Admin User")
        create_user(db, "demo",  "demo@bg.com",   "demo123",  "Demo User")
        print("Default users created: admin / guard123  and  demo / demo123")
    except Exception as e:
        print(f"Users may already exist: {e}")
    finally:
        db.close()