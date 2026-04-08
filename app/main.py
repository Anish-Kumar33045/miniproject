import streamlit as st
from database import (
    SessionLocal, get_user_by_username, verify_password,
    log_session, init_db
)
from datetime import datetime

st.set_page_config(
    page_title="BehaviorGuard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    init_db()
except Exception:
    pass

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# USERS = {"admin": "guard123", "demo": "demo123"}

def login_page():
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0 1rem;">
        <div style="font-size:52px;">🔐</div>
        <h1 style="font-size:2.2rem; font-weight:600; margin:0.5rem 0;">BehaviorGuard</h1>
        <p style="color:var(--color-text-secondary); font-size:1rem;">
            AI-powered UPI fraud detection platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div style="background:var(--color-background-secondary);
                    border:0.5px solid var(--color-border-tertiary);
                    border-radius:16px; padding:2rem;">
        """, unsafe_allow_html=True)
        st.markdown("#### Sign in")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Sign in →", use_container_width=True):
                db = SessionLocal()
                user = get_user_by_username(db, username)
                if user and verify_password(password, user.password_hash) and user.is_active:
                    user.last_login = datetime.now()
                    db.commit()
                    log_session(db, user.user_id)
                    st.session_state.logged_in = True
                    st.session_state.username  = user.username
                    st.session_state.user_id   = str(user.user_id)
                    st.session_state.full_name = user.full_name
                    db.close()
                    st.rerun()
                else:
                    db.close()
                    st.error("Invalid credentials or account inactive")

        with tab2:
            new_user  = st.text_input("Username",   key="reg_user")
            new_email = st.text_input("Email",       key="reg_email")
            new_name  = st.text_input("Full name",   key="reg_name")
            new_phone = st.text_input("Phone",       key="reg_phone")
            new_pass  = st.text_input("Password",    type="password", key="reg_pass")
            new_pass2 = st.text_input("Confirm password", type="password", key="reg_pass2")
            if st.button("Create account →", use_container_width=True):
                if new_pass != new_pass2:
                    st.error("Passwords do not match")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    from database import create_user
                    db = SessionLocal()
                    try:
                        create_user(db, new_user, new_email, new_pass, new_name, new_phone)
                        st.success("Account created! Please login.")
                    except Exception as e:
                        st.error(f"Username or email already exists.")
                    finally:
                        db.close()
        st.markdown("</div>", unsafe_allow_html=True)

    # feature cards — keep same as before
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    for col, icon, title, desc in [
        (c1,"🤖","Dual ML engine","Random Forest + Isolation Forest"),
        (c2,"🧠","SHAP explainability","Know why each txn is flagged"),
        (c3,"📊","Rich analytics","Heatmaps, trends, merchant risk"),
        (c4,"⚡","Real-time detection","Upload CSV, get instant results"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:var(--color-background-secondary);
                        border:0.5px solid var(--color-border-tertiary);
                        border-radius:12px; padding:1.2rem; text-align:center;">
                <div style="font-size:28px;">{icon}</div>
                <p style="font-weight:500; margin:0.5rem 0 0.25rem;">{title}</p>
                <p style="font-size:12px; color:var(--color-text-secondary); margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def logout():
    db = SessionLocal()
    db.query(__import__('database').UserSession).filter_by(
        user_id=st.session_state.user_id, is_active=True
    ).update({"is_active": False, "logout_at": datetime.now()})
    db.commit()
    db.close()
    for key in ['logged_in','user_id','username','full_name','results','X_raw','df_input']:
        st.session_state.pop(key, None)
    st.rerun()

if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.markdown(f"👤 **{st.session_state.get('full_name') or st.session_state.get('username')}**")
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.markdown("---")
    st.sidebar.page_link("pages/1_Dashboard.py",     label="Dashboard",           icon="🏠")
    st.sidebar.page_link("pages/2_Timeline.py",      label="Transaction timeline", icon="📅")
    st.sidebar.page_link("pages/3_Analytics.py",     label="Spending analytics",   icon="📊")
    st.sidebar.page_link("pages/4_Model_Metrics.py", label="Model metrics",        icon="🧪")
    st.title("🔐 BehaviorGuard")
    st.caption("Welcome back! Use the sidebar to navigate.")
    st.info("Go to **Dashboard** to upload transactions and detect fraud.")