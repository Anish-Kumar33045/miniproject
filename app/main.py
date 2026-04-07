import streamlit as st

st.set_page_config(
    page_title="BehaviorGuard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

USERS = {"admin": "guard123", "demo": "demo123"}

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
        with st.container():
            st.markdown("""
            <div style="background:var(--color-background-secondary);
                        border:0.5px solid var(--color-border-tertiary);
                        border-radius:16px; padding:2rem;">
            """, unsafe_allow_html=True)
            st.markdown("#### Sign in")
            username = st.text_input("Username", placeholder="admin or demo")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            if st.button("Sign in →", use_container_width=True):
                if username in USERS and USERS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center; margin-top:2rem;">
            <p style="font-size:13px; color:var(--color-text-secondary);">
                Demo credentials: <code>demo / demo123</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
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
    st.session_state.logged_in = False
    st.session_state.username  = ""
    st.rerun()

if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.markdown(f"👤 **{st.session_state.get('username','user')}**")
    if st.sidebar.button("Logout"):
        logout()
    st.sidebar.markdown("---")
    st.sidebar.page_link("pages/1_Dashboard.py",     label="Dashboard",          icon="🏠")
    st.sidebar.page_link("pages/2_Timeline.py",      label="Transaction timeline",icon="📅")
    st.sidebar.page_link("pages/3_Analytics.py",     label="Spending analytics",  icon="📊")
    st.sidebar.page_link("pages/4_Model_Metrics.py", label="Model metrics",       icon="🧪")

    st.title("🔐 BehaviorGuard")
    st.caption("Welcome back! Use the sidebar to navigate.")
    st.info("Go to **Dashboard** to upload transactions and detect fraud.")