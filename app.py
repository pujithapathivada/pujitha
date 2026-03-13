import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title='House Price Prediction',
    page_icon='🏠',
    layout='centered'
)

# custom css with animations
st.markdown("""
<style>
    /* floating houses animation */
    .floating-houses {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        overflow: hidden;
        z-index: -1;
    }
    .house {
        position: absolute;
        font-size: 30px;
        opacity: 0.15;
        animation: float 15s infinite ease-in-out;
    }
    .house:nth-child(1) { left: 10%; animation-delay: 0s; }
    .house:nth-child(2) { left: 25%; animation-delay: 2s; }
    .house:nth-child(3) { left: 40%; animation-delay: 4s; }
    .house:nth-child(4) { left: 55%; animation-delay: 6s; }
    .house:nth-child(5) { left: 70%; animation-delay: 8s; }
    .house:nth-child(6) { left: 85%; animation-delay: 10s; }
    
    @keyframes float {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 0.15; }
        90% { opacity: 0.15; }
        100% { transform: translateY(-100px) rotate(360deg); opacity: 0; }
    }
    
    /* main header with pulse */
    .main-header {
        text-align: center;
        padding: 25px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 30px;
        animation: headerPulse 3s infinite ease-in-out;
    }
    @keyframes headerPulse {
        0%, 100% { box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4); }
        50% { box-shadow: 0 5px 40px rgba(118, 75, 162, 0.6); }
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        animation: titleBounce 2s infinite ease-in-out;
    }
    @keyframes titleBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.1rem;
    }
    
    /* animated house icon */
    .house-icon {
        font-size: 60px;
        display: inline-block;
        animation: houseWiggle 2s infinite ease-in-out;
    }
    @keyframes houseWiggle {
        0%, 100% { transform: rotate(-5deg); }
        50% { transform: rotate(5deg); }
    }
    
    /* button hover effect */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
    }
    
    /* price box with shine effect */
    .price-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
        animation: pricePopIn 0.5s ease-out;
    }
    @keyframes pricePopIn {
        0% { transform: scale(0); opacity: 0; }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); opacity: 1; }
    }
    .price-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            to right,
            rgba(255,255,255,0) 0%,
            rgba(255,255,255,0.3) 50%,
            rgba(255,255,255,0) 100%
        );
        transform: rotate(30deg);
        animation: shine 3s infinite;
    }
    @keyframes shine {
        0% { transform: translateX(-100%) rotate(30deg); }
        100% { transform: translateX(100%) rotate(30deg); }
    }
    .price-box h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .price-box p {
        color: #f0f0f0;
        font-size: 1rem;
        position: relative;
        z-index: 1;
    }
    
    /* input cards slide in */
    .input-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        0% { transform: translateX(-30px); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    
    /* feature icons bounce */
    .feature-icon {
        display: inline-block;
        animation: iconBounce 1.5s infinite ease-in-out;
    }
    .feature-icon:nth-child(odd) { animation-delay: 0.2s; }
    
    @keyframes iconBounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    
    /* footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #888;
    }
    
    /* sidebar animation */
    .sidebar-house {
        animation: sidebarFloat 3s infinite ease-in-out;
    }
    @keyframes sidebarFloat {
        0%, 100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(5deg); }
    }
</style>

<!-- floating background houses -->
<div class="floating-houses">
    <div class="house">🏠</div>
    <div class="house">🏡</div>
    <div class="house">🏘️</div>
    <div class="house">🏠</div>
    <div class="house">🏡</div>
    <div class="house">🏘️</div>
</div>
""", unsafe_allow_html=True)

# load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('rf_model.joblib')
    enc = joblib.load('encoders.joblib')
    return model, enc

model, enc = load_model()

le_location = enc['le_location']
le_transaction = enc['le_transaction']
le_furnishing = enc['le_furnishing']
le_status = enc['le_status']

# animated header
st.markdown("""
<div class="main-header">
    <span class="house-icon">🏠</span>
    <h1>House Price Prediction</h1>
    <p>Enter property details to get an instant price estimate</p>
</div>
""", unsafe_allow_html=True)

# animated feature icons
st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <span class="feature-icon" style="font-size: 30px; margin: 0 10px;">🛏️</span>
    <span class="feature-icon" style="font-size: 30px; margin: 0 10px;">🛁</span>
    <span class="feature-icon" style="font-size: 30px; margin: 0 10px;">📍</span>
    <span class="feature-icon" style="font-size: 30px; margin: 0 10px;">🔑</span>
    <span class="feature-icon" style="font-size: 30px; margin: 0 10px;">🪑</span>
</div>
""", unsafe_allow_html=True)

# input section
st.subheader('📝 Property Details')

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🛏️ Rooms & Area")
    bhk = st.selectbox('Number of BHK', [1, 2, 3, 4, 5], index=2)
    carpet = st.number_input('Carpet Area (sqft)', min_value=100, max_value=10000, value=1200, step=50)
    bathrooms = st.selectbox('Number of Bathrooms', [1, 2, 3, 4, 5], index=1)

with col2:
    st.markdown("##### 📍 Location & Features")
    location = st.selectbox('Select Location', list(le_location.classes_)[:20])
    transaction = st.selectbox('Transaction Type', list(le_transaction.classes_))
    furnishing = st.selectbox('Furnishing Status', list(le_furnishing.classes_))
    status = st.selectbox('Property Status', list(le_status.classes_))

st.markdown("---")

# predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button('🔮 Predict Price', type='primary', use_container_width=True)

if predict_btn:
    # encode inputs
    loc_enc = le_location.transform([location])[0] if location in le_location.classes_ else 0
    trans_enc = le_transaction.transform([transaction])[0]
    furn_enc = le_furnishing.transform([furnishing])[0]
    stat_enc = le_status.transform([status])[0]
    
    # make prediction
    x = np.array([[bhk, carpet, bathrooms, loc_enc, trans_enc, furn_enc, stat_enc]])
    pred = model.predict(x)[0]
    price_inr = pred * 100000
    
    # show result with animation
    st.balloons()
    
    st.markdown(f"""
    <div class="price-box">
        <h2>💰 ₹{price_inr:,.0f}</h2>
        <p>Estimated Price: {pred:.2f} Lakhs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # animated house celebration
    st.markdown("""
    <div style="text-align: center; margin: 20px 0;">
        <span style="font-size: 50px; display: inline-block; animation: iconBounce 0.5s infinite;">🏠</span>
        <span style="font-size: 40px; display: inline-block; animation: iconBounce 0.5s infinite 0.1s;">✨</span>
        <span style="font-size: 50px; display: inline-block; animation: iconBounce 0.5s infinite 0.2s;">🎉</span>
    </div>
    """, unsafe_allow_html=True)
    
    # show summary
    st.subheader('📊 Prediction Summary')
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("BHK", f"{bhk} BHK")
    with summary_col2:
        st.metric("Area", f"{carpet} sqft")
    with summary_col3:
        st.metric("Location", location.title())

# sidebar with animated house
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;" class="sidebar-house">
        <span style="font-size: 80px;">🏠</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("About")
    st.info("""
    This app predicts house prices using a **Random Forest** ML model.
    
    **Features used:**
    - 🛏️ BHK (Bedrooms)
    - 📐 Carpet Area
    - 🛁 Bathrooms
    - 📍 Location
    - 🔄 Transaction Type
    - 🪑 Furnishing
    - ✅ Status
    """)
    
    st.markdown("---")
    st.subheader("📈 Model Info")
    st.write("**Algorithm:** Random Forest")
    st.write("**Accuracy:** ~84% R² Score")
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")

# footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🎓 House Price Prediction Project | Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
