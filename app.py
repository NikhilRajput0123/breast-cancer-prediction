import streamlit as st
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(to right,#f7f9ff,#eef4ff);
}

/* Main Title */
h1{
    text-align:center;
    font-size:52px !important;
    font-weight:800;
}

/* Subtitle */
.big-font{
    font-size:24px !important;
    color:black;
    text-align:center;
    margin-bottom:35px;
}

/* Cards */
.card{
    background-color:white;
    padding:30px;
    border-radius:22px;
    box-shadow:0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom:25px;
    color:#222;
}

/* Input Labels */
label{
    font-weight:bold !important;
    color:#333 !important;
}

/* Predict Button */
div.stButton > button:first-child {
    background: linear-gradient(to right,#ff4b6e,#ff6f91);
    color:white;
    height:3.5em;
    width:100%;
    border-radius:14px;
    font-size:20px;
    border:none;
    font-weight:bold;
}

/* Footer */
.footer{
    text-align:center;
    color:gray;
    margin-top:35px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown("""
<h1 style='text-align:center;'>

🩺 
<span style='color:black;'>
AI Powered
</span>

<span style='color:#ff4b6e;'>
Breast Cancer Prediction
</span>

</h1>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="big-font">Advanced Machine Learning for Early Detection & Better Tomorrow 💖</p>',
    unsafe_allow_html=True
)

# ---------------- TOP SECTION ---------------- #

col1, col2 = st.columns([2,1])

with col1:

    st.markdown("""
    <div class="card">

    <h2 style='color:#ff4b6e;font-size:40px;'>
    💡 About This Project
    </h2>

    <p style='font-size:20px;color:#333;line-height:1.8;'>

    This AI system predicts whether a breast tumor has a 
    <b>High</b> or <b>Low</b> cancer risk using diagnostic cell features.

    <br><br>

    ✅ Machine Learning Based <br>
    ✅ SVM Classifier <br>
    ✅ 96% Accuracy <br>
    ✅ Fast Prediction System

    </p>

    </div>
    """, unsafe_allow_html=True)

with col2:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/387/387561.png",
        width=300
    )

# ---------------- INPUT SECTION ---------------- #

st.markdown("""
<div class="card">
<h2 style='color:black;'>📋 Enter Tumor Details</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    r = st.number_input(
        "📏 Tumor Radius",
        min_value=0.0,
        format="%.3f"
    )

    p = st.number_input(
        "📐 Tumor Perimeter",
        min_value=0.0,
        format="%.3f"
    )

with col2:

    a = st.number_input(
        "🧩 Tumor Area",
        min_value=0.0,
        format="%.3f"
    )

    c = st.number_input(
        "🧬 Cell Irregularity Score",
        min_value=0.0,
        format="%.3f"
    )

# ---------------- PREDICTION ---------------- #

if st.button("🔍 Predict Cancer Risk"):

    # Correct feature order
    data = np.array([[c, p, r, a]])

    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)

    st.write("")

    if prediction[0] == 1:

        st.error("⚠️ High Risk of Cancer Detected")

        st.warning(
            "Please consult a medical professional for further diagnosis."
        )

        st.image(
            "https://cdn-icons-png.flaticon.com/512/2785/2785482.png",
            width=180
        )

    else:

        st.success("✅ Low Cancer Risk Detected")

        st.balloons()

        st.image(
            "https://cdn-icons-png.flaticon.com/512/845/845646.png",
            width=180
        )

# ---------------- HEALTH TIPS ---------------- #

st.markdown("---")

st.subheader("💖 Health Tips")

tip1, tip2, tip3 = st.columns(3)

with tip1:
    st.info("🥗 Eat healthy food & stay active")

with tip2:
    st.info("🏃 Regular exercise improves health")

with tip3:
    st.info("🩺 Regular checkups save lives")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("🩺 Healthcare AI")

st.sidebar.info(
    "This ML model predicts breast cancer risk using tumor cell measurements."
)

st.sidebar.success("💖 Early detection saves lives")

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4320/4320337.png",
    width=200
)

# ---------------- FOOTER ---------------- #

st.markdown(
    "<div class='footer'>Built with ❤️ using Python, Streamlit & Machine Learning</div>",
    unsafe_allow_html=True
)