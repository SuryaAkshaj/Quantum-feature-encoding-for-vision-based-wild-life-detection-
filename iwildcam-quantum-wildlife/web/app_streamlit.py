"""
Streamlit UI: upload image -> preview detections -> predictions
Run with: streamlit run web/app_streamlit.py
"""
import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Quantum Wildlife Detector",
    page_icon="🐾",
    layout="centered",
)

st.title("🐾 Quantum-Enhanced Wildlife Detector")
st.caption("Powered by YOLOv8 · ResNet-50 · Quantum VQC (PennyLane)")
st.divider()

uploaded = st.file_uploader("📷 Upload a camera-trap image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("🔍 Run Prediction", type="primary"):
        uploaded.seek(0)
        files = {"image": (uploaded.name, uploaded.getvalue(), "image/jpeg")}

        with st.spinner("Running YOLO detection + quantum classification…"):
            try:
                r = requests.post("http://localhost:8000/predict", files=files, timeout=60)

                if r.status_code == 200:
                    data = r.json()
                    preds = data.get("predictions", [])

                    if not preds:
                        st.warning(data.get("message", "No animals detected in image."))
                    else:
                        st.success(f"✅ Found **{len(preds)} animal crop(s)** detected!")
                        st.divider()

                        for i, p in enumerate(preds):
                            label     = p.get("predicted_label", f"Class {p.get('predicted_class', '?')}")
                            conf      = p.get("confidence", 0.0)
                            crop_name = p.get("crop", "")
                            scores    = p.get("class_scores", [])

                            st.subheader(f"🐾 Animal {i + 1}")

                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.markdown(f"**Predicted Species:** `{label}`")
                                st.markdown(f"**Confidence:** `{conf * 100:.1f}%`")
                                st.markdown(f"**Crop file:** `{crop_name}`")
                                st.caption(p.get("detection_note", ""))
                            with col2:
                                # confidence as a big metric
                                st.metric(label="Top Class", value=f"{conf * 100:.1f}%", delta=label)

                            # Bar chart: all class probabilities
                            if scores:
                                st.markdown("**Class probability breakdown:**")
                                chart_data = {
                                    s["label"]: s["probability"] for s in scores
                                }
                                st.bar_chart(chart_data)

                            st.divider()

                else:
                    st.error(f"Server returned status {r.status_code}: {r.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Flask API!")
                st.info("Make sure Flask is running:\n```\npython -m web.app_flask\n```")
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out — model may still be processing.")
            except Exception as e:
                st.error(f"Unexpected error: {type(e).__name__}: {e}")
