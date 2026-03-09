import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from PIL import Image
import io
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CLASSES = ["Normal", "Lung_Opacity", "Viral_Pneumonia"]
IMG_SIZE = 224

# ─────────────────────────────────────────────
# AGENT 1: DATA AGENT
# ─────────────────────────────────────────────
class DataAgent:
    def preprocess_uploaded_image(self, img_bytes):
        """Preprocess image bytes into normalized numpy array."""
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image. Please upload a valid X-ray image.")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        return img

# ─────────────────────────────────────────────
# AGENT 2: VISION AGENT
# ─────────────────────────────────────────────
class VisionAgent:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
            Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def infer(self, image):
        """Run inference on preprocessed image."""
        image_input = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        probs = self.model.predict(image_input, verbose=0)
        return probs[0]

# ─────────────────────────────────────────────
# AGENT 3: REASONING AGENT
# ─────────────────────────────────────────────
class ReasoningAgent:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def analyze(self, probs):
        """Analyze probabilities and return reasoning output."""
        max_prob = float(np.max(probs))
        pred_class = int(np.argmax(probs))
        return {
            "class": pred_class,
            "confidence": max_prob,
            "ambiguous": max_prob < self.threshold,
            "all_probs": probs
        }

# ─────────────────────────────────────────────
# AGENT 4: DECISION AGENT
# ─────────────────────────────────────────────
class DecisionAgent:
    def decide(self, reasoning):
        """Return final decision string."""
        if reasoning["ambiguous"]:
            return "⚠️ NEEDS DOCTOR REVIEW (Low Confidence)"
        else:
            return f"✅ {CLASSES[reasoning['class']]}"

# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Lung X-Ray Classification",
        page_icon="🫁",
        layout="wide"
    )

    # Header
    st.title("🫁 Multi-Agent Lung X-Ray Classification System")
    st.markdown("Upload a chest X-ray image to classify it as **Normal**, **Lung Opacity**, or **Viral Pneumonia**.")
    st.divider()

    # Initialize agents
    @st.cache_resource
    def load_agents():
        return {
            "data": DataAgent(),
            "vision": VisionAgent(),
            "reasoning": ReasoningAgent(threshold=0.6),
            "decision": DecisionAgent()
        }

    agents = load_agents()

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This system uses **4 AI Agents**:
        
        - 🗂️ **Data Agent** — Preprocesses the X-ray image
        - 👁️ **Vision Agent** — CNN model for classification
        - 🧠 **Reasoning Agent** — Analyzes confidence levels
        - ⚖️ **Decision Agent** — Makes the final diagnosis
        """)
        st.divider()
        st.markdown("**Confidence Threshold:** 60%")
        st.markdown("Below threshold → Doctor review recommended")

    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Upload Chest X-Ray Image",
        type=["jpg", "jpeg", "png"],
        help="Upload a grayscale or color chest X-ray image"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(uploaded_file, caption="Original X-Ray", use_column_width=True)

        if st.button("🚀 Run Diagnostic Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running multi-agent pipeline..."):

                img_bytes = uploaded_file.read()

                # ── AGENT 1: Data Agent ──────────────────────
                with st.expander("🗂️ Agent 1: Data Agent", expanded=True):
                    try:
                        img = agents["data"].preprocess_uploaded_image(img_bytes)
                        st.success("✅ Image preprocessed successfully")
                        st.write(f"**Shape:** {img.shape} | **Min:** {img.min():.3f} | **Max:** {img.max():.3f}")

                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.imshow(img, cmap='gray')
                        ax.set_title("Preprocessed X-Ray (224×224, Normalized)")
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close()

                    except Exception as e:
                        st.error(f"❌ Data Agent Error: {e}")
                        st.stop()

                # ── AGENT 2: Vision Agent ────────────────────
                with st.expander("👁️ Agent 2: Vision Agent", expanded=True):
                    probs = agents["vision"].infer(img)
                    st.success("✅ Classification complete")

                    # Probability bar chart
                    prob_data = {cls: float(prob) for cls, prob in zip(CLASSES, probs)}
                    st.bar_chart(prob_data)

                    for cls, prob in zip(CLASSES, probs):
                        st.progress(float(prob), text=f"{cls}: {prob:.4f}")

                # ── AGENT 3: Reasoning Agent ─────────────────
                with st.expander("🧠 Agent 3: Reasoning Agent", expanded=True):
                    reasoning = agents["reasoning"].analyze(probs)
                    st.write(f"**Predicted Class:** `{CLASSES[reasoning['class']]}`")
                    st.write(f"**Confidence:** `{reasoning['confidence']:.1%}`")

                    if reasoning["ambiguous"]:
                        st.warning("⚠️ Confidence is below 60% — case is ambiguous")
                    else:
                        st.success("✅ Confidence is sufficient for a reliable prediction")

                # ── AGENT 4: Decision Agent ───────────────────
                st.divider()
                st.subheader("⚖️ Agent 4: Final Decision")
                decision = agents["decision"].decide(reasoning)

                if reasoning["ambiguous"]:
                    st.error(decision)
                    st.info("💡 Recommendation: Please consult a radiologist for further evaluation.")
                else:
                    st.success(decision)
                    if reasoning["class"] == 0:
                        st.info("💚 Lungs appear normal. No significant abnormalities detected.")
                    elif reasoning["class"] == 1:
                        st.warning("🟠 Lung Opacity detected. May indicate infection, fluid, or other conditions.")
                    else:
                        st.error("🔴 Viral Pneumonia pattern detected. Medical attention recommended.")

                st.divider()
                st.caption(f"Analysis complete | Confidence: {reasoning['confidence']:.1%} | Model: CNN (untrained demo)")


if __name__ == "__main__":
    main()