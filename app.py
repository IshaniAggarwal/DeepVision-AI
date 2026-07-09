# =========================================================================
# PHASE 2.1: IMPORTS & APPLICATION CONFIGURATION
# =========================================================================

import time
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------------------------------------------------
# Streamlit Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="DeepFake Detection System",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ DeepFake Detection System")
st.write(
    "DeepFake detection using EfficientNet-B4 with Grad-CAM explainability."
)

# ----------------------------------------------------------
# Global Configuration
# ----------------------------------------------------------

IMG_SIZE = 380

MODEL_PATH = "final_model.keras"

LAST_CONV_LAYER = "top_activation"



# =========================================================================
# PHASE 2.2: LOAD TRAINED MODEL
# =========================================================================

@st.cache_resource
def load_detection_engine():
    """
    Loads the trained EfficientNet-B4 model once and caches it
    for the lifetime of the Streamlit session.
    """

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        st.error("Unable to load the trained model.")
        st.error(str(e))

        return None


model = load_detection_engine()

if model is not None:
    st.sidebar.success("Model Loaded")
else:
    st.sidebar.error("Model Not Loaded")



# =========================================================================
# PHASE 2.3: GRAD-CAM HEATMAP GENERATION
# =========================================================================

def process_gradcam(img_tensor, target_model, layer_name=LAST_CONV_LAYER):
    """
    Generates a Grad-CAM heatmap highlighting the regions
    that most influenced the model's prediction.
    """

    # Build Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=target_model.inputs,
        outputs=[
            target_model.get_layer(layer_name).output,
            target_model.output
        ]
    )

    # Record gradients
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_tensor)

        # Binary classifier score
        loss = predictions[:, 0]

    # Compute gradients
    gradients = tape.gradient(loss, conv_outputs)

    # Global Average Pooling over feature maps
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # Weighted feature map combination
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # Apply ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize heatmap
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    return heatmap.numpy()



# =========================================================================
# PHASE 2.4: GRAD-CAM HEATMAP OVERLAY
# =========================================================================

def blend_heatmap_overlay(original_image, heatmap, alpha=0.4):
    """
    Blends the Grad-CAM heatmap with the original RGB image
    to visualize the regions influencing the prediction.
    """

    # Resize heatmap to match original image dimensions
    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    # Convert heatmap to 8-bit image
    heatmap = np.uint8(255 * heatmap)

    # Apply JET color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert OpenCV BGR -> RGB
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Blend heatmap with original image
    overlay = cv2.addWeighted(
        original_image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay



# =========================================================================
# PHASE 2.5: PREDICTION ENGINE & USER INTERFACE
# =========================================================================

def predict_image(image):
    """
    Preprocesses an image and performs DeepFake prediction.

    Returns:
        prediction     : Raw model probability
        confidence     : Prediction confidence (%)
        status         : REAL or FAKE
        processed_img  : Preprocessed tensor
        inference_time : Prediction time (ms)
    """

    # Resize image
    resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert to tensor
    processed = np.expand_dims(
        resized.astype(np.float32),
        axis=0
    )

    # Apply EfficientNet preprocessing
    processed = preprocess_input(processed)

    # Measure inference time
    start = time.time()

    prediction = model.predict(
        processed,
        verbose=0
    )[0][0]

    inference_time = (time.time() - start) * 1000

    # Classification
    if prediction >= 0.5:

        status = "FAKE"

        confidence = prediction * 100

    else:

        status = "REAL"

        confidence = (1 - prediction) * 100

    return (
        prediction,
        confidence,
        status,
        processed,
        inference_time
    )


# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.header("Control Center")

mode = st.sidebar.radio(
    "Select Input Mode",
    (
        "Static Image Upload",
        "Live Webcam Feed"
    )
)

st.sidebar.markdown("---")

st.sidebar.write(f"**Model:** EfficientNet-B4")
st.sidebar.write(f"**Input Size:** {IMG_SIZE} × {IMG_SIZE}")

if model is not None:
    st.sidebar.success("Ready")
else:
    st.sidebar.error("Model Not Loaded")



# ---------------------------------------------------------------------
# PHASE 2.6: STATIC IMAGE FORENSIC ANALYSIS
# ---------------------------------------------------------------------

if model is not None:

    if mode == "Static Image Upload":

        st.subheader("🖼️ Static Image Analysis")

        uploaded_img = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_img:

            # Read image
            pil_img = Image.open(uploaded_img).convert("RGB")
            original_image = np.array(pil_img)

            # Prediction
            (
                prediction,
                confidence,
                status,
                processed_tensor,
                inference_time
            ) = predict_image(original_image)

            # Grad-CAM
            try:

                heatmap = process_gradcam(
                    processed_tensor,
                    model
                )

                gradcam_overlay = blend_heatmap_overlay(
                    original_image,
                    heatmap
                )

            except Exception:

                gradcam_overlay = None

            # --------------------------------------------------
            # Display Layout
            # --------------------------------------------------

            col1, col2 = st.columns([1, 1])

            with col1:

                st.image(
                    pil_img,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            with col2:

                st.subheader("Prediction Results")

                if status == "REAL":

                    st.success(
                        f"Prediction: {status}"
                    )

                else:

                    st.error(
                        f"Prediction: {status}"
                    )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.metric(
                    "Inference Time",
                    f"{inference_time:.2f} ms"
                )

                st.metric(
                    "Raw Prediction",
                    f"{prediction:.4f}"
                )

            if gradcam_overlay is not None:

                st.subheader("Grad-CAM Visualization")

                st.image(
                    gradcam_overlay,
                    caption="Model Attention Heatmap",
                    use_container_width=True
                )



# =========================================================================
# PHASE 3: REAL-TIME WEBCAM DETECTION
# =========================================================================

if model is not None and mode == "Live Webcam Feed":

    st.subheader("🎥 Real-Time Webcam Detection")

    st.write(
        "Start the webcam to perform real-time DeepFake detection "
        "with Grad-CAM visualization."
    )

    # ----------------------------------------------------------
    # Session State
    # ----------------------------------------------------------

    if "streaming" not in st.session_state:
        st.session_state.streaming = False

    # ----------------------------------------------------------
    # Controls
    # ----------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "▶ Start Webcam",
            use_container_width=True
        ):
            st.session_state.streaming = True

    with col2:

        if st.button(
            "⏹ Stop Webcam",
            use_container_width=True
        ):
            st.session_state.streaming = False

    # ----------------------------------------------------------
    # Webcam Pipeline
    # ----------------------------------------------------------

    if st.session_state.streaming:

        st.success("Webcam is running...")

        frame_placeholder = st.empty()

        metrics_placeholder = st.empty()

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            st.error("Unable to access the webcam.")

        else:

            while st.session_state.streaming:

                success, frame = camera.read()

                if not success:

                    st.error("Failed to capture frame.")

                    break

                # ------------------------------------------
                # Convert Frame
                # ------------------------------------------

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # ------------------------------------------
                # Prediction
                # ------------------------------------------

                (
                    prediction,
                    confidence,
                    status,
                    processed_tensor,
                    inference_time
                ) = predict_image(rgb_frame)

                # ------------------------------------------
                # Grad-CAM
                # ------------------------------------------

                try:

                    heatmap = process_gradcam(
                        processed_tensor,
                        model
                    )

                    display_frame = blend_heatmap_overlay(
                        rgb_frame,
                        heatmap
                    )

                except Exception:

                    display_frame = rgb_frame.copy()

                # ------------------------------------------
                # Prediction Label
                # ------------------------------------------

                color = (
                    (0, 255, 0)
                    if status == "REAL"
                    else
                    (255, 0, 0)
                )

                cv2.putText(
                    display_frame,
                    f"{status} ({confidence:.1f}%)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA
                )

                # ------------------------------------------
                # Show Webcam
                # ------------------------------------------

                frame_placeholder.image(
                    display_frame,
                    channels="RGB",
                    use_container_width=True
                )

                # ------------------------------------------
                # Live Metrics
                # ------------------------------------------

                metrics_placeholder.markdown(
                    f"""
                    ### Live Detection

                    **Prediction:** {status}

                    **Confidence:** {confidence:.2f}%

                    **Inference Time:** {inference_time:.2f} ms

                    **Raw Prediction:** {prediction:.4f}
                    """
                )

            # ------------------------------------------
            # Cleanup
            # ------------------------------------------

            camera.release()

            st.session_state.streaming = False

            st.success("Webcam stopped successfully.")

    else:

        st.info(
            "Click 'Start Webcam' to begin live detection."
        )