# =========================================================================
# PHASE 2.1: IMPORTS & APPLICATION CONFIGURATION
# =========================================================================

import os
import time
import threading
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ----------------------------------------------------------
# Streamlit Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="DeepVision AI",
    page_icon="🛡️",
    layout="wide"
)

# ----------------------------------------------------------
# Design System — loaded from style.css (kept separate from
# app logic; no backend code below is affected by this)
# ----------------------------------------------------------

def load_css(file_path: str):
    """
    Reads a local CSS file and injects it into the page.
    Keeping styling in style.css instead of an inline string
    makes it easy to tweak the design without touching app logic.
    """
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(
            f"'{file_path}' not found — the app will still work, "
            "just without the custom styling. Make sure style.css "
            "sits next to app.py."
        )


load_css("style.css")

st.markdown("""
<div class="df-hero">
    <div class="df-badge"><i class="bi bi-lightning-charge-fill"></i> Powered by EfficientNet-B4 + Grad-CAM</div>
    <div class="df-brand">
        <svg class="df-logo" width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M20 3L34 9V19C34 27.5 28 34.5 20 37C12 34.5 6 27.5 6 19V9L20 3Z"
                  stroke="#F1F5F9" stroke-width="1.6" stroke-linejoin="round" fill="none"/>
            <path d="M20 3L34 9V19C34 27.5 28 34.5 20 37V3Z" fill="#F1F5F9" fill-opacity="0.10"/>
            <circle cx="20" cy="19" r="6.5" stroke="#F1F5F9" stroke-width="1.4" fill="none"/>
            <circle cx="20" cy="19" r="1.6" fill="#F1F5F9"/>
        </svg>
        <h1>DeepVision AI</h1>
    </div>
    <p>AI-powered deepfake image analysis with real-time detection and
    explainable Grad-CAM visualization.</p>
</div>
""", unsafe_allow_html=True)

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

    if not os.path.exists(MODEL_PATH):

        st.error(
            f"Model file not found at '{MODEL_PATH}'. "
            "Place the trained .keras file next to app.py, or update MODEL_PATH."
        )

        return None

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


@st.cache_resource
def build_gradcam_model(_model, layer_name=LAST_CONV_LAYER):
    """
    Builds the Grad-CAM sub-model (last conv layer + output) ONCE and
    caches it, instead of rebuilding it on every single prediction.
    Rebuilding this graph per-frame was the dominant cost in live webcam mode.
    """

    return tf.keras.models.Model(
        inputs=_model.inputs,
        outputs=[
            _model.get_layer(layer_name).output,
            _model.output
        ]
    )


model = load_detection_engine()

gradcam_model = build_gradcam_model(model) if model is not None else None



# =========================================================================
# PHASE 2.3: GRAD-CAM HEATMAP GENERATION
# =========================================================================

def process_gradcam(img_tensor, grad_model):
    """
    Generates a Grad-CAM heatmap highlighting the regions
    that most influenced the model's prediction.

    `grad_model` is the pre-built (cached) sub-model exposing the last
    conv layer's output alongside the final prediction — see
    build_gradcam_model(). Rebuilding this per-call was expensive enough
    to be the main bottleneck in live webcam mode, so it's now built once.
    """

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

def predict_image(image, threshold=0.5):
    """
    Preprocesses an image and performs DeepFake prediction.

    Args:
        image     : RGB numpy array
        threshold : Decision threshold on the raw "fake" probability.
                    Exposed as a sidebar control rather than hardcoded,
                    since the right operating point depends on your
                    validation ROC curve and how costly false negatives
                    (missed fakes) are versus false positives.

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

    # A direct call (model(x)) skips the tf.data pipeline, batching,
    # and progress-tracking overhead that model.predict() sets up on every
    # invocation — overhead that's negligible once but adds up fast when
    # called 20-30 times a second from the webcam loop.
    prediction = model(processed, training=False).numpy()[0][0]

    inference_time = (time.time() - start) * 1000

    # Classification
    if prediction >= threshold:

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

st.sidebar.markdown(
    '<h3>⚙️ Settings</h3>',
    unsafe_allow_html=True
)

if model is not None:
    st.sidebar.markdown(
        '<p>🟢 <b>Model ready</b> &nbsp;·&nbsp; EfficientNet-B4</p>',
        unsafe_allow_html=True
    )
else:
    st.sidebar.markdown(
        '<p><i class="bi bi-x-circle-fill" style="color:#EF4444;"></i> '
        '<b>Model not loaded</b></p>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

decision_threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "Raw prediction scores at or above this value are classified as FAKE. "
        "Lower it to catch more fakes at the cost of more false alarms."
    )
)

show_gradcam_live = st.sidebar.checkbox(
    "Show Grad-CAM in webcam mode",
    value=False,
    help=(
        "Grad-CAM adds noticeable per-frame compute cost. "
        "Leave off for smoother live video, enable to inspect model attention."
    )
)

st.sidebar.markdown(
    '<p class="df-caption"><i class="bi bi-arrow-repeat"></i> '
    'Takes effect on next <b>START</b> — changing this while the webcam '
    'is already running won\'t affect the current stream.</p>',
    unsafe_allow_html=True
)

with st.sidebar.expander("Advanced Options"):

    gradcam_every_n = st.slider(
        "Recompute Grad-CAM every N frames",
        min_value=1,
        max_value=15,
        value=5,
        help="Higher values trade heatmap freshness for smoother frame rate.",
        disabled=not show_gradcam_live
    )

with st.sidebar.expander("About"):

    st.write(f"**Model:** EfficientNet-B4")
    st.write(f"**Input Size:** {IMG_SIZE} × {IMG_SIZE}")
    st.markdown(
        '<p class="df-caption"><i class="bi bi-exclamation-triangle-fill"></i> '
        'Research/demo tool. Predictions are not forensic-grade evidence '
        'and should not be used as the sole basis for real-world decisions.</p>',
        unsafe_allow_html=True
    )



# ---------------------------------------------------------------------
# Navigation — pill tabs replace the old sidebar radio
# ---------------------------------------------------------------------

tab1, tab2 = st.tabs(["Image Analysis", "Live Webcam"])


# ---------------------------------------------------------------------
# PHASE 2.6: STATIC IMAGE FORENSIC ANALYSIS
# ---------------------------------------------------------------------

with tab1:

    if model is not None:

        with st.container(border=True):
            st.markdown("#### Upload an image")
            uploaded_img = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
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
            ) = predict_image(original_image, threshold=decision_threshold)

            # Grad-CAM
            try:

                heatmap = process_gradcam(
                    processed_tensor,
                    gradcam_model
                )

                gradcam_overlay = blend_heatmap_overlay(
                    original_image,
                    heatmap
                )

            except Exception as e:

                gradcam_overlay = None
                st.warning(f"Grad-CAM visualization unavailable: {e}")

            # --------------------------------------------------
            # Display Layout
            # --------------------------------------------------

            col1, col2 = st.columns([1, 1])

            with col1:

                with st.container(border=True):
                    st.image(
                        pil_img,
                        caption="Uploaded Image",
                        use_container_width=True
                    )

            with col2:

                with st.container(border=True):

                    badge_class = "df-pred-real" if status == "REAL" else "df-pred-fake"
                    icon = (
                        '<i class="bi bi-patch-check-fill"></i>'
                        if status == "REAL"
                        else '<i class="bi bi-exclamation-octagon-fill"></i>'
                    )

                    st.markdown(
                        f"""
                        <div class="df-pred-badge {badge_class}">{icon} {status}</div>
                        <div class="df-metric-grid">
                            <div class="df-metric">
                                <div class="df-metric-label">Confidence</div>
                                <div class="df-metric-value">{confidence:.2f}%</div>
                            </div>
                            <div class="df-metric">
                                <div class="df-metric-label">Inference Time</div>
                                <div class="df-metric-value">{inference_time:.2f} ms</div>
                            </div>
                            <div class="df-metric">
                                <div class="df-metric-label">Raw Score</div>
                                <div class="df-metric-value">{prediction:.4f}</div>
                            </div>
                            <div class="df-metric">
                                <div class="df-metric-label">Model</div>
                                <div class="df-metric-value" style="font-size:0.95rem;">EfficientNet-B4</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if gradcam_overlay is not None:

                with st.container(border=True):
                    st.markdown("#### Grad-CAM Visualization")
                    st.image(
                        gradcam_overlay,
                        caption="Model Attention Heatmap",
                        use_container_width=True
                    )

    else:
        st.error("Model could not be loaded — check the sidebar for details.")


# =========================================================================
# PHASE 3.0: WEBRTC VIDEO PROCESSOR
# =========================================================================

class VideoProcessor(VideoProcessorBase):
    """
    recv() is called once per incoming video frame and whatever it returns
    becomes the displayed frame. If recv() blocks on model inference (and
    especially on Grad-CAM's extra backward pass), frames back up faster
    than they're consumed and the video visibly stalls/stutters.

    To keep video smooth, recv() never runs the model itself. It hands the
    latest frame to a background worker thread and immediately draws the
    most recently *finished* result onto the current frame. The camera feed
    always stays live; the prediction overlay just lags slightly behind
    (usually well under 100ms) instead of freezing the whole feed.
    """

    def __init__(self, threshold=0.5, gradcam_enabled=False, gradcam_every_n=5):

        self.threshold = threshold
        self.gradcam_enabled = gradcam_enabled
        self.gradcam_every_n = max(1, gradcam_every_n)

        self._lock = threading.Lock()
        self._pending_frame = None   # newest frame waiting to be processed
        self._latest_result = None   # most recent completed prediction/heatmap
        self._frame_count = 0
        self._stopped = False

        self._worker = threading.Thread(target=self._inference_loop, daemon=True)
        self._worker.start()

    def _inference_loop(self):
        """
        Runs continuously on its own thread, always working on the newest
        available frame (older queued frames are simply dropped — there's
        no value in predicting on stale video).
        """

        while not self._stopped:

            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            try:

                (
                    _prediction,
                    confidence,
                    status,
                    processed_tensor,
                    inference_time
                ) = predict_image(frame, threshold=self.threshold)

                heatmap = None

                if self.gradcam_enabled:

                    self._frame_count += 1

                    if self._frame_count % self.gradcam_every_n == 0:

                        try:
                            heatmap = process_gradcam(processed_tensor, gradcam_model)

                        except Exception as e:
                            print(f"[Grad-CAM] webcam frame failed: {e}")
                            heatmap = None

                with self._lock:

                    # Reuse the previous heatmap on frames we didn't recompute,
                    # so the overlay doesn't flicker on/off between recomputes.
                    prev_heatmap = (
                        self._latest_result["heatmap"]
                        if self._latest_result else None
                    )

                    self._latest_result = {
                        "status": status,
                        "confidence": confidence,
                        "inference_time": inference_time,
                        "heatmap": heatmap if heatmap is not None else prev_heatmap,
                    }

            except Exception as e:
                print(f"[Inference] webcam frame failed: {e}")

    def stop(self):
        self._stopped = True

    def __del__(self):
        # Best-effort: if the stream is stopped/restarted and this instance
        # is garbage collected, make sure the worker thread doesn't spin
        # forever in the background.
        self._stopped = True

    def recv(self, frame):

        # Browser frame → OpenCV
        image = frame.to_ndarray(format="bgr24")

        # OpenCV BGR → RGB
        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Hand the newest frame to the background worker (non-blocking —
        # this just replaces whatever frame was previously waiting).
        with self._lock:
            self._pending_frame = rgb_image
            result = self._latest_result

        display_frame = rgb_image.copy()

        if result is None:
            # Nothing computed yet (first ~1 frame after start)
            status = "..."
            confidence = 0.0
            inference_time = 0.0
        else:
            status = result["status"]
            confidence = result["confidence"]
            inference_time = result["inference_time"]

            if result["heatmap"] is not None:
                try:
                    display_frame = blend_heatmap_overlay(rgb_image, result["heatmap"])
                except Exception:
                    display_frame = rgb_image.copy()

        # -----------------------------
        # Prediction Label
        # -----------------------------

        if status == "REAL":
            color = (0, 255, 0)
        elif status == "FAKE":
            color = (255, 70, 70)
        else:
            color = (180, 180, 180)

        overlay = display_frame.copy()

        # Compact background card
        cv2.rectangle(
            overlay,
            (15, 15),
            (205, 95),
            (25, 25, 25),
            -1
        )

        # Blend for transparency
        cv2.addWeighted(
            overlay,
            0.35,
            display_frame,
            0.65,
            0,
            display_frame
        )

        # Prediction
        cv2.putText(
            display_frame,
            status,
            (28, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            color,
            2,
            cv2.LINE_AA
        )

        # Confidence
        cv2.putText(
            display_frame,
            f"{confidence:.1f}%",
            (28, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255,255,255),
            2,
            cv2.LINE_AA
        )

        # Inference
        cv2.putText(
            display_frame,
            f"{inference_time:.0f} ms",
            (120,68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180,180,180),
            1,
            cv2.LINE_AA
        )

        return av.VideoFrame.from_ndarray(
            display_frame,
            format="rgb24"
        )



# =========================================================================
# PHASE 3: LIVE WEBCAM USING WEBRTC
# =========================================================================

with tab2:

    if model is not None:

        with st.container(border=True):

            st.markdown("#### Live Webcam Feed")
            st.caption("Click **START** below to enable your webcam.")

            gradcam_pill_class = "df-toggle-on" if show_gradcam_live else "df-toggle-off"
            gradcam_pill_text = "Grad-CAM: ON" if show_gradcam_live else "Grad-CAM: OFF"
            st.markdown(
                f'<div class="df-toggle-pill {gradcam_pill_class}">'
                f'<span class="df-toggle-dot"></span>{gradcam_pill_text}</div>',
                unsafe_allow_html=True
            )

            if not show_gradcam_live:
                st.caption(
                    "Grad-CAM is off for a smoother feed. Enable it in the sidebar "
                    "if you want to see the attention heatmap (adds latency)."
                )

            def _video_processor_factory():
                # Record exactly what config this processor was built with, so we
                # can later tell if the sidebar has since changed underneath it.
                st.session_state["active_webcam_config"] = {
                    "threshold": decision_threshold,
                    "gradcam_enabled": show_gradcam_live,
                    "gradcam_every_n": gradcam_every_n,
                }
                return VideoProcessor(
                    threshold=decision_threshold,
                    gradcam_enabled=show_gradcam_live,
                    gradcam_every_n=gradcam_every_n,
                )

            ctx = webrtc_streamer(
                key="deepfake-webcam",
                video_processor_factory=_video_processor_factory,
                rtc_configuration={
                    # Required once this app is running somewhere other than
                    # localhost — without a STUN server, WebRTC can't establish
                    # the connection through NAT and the webcam will just hang
                    # on "loading" indefinitely after deployment.
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={
                    "video": True,
                    "audio": False,
                },
                async_processing=True,
            )

            active_config = st.session_state.get("active_webcam_config")

            if ctx.state.playing and active_config is not None:

                current_config = {
                    "threshold": decision_threshold,
                    "gradcam_enabled": show_gradcam_live,
                    "gradcam_every_n": gradcam_every_n,
                }

                if current_config != active_config:

                    st.warning(
                        "Sidebar settings changed since this stream started — "
                        "the running feed is still using the old settings "
                        f"(Grad-CAM {'ON' if active_config['gradcam_enabled'] else 'OFF'}, "
                        f"threshold {active_config['threshold']:.2f}). "
                        "Click **STOP** then **START** below to apply your changes."
                    )

    else:
        st.error("Model could not be loaded — check the sidebar for details.")


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

st.markdown(
    """
    <div class="df-footer">
        Powered by <b>TensorFlow</b> · <b>EfficientNet-B4</b> ·
        <b>Grad-CAM</b> · <b>streamlit-webrtc</b>
    </div>
    """,
    unsafe_allow_html=True
)