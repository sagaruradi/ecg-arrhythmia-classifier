"""
ECG Arrhythmia Signal Classification System
Real-Time Cardiac Signal Classification Using 1D-CNN.
Final Year Engineering Project Demonstration Application.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import torch

import config
from src.data_loader import (
    get_available_records, load_record_signal, 
    load_record_annotations
)
from src.preprocessing import apply_bandpass_filter, normalize_segment
from src.predict import get_model, predict_heartbeat

# Page configuration
st.set_page_config(
    page_title="ECG Arrhythmia Classification System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme-Adaptive CSS for high-end professional appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .card {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
    }
    .badge-normal {
        background-color: #2a9d8f;
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-arrhythmia {
        background-color: #e63946;
        color: white;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        display: inline-block;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #00b4d8;
    }
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 2px;
    }
    .explanation-box {
        background: rgba(0, 180, 216, 0.08);
        border-left: 4px solid #00b4d8;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.95rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.markdown('<div class="sidebar-header">🫀 CardioPulse AI</div>', unsafe_allow_html=True)
st.sidebar.markdown("**ECG Arrhythmia Classification System**  \n*1D-CNN Neural Architecture*")
st.sidebar.markdown("---")

section = st.sidebar.radio(
    "System Navigation",
    [
        "🏠 1. Home & Overview",
        "📈 2. ECG Signal Viewer",
        "🩺 3. Heartbeat Prediction",
        "📊 4. Model Results & Metrics"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Benchmark: MIT-BIH Arrhythmia Database  \nArchitecture: 1D-CNN (PyTorch)  \nSampling Rate: 360 Hz")


# ==============================================================================
# SECTION 1: HOME & OVERVIEW
# ==============================================================================
if section == "🏠 1. Home & Overview":
    st.markdown('<div class="main-title">ECG Arrhythmia Classification System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated Cardiac Signal Classification Using 1D Convolutional Neural Networks</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Objective
        Electrocardiography (ECG) is the primary non-invasive diagnostic modality for evaluating cardiac electrical activity.
        Continuous ambulatory Holter monitoring generates up to 100,000 heartbeats per patient in a 24-hour window, making manual visual review exhaustive and susceptible to human diagnostic error.
        
        This engineering project delivers an **automated, high-precision, and real-time cardiac signal classification system** that:
        1. Ingests raw multi-patient ECG recordings directly from the gold-standard **MIT-BIH Arrhythmia Database**.
        2. Attenuates low-frequency baseline drift (<0.5 Hz) and high-frequency powerline/EMG noise (>45 Hz) using a digital **Butterworth bandpass filter**.
        3. Segments continuous waveforms into fixed 256-sample single-heartbeat windows around annotated R-peaks.
        4. Classifies heartbeats into **4 AAMI categories** using a custom **1D Convolutional Neural Network (1D-CNN)** with 4-phase cardiac temporal pooling.
        5. Strictly prevents data leakage by employing **patient-level dataset partitioning**.
        """)
        
        st.markdown("""
        ### 🔄 End-to-End Processing Pipeline
        """)
        st.info(
            "**Raw ECG Voltage (.dat/.hea)** ➔ **Butterworth Bandpass Filter (0.5–45 Hz)** ➔ "
            "**R-Peak Segmentation (256 samples)** ➔ **Z-Score Normalization** ➔ "
            "**1D-CNN Feature Extraction** ➔ **Softmax Probability Distribution** ➔ **Arrhythmia Classification**"
        )
        
    with col2:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        * **Core Programming:** Python 3.10+
        * **Biomedical Signal I/O:** `wfdb` (Official PhysioNet library)
        * **Signal Preprocessing:** `scipy.signal` (Digital Butterworth filtering)
        * **Numerical Computing:** `numpy`
        * **Deep Learning Framework:** `PyTorch` (CPU-compatible 1D-CNN)
        * **Performance Metrics:** `scikit-learn`
        * **Visualization:** `matplotlib`
        * **Demonstration Dashboard:** `streamlit`
        """)
        
        st.markdown("### 🫀 Target Heartbeat Categories (AAMI)")
        st.markdown("""
        * **Normal (N):** Normal sinus rhythm, bundle branch blocks (LBBB/RBBB)
        * **Supraventricular Ectopic (S):** Premature atrial or nodal contractions
        * **Ventricular Ectopic (V):** Premature ventricular contractions (PVCs)
        * **Fusion (F):** Hybrid collision of normal and ventricular ectopic beats
        """)

    st.markdown("---")
    st.markdown("### 📋 System Demonstration Guide")
    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        st.markdown("**1. Signal Viewer (Page 2)**")
        st.caption("Inspect raw clinical signals vs. bandpass-filtered waveforms and doctor R-peak annotations.")
    with vcol2:
        st.markdown("**2. Real-Time Classifier (Page 3)**")
        st.caption("Select any heartbeat window, run live 1D-CNN inference, and view prediction confidence.")
    with vcol3:
        st.markdown("**3. Clinical Metrics (Page 4)**")
        st.caption("Review genuine test accuracy, per-class sensitivity, confusion matrix, and training curves.")


# ==============================================================================
# SECTION 2: ECG SIGNAL VIEWER
# ==============================================================================
elif section == "📈 2. ECG Signal Viewer":
    st.markdown('<div class="main-title">ECG Signal & Annotation Viewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect raw clinical waveforms, Butterworth bandpass filtering, and doctor R-peak annotations</div>', unsafe_allow_html=True)
    
    records = get_available_records()
    if not records:
        st.error("No MIT-BIH records found in data directory. Please check the dataset path in config.py.")
        st.stop()
        
    # Controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 3, 2])
    with col_ctrl1:
        selected_record = st.selectbox("Select Patient Record:", records, index=0)
    with col_ctrl2:
        start_time_sec = st.slider("Start Time (seconds):", min_value=0, max_value=300, value=10, step=2)
    with col_ctrl3:
        duration_sec = st.selectbox("Window Duration:", [3, 5, 8, 10], index=1)
        
    # Load signal and annotations
    try:
        raw_dir = config.get_raw_data_dir()
        raw_signal, fs, lead_name = load_record_signal(selected_record, raw_dir)
        annotation = load_record_annotations(selected_record, raw_dir)
        filtered_signal = apply_bandpass_filter(raw_signal, fs=fs)
    except Exception as e:
        st.error(f"Error reading record {selected_record}: {e}")
        st.stop()
        
    start_sample = int(start_time_sec * fs)
    end_sample = int((start_time_sec + duration_sec) * fs)
    
    if end_sample > len(raw_signal):
        end_sample = len(raw_signal)
        start_sample = max(0, end_sample - int(duration_sec * fs))
        
    t = np.linspace(start_time_sec, start_time_sec + duration_sec, end_sample - start_sample)
    raw_segment = raw_signal[start_sample:end_sample]
    filt_segment = filtered_signal[start_sample:end_sample]
    
    # R-peaks in this window
    peaks_in_window = [
        (s, sym) for s, sym in zip(annotation.sample, annotation.symbol)
        if start_sample <= s < end_sample
    ]
    
    # Display record metadata
    st.markdown(f"""
    **Record:** `{selected_record}` | **Lead:** `{lead_name}` | **Sampling Rate:** `{fs} Hz` | 
    **Annotated Beats in Window:** `{len(peaks_in_window)}` | 
    **Total Record Samples:** `{len(raw_signal):,}` (~{len(raw_signal)/(fs*60):.1f} mins)
    """)
    
    # Dark-Theme Matplotlib Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, dpi=130)
    fig.patch.set_facecolor('#0e1117')
    
    # Raw signal
    ax1.set_facecolor('#161b22')
    ax1.plot(t, raw_segment, color="#888888", linewidth=1.1)
    ax1.set_title(f"Raw ECG Signal (Lead {lead_name}) - Baseline Wander & Noise", fontsize=11, fontweight="bold", color="white")
    ax1.set_ylabel("Voltage (mV)", fontsize=9, color="white")
    ax1.tick_params(colors="white")
    ax1.grid(True, linestyle=":", alpha=0.3, color="gray")
    
    # Filtered signal
    ax2.set_facecolor('#161b22')
    ax2.plot(t, filt_segment, color="#00b4d8", linewidth=1.4, label="Filtered ECG (0.5 - 45 Hz)")
    
    for s_idx, sym in peaks_in_window:
        t_peak = s_idx / fs
        rel_idx = min(len(filt_segment) - 1, max(0, s_idx - start_sample))
        y_val = filt_segment[rel_idx]
        
        # Color code annotation
        marker_color = "#2a9d8f" if sym in ['N', 'L', 'R'] else "#e63946"
        ax2.plot(t_peak, y_val, 'o', color=marker_color, markersize=7)
        ax2.annotate(f"{sym}", xy=(t_peak, y_val), xytext=(0, 10),
                     textcoords='offset points', ha='center', fontsize=9, fontweight='bold', color=marker_color)
                     
    ax2.set_title("Butterworth Bandpass Filtered Signal with R-Peak Annotations", fontsize=11, fontweight="bold", color="white")
    ax2.set_xlabel("Time (seconds)", fontsize=9, color="white")
    ax2.set_ylabel("Voltage (mV)", fontsize=9, color="white")
    ax2.tick_params(colors="white")
    ax2.grid(True, linestyle=":", alpha=0.3, color="gray")
    ax2.legend(loc="upper right", facecolor="#161b22", edgecolor="gray", labelcolor="white", fontsize=8)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.caption("🟢 Green markers: Normal/Conduction beats. 🔴 Red markers: Arrhythmia / Ectopic beats.")


# ==============================================================================
# SECTION 3: HEARTBEAT PREDICTION
# ==============================================================================
elif section == "🩺 3. Heartbeat Prediction":
    st.markdown('<div class="main-title">Heartbeat Arrhythmia Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Classify isolated 256-sample heartbeat waveforms using the trained 1D-CNN</div>', unsafe_allow_html=True)
    
    if not config.MODEL_PATH.exists():
        st.warning("⚠️ Trained model not found. Please train the model first by running: `python main.py --train`")
        st.stop()
        
    test_cache = config.PROCESSED_DATA_DIR / "test_data.npz"
    if not test_cache.exists():
        st.warning("⚠️ Processed test data not found. Please run: `python main.py --prepare`")
        st.stop()
        
    test_data = np.load(test_cache, allow_pickle=True)
    test_segments = test_data['segments']
    test_labels = test_data['labels']
    test_record_ids = test_data['record_ids']
    
    st.markdown("### 1. Select a Heartbeat to Classify")
    
    c1, c2, c3 = st.columns([2, 2, 3])
    
    with c1:
        class_filter = st.selectbox(
            "Filter Category:",
            ["All Classes"] + config.CLASS_NAMES
        )
        
    if class_filter == "All Classes":
        candidate_indices = np.arange(len(test_labels))
    else:
        target_cls = config.CLASS_NAMES.index(class_filter)
        candidate_indices = np.where(test_labels == target_cls)[0]
        
    if len(candidate_indices) == 0:
        st.error(f"No samples of class {class_filter} in test split.")
        st.stop()
        
    with c2:
        selected_idx_local = st.selectbox(
            f"Select Beat (Total: {len(candidate_indices)}):",
            range(min(50, len(candidate_indices))),
            format_func=lambda i: f"Sample #{candidate_indices[i]} (Record {test_record_ids[candidate_indices[i]]})"
        )
        actual_sample_idx = candidate_indices[selected_idx_local]
        
    with c3:
        st.write("")
        st.write("")
        if st.button("🎲 Pick Random Sample", use_container_width=True):
            actual_sample_idx = np.random.choice(candidate_indices)
            
    # Extract segment
    segment = test_segments[actual_sample_idx]
    true_label_idx = int(test_labels[actual_sample_idx])
    source_record = test_record_ids[actual_sample_idx]
    
    # Run prediction
    prediction = predict_heartbeat(segment)
    
    st.markdown("---")
    
    # Display 2 columns: Waveform on left, Prediction results on right
    col_wave, col_pred = st.columns([3, 2])
    
    with col_wave:
        st.markdown(f"**Segment Waveform (256 samples, Record {source_record})**")
        fig, ax = plt.subplots(figsize=(7, 3.8), dpi=130)
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#161b22')
        
        time_axis = np.arange(len(segment)) / config.SAMPLING_RATE * 1000  # in ms
        
        ax.plot(time_axis, segment, color="#00b4d8", linewidth=2.0, label="Normalized Beat")
        # Mark R-peak location (sample 100)
        r_time = (config.WINDOW_BEFORE / config.SAMPLING_RATE) * 1000
        ax.axvline(r_time, color="#e63946", linestyle="--", alpha=0.7, label="R-Peak Center")
        ax.plot(r_time, segment[config.WINDOW_BEFORE], 'ro', markersize=6)
        
        ax.set_title(f"Isolated Heartbeat Window (~711 ms)", fontsize=11, fontweight="bold", color="white")
        ax.set_xlabel("Time (milliseconds)", fontsize=9, color="white")
        ax.set_ylabel("Normalized Voltage (Z-Score)", fontsize=9, color="white")
        ax.tick_params(colors="white")
        ax.grid(True, linestyle=":", alpha=0.3, color="gray")
        ax.legend(loc="upper right", facecolor="#161b22", edgecolor="gray", labelcolor="white", fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with col_pred:
        st.markdown("**1D-CNN Classification Result:**")
        
        is_normal = (prediction['class_id'] == 0)
        badge_style = "badge-normal" if is_normal else "badge-arrhythmia"
        
        st.markdown(f"""
        <div class="card">
            <span class="{badge_style}" style="font-size: 1.15rem;">
                {prediction['class_name']}
            </span>
            <div style="margin-top: 12px; font-size: 1.05rem;">
                <strong>Confidence:</strong> <span style="color: #00b4d8; font-weight: 700;">{prediction['confidence_percent']}%</span>
            </div>
            <div style="font-size: 0.88rem; opacity: 0.85; margin-top: 6px;">
                {prediction['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Ground truth comparison
        is_match = (prediction['class_id'] == true_label_idx)
        match_symbol = "✅ Correct" if is_match else "❌ Misclassified"
        
        st.markdown(f"""
        **Expert Physician Reference:** `{config.CLASS_NAMES[true_label_idx]}` ({match_symbol})  
        **Patient Record ID:** `{source_record}` (Held-out Test Record)
        """)
        
        # Probability distribution bars
        st.markdown("**Softmax Probability Breakdown:**")
        for cls_name, prob in prediction['probabilities'].items():
            st.write(f"{cls_name}: **{prob}%**")
            st.progress(float(prob) / 100.0)

    # Educational class overview card
    st.markdown("---")
    st.markdown("### 🫀 Supported AAMI Arrhythmia Categories Explained")
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.markdown("""
        * **Normal Beat (N):** Standard sinus beat initiated in the SA node with normal conduction, or bundle branch blocks (LBBB/RBBB).
        * **Supraventricular Ectopic Beat (S):** Premature beat originating abnormally in the atria or AV node (Atrial Premature Contractions).
        """)
    with ecol2:
        st.markdown("""
        * **Ventricular Ectopic Beat (V):** Ectopic beat originating abnormally inside the ventricles (Premature Ventricular Contractions / PVCs).
        * **Fusion Beat (F):** Electrical collision occurring when a ventricular ectopic impulse and a normal supraventricular impulse meet simultaneously.
        """)


# ==============================================================================
# SECTION 4: MODEL RESULTS & METRICS
# ==============================================================================
elif section == "📊 4. Model Results & Metrics":
    st.markdown('<div class="main-title">Evaluation Metrics & Clinical Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Experimental results validated on unseen patient recordings (Records: ' + ", ".join(config.TEST_RECORDS) + ')</div>', unsafe_allow_html=True)
    
    if not config.METRICS_PATH.exists():
        st.warning("⚠️ Metrics have not been computed yet. Please evaluate the model first by running: `python main.py --evaluate`")
        st.stop()
        
    with open(config.METRICS_PATH, 'r') as f:
        metrics = json.load(f)
        
    # High Level KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 0.85rem; opacity: 0.8;">Overall Accuracy</div>
            <div class="metric-value">""" + f"{metrics['accuracy_percent']}%" + """</div>
            <div style="font-size: 0.75rem; color: #2a9d8f;">All test beats</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 0.85rem; opacity: 0.8;">Macro F1-Score</div>
            <div class="metric-value">""" + f"{metrics['macro_f1']*100:.1f}%" + """</div>
            <div style="font-size: 0.75rem; color: #00b4d8;">Equal class weight</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 0.85rem; opacity: 0.8;">Weighted Precision</div>
            <div class="metric-value">""" + f"{metrics['weighted_precision']*100:.1f}%" + """</div>
            <div style="font-size: 0.75rem; color: #2a9d8f;">True positive rate</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown("""
        <div class="card" style="text-align: center;">
            <div style="font-size: 0.85rem; opacity: 0.8;">Weighted Recall</div>
            <div class="metric-value">""" + f"{metrics['weighted_recall']*100:.1f}%" + """</div>
            <div style="font-size: 0.75rem; color: #e63946;">Detection rate</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"**Test Validation:** Evaluated on **{metrics['total_test_samples']:,}** heartbeat windows from held-out patient recordings **{metrics['test_records']}**.")
    
    # Per-Class Table
    st.markdown("### 📋 Per-Class Performance Breakdown")
    per_class = metrics['per_class_metrics']
    
    table_data = []
    for cls_name, vals in per_class.items():
        table_data.append({
            "Heartbeat Category": cls_name,
            "Precision (%)": f"{vals['precision']*100:.2f}%",
            "Recall (%)": f"{vals['recall']*100:.2f}%",
            "F1-Score (%)": f"{vals['f1_score']*100:.2f}%",
            "Test Beats (Support)": f"{vals['support']:,}"
        })
    st.table(table_data)
    
    # Clinical Insight Box on the Accuracy Paradox
    st.markdown("""
    <div class="explanation-box">
        <strong>⚠️ Clinical Insight — Why Accuracy Alone is Misleading in ECG Analysis:</strong><br>
        In real medical datasets, normal sinus beats constitute over 80% of all heartbeats, while arrhythmias are rare (minority classes).
        A naive model predicting 'Normal' on every single beat would achieve 80%+ accuracy, yet have a <strong>0% detection rate (recall)</strong> on life-threatening arrhythmias.<br>
        Therefore, clinical AI must be judged by <strong>Per-Class Recall (Sensitivity)</strong> and <strong>Macro F1-Score</strong>, which treat all categories equally regardless of sample frequency.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Graphical Plots
    st.markdown("### 📈 Visual Performance Analysis")
    tab1, tab2, tab3 = st.tabs(["🔲 Confusion Matrix", "📉 Training Curves", "📊 Class Distribution"])
    
    with tab1:
        cm_path = config.RESULTS_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix on Held-Out Test Patients", use_container_width=True)
        else:
            st.info("Run `python main.py --evaluate` to generate the confusion matrix plot.")
            
    with tab2:
        curves_path = config.RESULTS_DIR / "training_curves.png"
        if curves_path.exists():
            st.image(str(curves_path), caption="Cross-Entropy Loss, Accuracy, and Validation Macro-F1 per Epoch", use_container_width=True)
        else:
            st.info("Run `python main.py --train` to generate training curves.")
            
    with tab3:
        dist_path = config.RESULTS_DIR / "class_distribution.png"
        if dist_path.exists():
            st.image(str(dist_path), caption="Heartbeat Count per Class (Train vs Val vs Test)", use_container_width=True)
        else:
            st.info("Run `python main.py --evaluate` to generate class distribution plot.")
            
    st.markdown("---")
    st.markdown("### 💡 Clinical Performance Metric Definitions")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("""
        * **Accuracy:** The overall proportion of heartbeats that were correctly classified across the entire test set.
        * **Precision:** When the model alerts that a heartbeat is of a specific class (e.g. Ventricular Ectopic), how often is it actually correct? Higher precision means fewer false alarms.
        """)
    with v2:
        st.markdown("""
        * **Recall (Sensitivity):** Out of all the real abnormal beats that existed in the patient's recording, how many did the model successfully detect? Crucial in healthcare to avoid missing arrhythmias.
        * **F1-Score:** The harmonic mean between precision and recall, providing a balanced single performance metric even when classes are heavily imbalanced.
        """)
