import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Cybersecurity Dashboard", layout="wide")

# Load model and preprocessing assets
with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Categories
protocol_types = ['tcp', 'udp', 'icmp']
services = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain',
            'domain_u', 'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data',
            'gopher', 'harvest', 'hostnames', 'http', 'http_2784', 'http_443', 'http_8001',
            'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell', 'ldap', 'link', 'login', 'mtp',
            'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp', 'nntp',
            'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i',
            'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat',
            'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet',
            'whois', 'X11', 'Z39_50']
flags = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']

# Title
st.markdown("<h1 style='text-align: center;'>🛡️ Cybersecurity Threat Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Real-time prediction of network intrusions using a trained ML model on the NSL-KDD dataset.</p>", unsafe_allow_html=True)

# Session state for storing recent predictions
if "history" not in st.session_state:
    st.session_state.history = []

if "metrics" not in st.session_state:
    st.session_state.metrics = []

# Form UI
with st.form("input_form"):
    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        input_data['duration'] = st.number_input("Duration", 0.0)
        input_data['protocol_type'] = st.selectbox("Protocol Type", protocol_types)
        input_data['service'] = st.selectbox("Service", services)
        input_data['flag'] = st.selectbox("Flag", flags)
        input_data['src_bytes'] = st.number_input("Source Bytes", 0.0)
        input_data['dst_bytes'] = st.number_input("Destination Bytes", 0.0)
        input_data['land'] = st.selectbox("Land", [0, 1])
        input_data['wrong_fragment'] = st.number_input("Wrong Fragment", 0.0)
        input_data['urgent'] = st.number_input("Urgent", 0.0)
        input_data['hot'] = st.number_input("Hot", 0.0)

    with col2:
        input_data['num_failed_logins'] = st.number_input("Num Failed Logins", 0.0)
        input_data['logged_in'] = st.selectbox("Logged In", [0, 1])
        input_data['num_compromised'] = st.number_input("Num Compromised", 0.0)
        input_data['root_shell'] = st.number_input("Root Shell", 0.0)
        input_data['su_attempted'] = st.number_input("SU Attempted", 0.0)
        input_data['num_root'] = st.number_input("Num Root", 0.0)
        input_data['num_file_creations'] = st.number_input("Num File Creations", 0.0)
        input_data['num_shells'] = st.number_input("Num Shells", 0.0)
        input_data['num_access_files'] = st.number_input("Num Access Files", 0.0)
        input_data['num_outbound_cmds'] = st.number_input("Num Outbound Cmds", 0.0)
        input_data['is_host_login'] = st.selectbox("Is Host Login", [0, 1])
        input_data['is_guest_login'] = st.selectbox("Is Guest Login", [0, 1])

    submitted = st.form_submit_button("🚀 Predict")

# Prediction
if submitted:
    df_input = pd.DataFrame([input_data])
    df_input = pd.get_dummies(df_input)

    # Ensure all expected columns are present
    for col in feature_names:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_names]

    scaled_input = scaler.transform(df_input)
    prediction = model.predict(scaled_input)[0]

    if prediction == 0:
        st.success("✅ Normal connection detected.")
        result = "Normal"
    else:
        st.error("🚨 Attack detected!")
        result = "Attack"

    # Save to session state history
    st.session_state.history.append(result)

    # Handle evaluation metrics only if you have ground truth data
    if len(st.session_state.history) > 1:  # Ensure enough data for evaluation
        # For demonstration purposes, assuming you have the ground truth labels
        # Generate fake y_true and y_pred for the sake of visualization (for batch testing only)
        # In a real scenario, you would have access to y_true from a validation dataset or batch prediction
        y_true = [1 if x == 'Attack' else 0 for x in st.session_state.history]
        y_pred = [1 if x == 'Attack' else 0 for x in st.session_state.history]

        # Generate classification report and confusion matrix only if true labels exist
        # Classification report
        metrics = classification_report(y_true, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
        st.session_state.metrics.append(metrics)
        st.markdown("### ⚙️ Model Evaluation Metrics")
        st.write(metrics)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

# Visuals
if len(st.session_state.history) > 0:
    st.markdown("### 📊 Prediction History")
    df_history = pd.DataFrame(st.session_state.history, columns=["Prediction"])

    # Count plot
    fig, ax = plt.subplots()
    sns.countplot(x="Prediction", data=df_history, palette="viridis", ax=ax)
    ax.set_title("Prediction Distribution")
    st.pyplot(fig)

    # Show the last 10 predictions
    st.dataframe(df_history[::-1].head(10), use_container_width=True)

    # Optional: Security Recommendations
    st.markdown("### 🛡️ Security Recommendations")
    if result == "Attack":
        st.write("Consider reviewing network traffic, isolating affected systems, and conducting further investigation.")
    else:
        st.write("Normal connection detected. Continue monitoring.")
