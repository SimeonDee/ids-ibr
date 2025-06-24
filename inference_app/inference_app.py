import numpy as np
import pandas as pd
import streamlit as st
# from sklearn.base import ClassifierMixin

from utils import (
    load_model,
    load_encoding,
    load_scaler,
    get_reverse_encoding_dict,
)

from textwrap import dedent

ENCODINGS: dict = load_encoding()
REVERSE_ENCODINGS: dict = get_reverse_encoding_dict(encoding_dict=ENCODINGS)
scaler = load_scaler()
PROTOCOL_DATA = ('tcp', 'udp', 'icmp')
FLAG_DATA = (
    'SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH'
)
SERVICE_SELECTION_DATA = (
    'ftp_data', 'other', 'private', 'http', 'remote_job', 'name',
    'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u',
    'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp',
    'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap',
    'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois',
    'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login',
    'kshell', 'sql_net', 'time', 'hostnames', 'exec', 'ntp_u',
    'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell',
    'netstat', 'pop_3', 'nnsp', 'IRC', 'pop_2', 'printer', 'tim_i',
    'pm_dump', 'red_i', 'netbios_ssn', 'rje', 'X11', 'urh_i', 'http_8001'
)

if "result" not in st.session_state:
    st.session_state.result = ""
    st.session_state.model = ""

st.header(dedent("""
    Design and Implementation of a Machine Learning-based \
    Intrusion Detection System using KNN, SVM and Random \
    Forest to Classify Network Traffic as Normal or Abnormal
"""))

with st.sidebar:
    model_name = st.selectbox(
        "Select Model",
        options=("Random Forest", "SVM", "KNN"),
        index=0
    )
    model = load_model(model_name)

with st.container(border=True):
    protocol_type = st.selectbox(
        "Protocol Type",
        options=PROTOCOL_DATA,
        index=0,
    )
    service = st.selectbox("Service", options=SERVICE_SELECTION_DATA, index=0)
    flag = st.selectbox("Flag", options=FLAG_DATA, index=1)
    src_bytes = st.number_input("Source Bytes", min_value=0)
    dest_bytes = st.number_input("Destination Bytes", min_value=0)
    same_srv_rate = st.slider(
        "Same Server Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )
    diff_srv_rate = st.slider(
        "Different Server Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )
    dst_host_srv_count = st.number_input(
        "Destination Host Server Count",
        min_value=0,
        max_value=255,
        value=255
    )
    dst_host_same_srv_rate = st.slider(
        "Destination Host Same Server Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )
    dst_host_same_src_port_rate = st.slider(
        "Destination Host Same Source Port Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
    )

with st.container(border=True):
    predict_button = st.button(
        "Predict", type="primary", use_container_width=True)

    if predict_button:
        data = np.array([
            PROTOCOL_DATA.index(protocol_type),
            SERVICE_SELECTION_DATA.index(service),
            FLAG_DATA.index(flag),
            int(src_bytes),
            int(dest_bytes),
            float(same_srv_rate),
            float(diff_srv_rate),
            int(dst_host_srv_count),
            float(dst_host_same_srv_rate),
            float(dst_host_same_src_port_rate)
        ])
        scaled_data = scaler.transform(data.reshape(1, -1))
        predicted_label = model.predict(scaled_data)[0]
        predicted_value = REVERSE_ENCODINGS['class'][predicted_label]
        # data_types = [type(val) for val in data]

        st.session_state.model = f"**Model:** `{model_name}`"
        st.session_state.result = (
            "**Prediction:** "
            f"`{predicted_value.upper()}`"
        )  # no-qa
        st.write(f"Prediction: {predicted_value}")


if st.session_state.result:
    with st.sidebar:
        with st.container(border=True):
            st.write("# Prediction Results\n---\n")
            st.write(f"{st.session_state.model}")
            st.write(f"{st.session_state.result}")
        
        with st.container(border=True):
            st.write("### Model Evaluation Performances")
            perf_results = pd.read_csv(
                "results/results_all_metrics.csv", index_col=0)
            st.write(perf_results)

