import gradio as gr
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import io
import base64

# Configuration
API_BASE_URL = "http://localhost:8000"  # FastAPI backend URL

# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}

.fraud-detected {
    background-color: #fee2e2 !important;
    border: 2px solid #dc2626 !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.fraud-clear {
    background-color: #dcfce7 !important;
    border: 2px solid #16a34a !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.high-risk {
    color: #dc2626 !important;
    font-weight: bold !important;
}

.low-risk {
    color: #16a34a !important;
    font-weight: bold !important;
}

.medium-risk {
    color: #ea580c !important;
    font-weight: bold !important;
}
"""

def call_api(endpoint: str, data: Dict = None, files: Dict = None) -> Dict:
    """Make API call to FastAPI backend"""
    try:
        url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
        
        if files:
            response = requests.post(url, files=files)
        elif data:
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return {"error": f"API call failed: {str(e)}"}

def format_prediction_result(result: Dict) -> Tuple[str, str, str, str]:
    """Format prediction result for display"""
    if "error" in result:
        return "Error", result["error"], "", ""
    
    # Main result
    fraud_status = "🚨 FRAUD DETECTED" if result["is_fraud"] else "✅ LEGITIMATE"
    risk_class = "fraud-detected" if result["is_fraud"] else "fraud-clear"
    
    # Risk information
    risk_info = f"""
    **Risk Level**: {result["risk_score"]}
    **Fraud Probability**: {result["fraud_probability"]:.1%}
    **Confidence**: {result["confidence"]:.1%}
    **Model Used**: {result["model_used"]}
    """
    
    # Explanation
    explanation = ""
    if "explanation" in result and "top_factors" in result["explanation"]:
        explanation = "**Top Contributing Factors:**\n\n"
        for i, (factor, details) in enumerate(result["explanation"]["top_factors"].items(), 1):
            impact = details["contribution"]
            value = details["value"]
            shap_val = details["shap_value"]
            explanation += f"{i}. **{factor}**: {value:.3f} ({impact} risk by {abs(shap_val):.3f})\n"
    
    # Technical details
    technical = f"""
    **Timestamp**: {result.get("timestamp", "N/A")}
    **Model Confidence**: {result["explanation"].get("model_confidence", 0):.1%}
    **Baseline Risk**: {result["explanation"].get("baseline_prediction", 0):.1%}
    """
    
    return fraud_status, risk_info, explanation, technical

def create_risk_visualization(result: Dict) -> go.Figure:
    """Create risk visualization"""
    if "error" in result:
        return go.Figure()
    
    prob = result["fraud_probability"]
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fraud Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300, showlegend=False)
    return fig

def predict_transaction(signup_time, purchase_time, purchase_value, device_id, 
                       source, browser, sex, age, ip_address):
    """Predict fraud for e-commerce transaction"""
    
    # Validate inputs
    if not all([signup_time, purchase_time, device_id, source, browser, sex, ip_address]):
        return "❌ Error", "Please fill in all required fields", "", "", None
    
    if age < 18 or age > 100:
        return "❌ Error", "Age must be between 18 and 100", "", "", None
    
    if purchase_value <= 0:
        return "❌ Error", "Purchase value must be positive", "", "", None
    
    # Prepare data
    data = {
        "signup_time": signup_time,
        "purchase_time": purchase_time,
        "purchase_value": purchase_value,
        "device_id": device_id,
        "source": source,
        "browser": browser,
        "sex": sex,
        "age": age,
        "ip_address": ip_address
    }
    
    # Call API
    result = call_api("/predict/transaction", data)
    
    # Format result
    fraud_status, risk_info, explanation, technical = format_prediction_result(result)
    
    # Create visualization
    risk_viz = create_risk_visualization(result)
    
    return fraud_status, risk_info, explanation, technical, risk_viz

def predict_creditcard(time_val, amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                      v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                      v21, v22, v23, v24, v25, v26, v27, v28):
    """Predict fraud for credit card transaction"""
    
    if amount < 0:
        return "❌ Error", "Amount must be non-negative", "", "", None
    
    # Prepare data
    data = {
        "time": time_val,
        "amount": amount,
        **{f"v{i}": locals()[f"v{i}"] for i in range(1, 29)}
    }
    
    # Call API
    result = call_api("/predict/creditcard", data)
    
    # Format result
    fraud_status, risk_info, explanation, technical = format_prediction_result(result)
    
    # Create visualization
    risk_viz = create_risk_visualization(result)
    
    return fraud_status, risk_info, explanation, technical, risk_viz

def process_batch_file(file):
    """Process batch predictions from uploaded file"""
    if file is None:
        return "Please upload a CSV file", None, None
    
    try:
        # Read the file
        df = pd.read_csv(file.name)
        
        # Prepare file for API
        with open(file.name, 'rb') as f:
            files = {"file": (file.name, f, "text/csv")}
            result = call_api("/predict/batch", files=files)
        
        if "error" in result:
            return f"Error: {result['error']}", None, None
        
        # Create summary
        summary = f"""
        ## Batch Processing Results
        
        **Total Transactions**: {result['total_transactions']}
        **Fraud Detected**: {result['fraud_detected']}
        **Fraud Rate**: {result['fraud_rate']:.1%}
        
        ### Summary Statistics
        - **Average Fraud Probability**: {result['summary_stats']['avg_fraud_probability']:.1%}
        - **Maximum Risk Score**: {result['summary_stats']['max_fraud_probability']:.1%}
        - **Standard Deviation**: {result['summary_stats']['std_fraud_probability']:.1%}
        """
        
        # Create visualizations
        probabilities = [pred["fraud_probability"] for pred in result["predictions"]]
        fraud_labels = ["Fraud" if pred["is_fraud"] else "Legitimate" for pred in result["predictions"]]
        
        # Distribution plot
        fig1 = px.histogram(
            x=probabilities, 
            nbins=50, 
            title="Distribution of Fraud Probabilities",
            labels={"x": "Fraud Probability", "y": "Count"}
        )
        
        # Fraud vs Legitimate pie chart
        fraud_counts = pd.Series(fraud_labels).value_counts()
        fig2 = px.pie(
            values=fraud_counts.values, 
            names=fraud_counts.index,
            title="Fraud Detection Summary"
        )
        
        # Create detailed results DataFrame
        results_df = pd.DataFrame([
            {
                "Transaction_ID": i+1,
                "Is_Fraud": pred["is_fraud"],
                "Fraud_Probability": pred["fraud_probability"],
                "Risk_Level": pred["risk_score"],
                "Confidence": pred["confidence"],
                "Model": pred["model_used"]
            }
            for i, pred in enumerate(result["predictions"])
        ])
        
        return summary, fig1, fig2, results_df
        
    except Exception as e:
        return f"Error processing file: {str(e)}", None, None, None

def get_model_info():
    """Get model information from API"""
    result = call_api("/model/info")
    
    if "error" in result:
        return f"Error fetching model info: {result['error']}"
    
    info = f"""
    ## Model Information
    
    ### E-commerce Fraud Model
    - **Type**: {result['fraud_model']['type']}
    - **Features**: {result['fraud_model']['features']}
    - **Accuracy**: {result['fraud_model']['accuracy']:.1%}
    - **Precision**: {result['fraud_model']['precision']:.1%}
    - **Recall**: {result['fraud_model']['recall']:.1%}
    - **F1-Score**: {result['fraud_model']['f1_score']:.1%}
    
    ### Credit Card Fraud Model  
    - **Type**: {result['creditcard_model']['type']}
    - **Features**: {result['creditcard_model']['features']}
    - **Accuracy**: {result['creditcard_model']['accuracy']:.1%}
    - **Precision**: {result['creditcard_model']['precision']:.1%}
    - **Recall**: {result['creditcard_model']['recall']:.1%}
    - **F1-Score**: {result['creditcard_model']['f1_score']:.1%}
    
    **Last Updated**: {result['last_updated']}
    """
    
    return info

# Create Gradio Interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(css=custom_css, title="Advanced Fraud Detection System") as demo:
        
        gr.Markdown("""
        # 🔍 Advanced Fraud Detection System
        ### Powered by Machine Learning & Explainable AI
        
        This system provides real-time fraud detection for e-commerce transactions and credit card payments,
        with detailed explanations powered by SHAP (SHapley Additive exPlanations).
        """)
        
        with gr.Tabs() as tabs:
            
            # E-commerce Transaction Tab
            with gr.TabItem("🛒 E-commerce Transaction"):
                gr.Markdown("### Analyze e-commerce transaction for fraud")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        signup_time = gr.Textbox(
                            label="Signup Time", 
                            value="2024-01-01 10:00:00",
                            info="Format: YYYY-MM-DD HH:MM:SS"
                        )
                        purchase_time = gr.Textbox(
                            label="Purchase Time", 
                            value="2024-01-01 11:00:00",
                            info="Format: YYYY-MM-DD HH:MM:SS"
                        )
                        purchase_value = gr.Number(label="Purchase Value ($)", value=100.0)
                        device_id = gr.Textbox(label="Device ID", value="device_12345")
                        ip_address = gr.Textbox(label="IP Address", value="192.168.1.1")
                        
                    with gr.Column(scale=1):
                        source = gr.Dropdown(
                            label="Traffic Source", 
                            choices=["SEO", "Ads", "Direct"], 
                            value="SEO"
                        )
                        browser = gr.Dropdown(
                            label="Browser", 
                            choices=["Chrome", "Firefox", "Safari", "IE", "Opera"], 
                            value="Chrome"
                        )
                        sex = gr.Dropdown(label="Gender", choices=["M", "F"], value="M")
                        age = gr.Number(label="Age", value=35)
                        
                        predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        fraud_status = gr.Textbox(label="Fraud Detection Result", interactive=False)
                        risk_info = gr.Markdown(label="Risk Information")
                        
                    with gr.Column(scale=1):
                        risk_viz = gr.Plot(label="Risk Visualization")
                
                with gr.Row():
                    explanation = gr.Markdown(label="AI Explanation")
                    technical_details = gr.Markdown(label="Technical Details")
                
                predict_btn.click(
                    fn=predict_transaction,
                    inputs=[signup_time, purchase_time, purchase_value, device_id, 
                           source, browser, sex, age, ip_address],
                    outputs=[fraud_status, risk_info, explanation, technical_details, risk_viz]
                )
            
            # Credit Card Transaction Tab
            with gr.TabItem("💳 Credit Card Transaction"):
                gr.Markdown("### Analyze credit card transaction for fraud")
                gr.Markdown("*Note: V1-V28 are PCA-transformed features from the original dataset*")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        time_val = gr.Number(label="Time", value=0.0)
                        amount = gr.Number(label="Amount ($)", value=100.0)
                        
                    with gr.Column(scale=1):
                        v1 = gr.Number(label="V1", value=0.0)
                        v2 = gr.Number(label="V2", value=0.0)
                        v3 = gr.Number(label="V3", value=0.0)
                        v4 = gr.Number(label="V4", value=0.0)
                        v5 = gr.Number(label="V5", value=0.0)
                        v6 = gr.Number(label="V6", value=0.0)
                        v7 = gr.Number(label="V7", value=0.0)
                        
                    with gr.Column(scale=1):
                        v8 = gr.Number(label="V8", value=0.0)
                        v9 = gr.Number(label="V9", value=0.0)
                        v10 = gr.Number(label="V10", value=0.0)
                        v11 = gr.Number(label="V11", value=0.0)
                        v12 = gr.Number(label="V12", value=0.0)
                        v13 = gr.Number(label="V13", value=0.0)
                        v14 = gr.Number(label="V14", value=0.0)
                        
                    with gr.Column(scale=1):
                        v15 = gr.Number(label="V15", value=0.0)
                        v16 = gr.Number(label="V16", value=0.0)
                        v17 = gr.Number(label="V17", value=0.0)
                        v18 = gr.Number(label="V18", value=0.0)
                        v19 = gr.Number(label="V19", value=0.0)
                        v20 = gr.Number(label="V20", value=0.0)
                        v21 = gr.Number(label="V21", value=0.0)
                        
                    with gr.Column(scale=1):
                        v22 = gr.Number(label="V22", value=0.0)
                        v23 = gr.Number(label="V23", value=0.0)
                        v24 = gr.Number(label="V24", value=0.0)
                        v25 = gr.Number(label="V25", value=0.0)
                        v26 = gr.Number(label="V26", value=0.0)
                        v27 = gr.Number(label="V27", value=0.0)
                        v28 = gr.Number(label="V28", value=0.0)
                
                cc_predict_btn = gr.Button("Analyze Credit Card Transaction", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        cc_fraud_status = gr.Textbox(label="Fraud Detection Result", interactive=False)
                        cc_risk_info = gr.Markdown(label="Risk Information")
                        
                    with gr.Column(scale=1):
                        cc_risk_viz = gr.Plot(label="Risk Visualization")
                
                with gr.Row():
                    cc_explanation = gr.Markdown(label="AI Explanation")
                    cc_technical_details = gr.Markdown(label="Technical Details")
                
                cc_predict_btn.click(
                    fn=predict_creditcard,
                    inputs=[time_val, amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                           v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                           v21, v22, v23, v24, v25, v26, v27, v28],
                    outputs=[cc_fraud_status, cc_risk_info, cc_explanation, cc_technical_details, cc_risk_viz]
                )
            
            # Batch Processing Tab
            with gr.TabItem("📊 Batch Processing"):
                gr.Markdown("### Upload CSV file for batch fraud detection")
                gr.Markdown("Upload a CSV file with transaction data to process multiple transactions at once.")
                
                with gr.Row():
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload CSV File",
                            file_types=[".csv"],
                            type="filepath"
                        )
                        
                        gr.Markdown("""
                        **Expected CSV Format for E-commerce:**
                        - signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address
                        
                        **Expected CSV Format for Credit Card:**
                        - Time, Amount, V1, V2, ..., V28
                        """)
                        
                        batch_btn = gr.Button("Process Batch File", variant="primary")
                
                batch_summary = gr.Markdown(label="Processing Summary")
                
                with gr.Row():
                    batch_viz1 = gr.Plot(label="Fraud Probability Distribution")
                    batch_viz2 = gr.Plot(label="Detection Summary")
                
                batch_results = gr.Dataframe(
                    label="Detailed Results",
                    headers=["Transaction_ID", "Is_Fraud", "Fraud_Probability", "Risk_Level", "Confidence", "Model"],
                    interactive=False
                )
                
                batch_btn.click(
                    fn=process_batch_file,
                    inputs=[file_upload],
                    outputs=[batch_summary, batch_viz1, batch_viz2, batch_results]
                )
            
            # Model Information Tab
            with gr.TabItem("🤖 Model Information"):
                gr.Markdown("### Model Performance and Technical Details")
                
                model_info_btn = gr.Button("Refresh Model Information", variant="secondary")
                model_info_display = gr.Markdown()
                
                model_info_btn.click(
                    fn=get_model_info,
                    outputs=[model_info_display]
                )
                
                # Load initial model info
                demo.load(
                    fn=get_model_info,
                    outputs=[model_info_display]
                )
            
            # API Documentation Tab
            with gr.TabItem("📚 API Documentation"):
                gr.Markdown("""
                ### API Endpoints
                
                The fraud detection system provides the following REST API endpoints:
                
                #### 1. Transaction Fraud Detection
                **POST** `/predict/transaction`
                
                **Request Body:**
                ```json
                {
                    "signup_time": "2024-01-01 10:00:00",
                    "purchase_time": "2024-01-01 11:00:00", 
                    "purchase_value": 100.0,
                    "device_id": "device_12345",
                    "source": "SEO",
                    "browser": "Chrome",
                    "sex": "M",
                    "age": 35,
                    "ip_address": "192.168.1.1"
                }
                ```
                
                #### 2. Credit Card Fraud Detection
                **POST** `/predict/creditcard`
                
                **Request Body:**
                ```json
                {
                    "time": 0.0,
                    "amount": 100.0,
                    "v1": 0.0,
                    "v2": 0.0,
                    ...
                    "v28": 0.0
                }
                ```
                
                #### 3. Batch Processing
                **POST** `/predict/batch`
                
                Upload CSV file with multiple transactions.
                
                #### 4. Model Information
                **GET** `/model/info`
                
                Returns current model performance metrics and details.
                
                #### 5. Health Check
                **GET** `/health`
                
                Returns API health status.
                
                ### Response Format
                
                All prediction endpoints return:
                ```json
                {
                    "is_fraud": boolean,
                    "fraud_probability": float,
                    "risk_score": string,
                    "confidence": float,
                    "model_used": string,
                    "explanation": {
                        "top_factors": {...},
                        "model_confidence": float,
                        "explanation_method": "SHAP",
                        "baseline_prediction": float
                    },
                    "timestamp": string
                }
                ```
                
                ### SHAP Explanations
                
                The system provides explainable AI through SHAP (SHapley Additive exPlanations):
                - **Feature Importance**: Shows which features contributed most to the prediction
                - **Impact Direction**: Indicates whether each feature increases or decreases fraud risk
                - **Confidence Metrics**: Provides model confidence levels
                - **Baseline Comparison**: Shows how the prediction compares to the baseline risk
                """)
        
        # Footer
        gr.Markdown("""
        ---
        **Advanced Fraud Detection System** | Built with FastAPI + Gradio | Powered by XGBoost & Random Forest
        
         **Privacy Notice**: This system processes transaction data locally. No data is stored permanently.
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    # Launch with custom configuration
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        debug=True,
        show_api=False,
        quiet=False
    )
