# Advanced Fraud Detection System

A comprehensive fraud detection system built with FastAPI backend and Gradio frontend, featuring explainable AI through SHAP analysis.

## Features

- **Real-time Fraud Detection**: Analyze e-commerce and credit card transactions
- **Explainable AI**: SHAP-powered explanations for model predictions
- **Batch Processing**: Upload CSV files for bulk fraud analysis
- **Interactive Dashboard**: User-friendly Gradio interface
- **REST API**: Complete FastAPI backend with comprehensive endpoints
- **Production Ready**: Docker containerization and health monitoring

## System Architecture

```
┌─────────────────┐    HTTP     ┌─────────────────┐
│   Gradio        │<----------->│   FastAPI       │
│   Frontend      │   Requests  │   Backend       │
│   (Port 7860)   │             │   (Port 8000)   │
└─────────────────┘             └─────────────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │   ML Models     │
                                │   - RandomForest│
                                │   - XGBoost     │
                                │   - SHAP        │
                                └─────────────────┘
```

## Quick Start

### Option 1: Docker Compose (Recommended)

1. **Clone and prepare the project:**

```bash
git clone <your-repo>
cd fraud-detection-system
```

2. **Create required directories:**

```bash
mkdir models logs
```

3. **Start the services:**

```bash
docker-compose up --build
```

4. **Access the applications:**

- Gradio Frontend: http://localhost:7860
- FastAPI Backend: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Option 2: Local Development

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Start FastAPI backend:**

```bash
uvicorn main:app --reload --port 8000
```

3. **Start Gradio frontend (in new terminal):**

```bash
python app.py
```

## Model Integration

To integrate your trained models from the Colab notebook:

### 1. Save Your Trained Models

Add this to your Colab notebook:

```python
import joblib

# Save models
joblib.dump(fraud_model, '/content/drive/MyDrive/fraud_detection/fraud_model.pkl')
joblib.dump(cc_model, '/content/drive/MyDrive/fraud_detection/cc_model.pkl')

# Save preprocessors
joblib.dump(preprocessor, '/content/drive/MyDrive/fraud_detection/preprocessor.pkl')

# Save SHAP explainer
joblib.dump(explainer, '/content/drive/MyDrive/fraud_detection/explainer.pkl')
```

### 2. Update Backend Configuration

Modify `main.py` to load your actual models:

```python
def load_models():
    global fraud_model, cc_model, fraud_preprocessor, cc_preprocessor

    fraud_model = joblib.load('models/fraud_model.pkl')
    cc_model = joblib.load('models/cc_model.pkl')
    fraud_preprocessor = joblib.load('models/preprocessor.pkl')
```

### 3. Update Feature Engineering

Replace the simplified preprocessing functions with your actual feature engineering pipeline from the notebook.

## API Endpoints

### Core Prediction Endpoints

- `POST /predict/transaction` - E-commerce fraud prediction
- `POST /predict/creditcard` - Credit card fraud prediction
- `POST /predict/batch` - Batch processing via CSV upload

### Information Endpoints

- `GET /health` - System health check
- `GET /model/info` - Model performance metrics
- `GET /` - API status

## Frontend Features

### 1. E-commerce Transaction Analysis

- Input transaction details through intuitive form
- Real-time fraud probability calculation
- SHAP-based feature importance explanations
- Risk level visualization

### 2. Credit Card Transaction Analysis

- Support for PCA-transformed features (V1-V28)
- Specialized credit card fraud detection
- Detailed risk assessment

### 3. Batch Processing

- CSV file upload support
- Bulk fraud detection
- Summary statistics and visualizations
- Downloadable results

### 4. Model Information

- Real-time model performance metrics
- Feature importance insights
- Model comparison statistics

## Configuration

### Environment Variables

Create `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
GRADIO_HOST=0.0.0.0
GRADIO_PORT=7860

# Model Paths
FRAUD_MODEL_PATH=models/fraud_model.pkl
CC_MODEL_PATH=models/cc_model.pkl
PREPROCESSOR_PATH=models/preprocessor.pkl

# Optional: Database
DATABASE_URL=postgresql://user:password@localhost:5432/fraud_db

# Optional: Redis Cache
REDIS_URL=redis://localhost:6379
```

### Docker Configuration

The system includes:

- FastAPI backend container
- Gradio frontend container
- Optional Redis for caching
- Optional PostgreSQL for logging

## Production Deployment

### 1. Cloud Deployment (AWS/GCP/Azure)

```bash
# Build and push images
docker build -f Dockerfile.api -t fraud-api:latest .
docker build -f Dockerfile.frontend -t fraud-frontend:latest .

# Deploy to cloud container service
```

### 2. Kubernetes Deployment

Create Kubernetes manifests:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-api
  template:
    metadata:
      labels:
        app: fraud-api
    spec:
      containers:
        - name: fraud-api
          image: fraud-api:latest
          ports:
            - containerPort: 8000
```

### 3. Security Considerations

- Implement authentication/authorization
- Add rate limiting
- Enable HTTPS/TLS
- Input validation and sanitization
- Audit logging
- Data encryption

## Monitoring and Logging

### Health Checks

- `/health` endpoint monitors system status
- Docker health checks included
- Kubernetes readiness/liveness probes supported

### Logging

The system logs:

- Prediction requests and responses
- Model performance metrics
- Error events and stack traces
- System health status

### Metrics

Consider adding:

- Prometheus metrics endpoint
- Grafana dashboards
- Alert management
- Performance monitoring

## Testing

### Unit Tests

```bash
pytest tests/
```

### API Testing

```bash
# Test fraud prediction
curl -X POST "http://localhost:8000/predict/transaction" \
     -H "Content-Type: application/json" \
     -d '{...transaction_data...}'
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -p transaction.json -T application/json \
   http://localhost:8000/predict/transaction
```

## Troubleshooting

### Common Issues

1. **Models not loading**

   - Ensure model files exist in `models/` directory
   - Check file permissions and paths
   - Verify model compatibility

2. **SHAP explanations failing**

   - Update SHAP version
   - Check feature consistency
   - Verify model type support

3. **High memory usage**

   - Reduce SHAP sample size
   - Implement model caching
   - Monitor container resources

4. **API connection errors**
   - Verify backend is running on correct port
   - Check firewall settings
   - Validate network connectivity

### Performance Optimization

- Enable request caching with Redis
- Implement async processing for batch jobs
- Use model quantization for faster inference
- Add load balancing for multiple replicas

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues and questions:

- Check the troubleshooting section
- Review API documentation at `/docs`
- Create GitHub issue with detailed description
