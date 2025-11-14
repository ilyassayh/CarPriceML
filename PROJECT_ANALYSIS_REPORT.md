# CarPriceML - Detailed Project Analysis Report

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Project Architecture](#project-architecture)
3. [File-by-File Analysis](#file-by-file-analysis)
4. [How the System Works](#how-the-system-works)
5. [How to Run the Project](#how-to-run-the-project)
6. [Technical Details](#technical-details)

---

## 🎯 Project Overview

**CarPriceML** is a complete end-to-end Machine Learning application designed to predict the price of used cars. It's a full-stack ML project that includes:

- **Machine Learning Pipeline**: Automated training with data preprocessing
- **REST API**: FastAPI backend for serving predictions
- **Web Interface**: Streamlit frontend for user interaction
- **Containerization**: Docker setup for easy deployment
- **Testing**: Unit tests for API endpoints

The project uses a **Random Forest Regressor** model with automated feature engineering (OneHotEncoding for categorical features, StandardScaler for numerical features).

---

## 🏗️ Project Architecture

```
CarPriceML/
├── app/                          # Application code
│   ├── api/                      # Backend API
│   │   ├── __init__.py          # Package initializer
│   │   └── main.py              # FastAPI application
│   └── frontend/                 # Frontend UI
│       └── streamlit_app.py     # Streamlit web interface
│
├── pipeline/                     # ML Training pipeline
│   ├── __init__.py              # Package initializer
│   └── train.py                # Model training script
│
├── models/                       # Saved models and metadata
│   ├── rf_model.joblib          # Trained model (generated)
│   └── metadata.json            # Model metadata (generated)
│
├── tests/                        # Test suite
│   └── test_api.py              # API endpoint tests
│
├── car-details.csv              # Training data (user-provided)
│
├── Dockerfile                   # Backend container definition
├── Dockerfile.frontend          # Frontend container definition
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

**Data Flow:**
```
CSV Data → train.py → Model + Metadata → API → Frontend → User
```

---

## 📁 File-by-File Analysis

### 1. `pipeline/train.py` - Model Training Script

**Purpose**: This is the core ML training script that processes raw car data, trains a Random Forest model, and saves both the model and metadata.

**Key Functions:**

#### `detect_feature_types(df, target)`
- **Purpose**: Automatically identifies which columns are numerical vs categorical
- **Logic**: 
  - Categorical: columns with `dtype == 'object'`
  - Numerical: columns with numeric dtypes
- **Returns**: Tuple of (numeric_cols, categorical_cols)

#### `build_pipeline(numeric_cols, categorical_cols)`
- **Purpose**: Constructs a complete ML pipeline with preprocessing and model
- **Components**:
  - **Numerical Pipeline**: 
    - `SimpleImputer(strategy='median')` - Fills missing numeric values
    - `StandardScaler()` - Normalizes numeric features
  - **Categorical Pipeline**:
    - `SimpleImputer(strategy='most_frequent')` - Fills missing categorical values
    - `OneHotEncoder(handle_unknown='ignore')` - Converts categories to binary features
  - **Model**: `RandomForestRegressor(n_estimators=300, random_state=42)`
- **Returns**: Complete sklearn Pipeline

#### `main()`
- **Purpose**: Orchestrates the entire training process
- **Steps**:
  1. Parses command-line arguments
  2. Loads and validates CSV data
  3. Cleans data (removes duplicates, drops rows with missing target)
  4. Applies currency conversion if needed
  5. Splits data into train/test (default 70/30)
  6. Trains the pipeline
  7. Evaluates on test set (RMSE, MAE, R²)
  8. Saves model as `.joblib` file
  9. Saves metadata as JSON

**Command-Line Arguments:**
- `--csv`: Path to CSV file (required)
- `--target`: Target column name (default: 'price')
- `--test-size`: Test split ratio (default: 0.3)
- `--currency-rate`: Multiplier for price conversion (default: 1.0)
- `--out-model`: Output path for model (default: 'models/rf_model.joblib')
- `--out-meta`: Output path for metadata (default: 'models/metadata.json')

**Output Files:**
- `rf_model.joblib`: Complete pipeline (preprocessing + model) saved with joblib
- `metadata.json`: Contains feature names, metrics, training info

---

### 2. `app/api/main.py` - FastAPI Backend

**Purpose**: RESTful API server that loads the trained model and serves predictions.

**Key Components:**

#### FastAPI Application Setup
- Creates FastAPI app instance
- Adds CORS middleware (allows cross-origin requests from frontend)
- Defines paths to model and metadata files

#### `load_artifacts()`
- **Purpose**: Loads the trained model and metadata from disk
- **Returns**: Tuple of (model, metadata_dict)
- **Error Handling**: Raises FileNotFoundError if files don't exist

#### `GET /health` Endpoint
- **Purpose**: Health check endpoint to verify API is running
- **Returns**: 
  ```json
  {
    "status": "ok",
    "model": "rf",
    "features": ["feature1", "feature2", ...]
  }
  ```
- **Error Handling**: Returns error status if model can't be loaded

#### `POST /predict` Endpoint
- **Purpose**: Main prediction endpoint
- **Input**: JSON object with car features
  ```json
  {
    "year": 2020,
    "km_driven": 50000,
    "fuel": "Petrol",
    ...
  }
  ```
- **Process**:
  1. Loads model and metadata
  2. Validates expected features from metadata
  3. Creates DataFrame with single row
  4. Runs prediction through pipeline
  5. Returns predicted price
- **Output**: 
  ```json
  {
    "price": 450000.50
  }
  ```
- **Error Handling**: 
  - 500 if model loading fails
  - 400 if prediction fails (invalid input)

**Technical Details:**
- Uses `Path(__file__).resolve().parents[2]` to find project root
- Model is loaded once per request (could be optimized with caching)
- Handles missing features gracefully (fills with None)

---

### 3. `app/frontend/streamlit_app.py` - Streamlit Web Interface

**Purpose**: User-friendly web interface for interacting with the prediction API.

**Key Features:**

#### Page Configuration
- Sets page title and icon
- Displays API URL being used

#### Metadata Loading
- Reads `metadata.json` to determine which features the model expects
- Separates features into numeric and categorical

#### Dynamic Form Generation
- **Numeric Features**: Creates `st.number_input` widgets
- **Categorical Features**: Creates `st.text_input` widgets
- Form is generated dynamically based on metadata

#### Prediction Flow
1. User fills form with car details
2. On submit, converts form data to JSON payload
3. Sends POST request to `/predict` endpoint
4. Displays predicted price or error message

#### Additional Features
- Expandable section showing expected features and their types
- Error handling for API connection issues
- Success/error messages with Streamlit's native components

**Environment Variables:**
- `API_URL`: URL of the backend API (default: 'http://localhost:8000')

---

### 4. `requirements.txt` - Python Dependencies

**Purpose**: Lists all Python packages needed for the project.

**Key Dependencies:**
- `fastapi==0.115.2`: Web framework for API
- `uvicorn==0.30.6`: ASGI server for FastAPI
- `scikit-learn==1.5.2`: ML library (RandomForest, preprocessing)
- `pandas==2.2.3`: Data manipulation
- `numpy==2.1.2`: Numerical computing
- `joblib==1.4.2`: Model serialization
- `streamlit==1.39.0`: Web UI framework
- `requests==2.32.3`: HTTP client (for frontend to call API)
- `pytest==8.3.3`: Testing framework
- `httpx==0.27.2`: HTTP client for testing

---

### 5. `Dockerfile` - Backend Container

**Purpose**: Defines the Docker image for the FastAPI backend.

**Structure:**
1. **Base Image**: `python:3.11-slim` (lightweight Python 3.11)
2. **Working Directory**: `/app`
3. **Environment Variables**: 
   - `PYTHONDONTWRITEBYTECODE=1`: Prevents .pyc files
   - `PYTHONUNBUFFERED=1`: Real-time logging
4. **Dependencies**: Installs from `requirements.txt`
5. **Code Copy**: Copies `app/` and `models/` directories
6. **Port**: Exposes port 8000
7. **Command**: Runs uvicorn server

**Note**: Assumes `models/` directory contains trained model files.

---

### 6. `Dockerfile.frontend` - Frontend Container

**Purpose**: Defines the Docker image for the Streamlit frontend.

**Structure:**
1. **Base Image**: `python:3.11-slim`
2. **Dependencies**: Only installs Streamlit, requests, and python-dotenv (not full requirements.txt)
3. **Code Copy**: Copies `app/frontend/` and `models/` (for metadata)
4. **Port**: Exposes port 8501 (Streamlit default)
5. **Environment**: Sets `API_URL=http://backend:8000` (Docker service name)
6. **Command**: Runs Streamlit server

---

### 7. `docker-compose.yml` - Container Orchestration

**Purpose**: Defines and orchestrates both backend and frontend containers.

**Services:**

#### `backend` Service
- Builds from `Dockerfile`
- Port mapping: `8000:8000`
- Volume mount: `./models:/app/models` (persists models)
- Restart policy: `unless-stopped`

#### `frontend` Service
- Builds from `Dockerfile.frontend`
- Port mapping: `8501:8501`
- Environment: `API_URL=http://backend:8000`
- Depends on: `backend` (waits for backend to start)
- Restart policy: `unless-stopped`

**Network**: Creates `carprice-net` for inter-container communication.

---

### 8. `tests/test_api.py` - API Tests

**Purpose**: Unit tests for API endpoints using pytest.

**Tests:**

#### `test_health()`
- Tests the `/health` endpoint
- Verifies status code 200
- Checks response contains 'status' key

#### `test_predict_minimal()`
- Tests the `/predict` endpoint
- Creates minimal payload (all features as None)
- Verifies status code 200
- Checks response contains 'price' key
- **Skip Condition**: Skips if metadata.json doesn't exist (model not trained)

**Testing Framework**: Uses FastAPI's `TestClient` for in-memory testing.

---

### 9. `models/metadata.json` - Model Metadata

**Purpose**: Stores information about the trained model for API and frontend to use.

**Structure:**
```json
{
  "target": "selling_price",           // Target column name
  "numeric_features": [...],            // List of numeric feature names
  "categorical_features": [...],       // List of categorical feature names
  "training_rows": 4848,               // Number of training samples
  "test_rows": 2078,                   // Number of test samples
  "metrics": {                         // Model performance metrics
    "rmse": 144563.78,
    "mae": 71390.44,
    "r2": 0.9097
  },
  "currency_rate": 1.0                 // Currency conversion rate used
}
```

**Usage:**
- API uses it to validate input features
- Frontend uses it to generate form fields dynamically
- Provides transparency about model requirements

---

### 10. `README.md` - Project Documentation

**Purpose**: User-facing documentation with setup and usage instructions.

**Contents:**
- Project overview and features
- Project structure
- Prerequisites
- Installation instructions
- Training commands
- Running instructions (local and Docker)
- Testing instructions
- API endpoint documentation

---

## 🔄 How the System Works

### Training Phase

1. **Data Preparation**:
   - User provides CSV file with car data
   - Script loads and validates data
   - Removes duplicates and rows with missing target values

2. **Feature Engineering**:
   - Automatically detects feature types (numeric vs categorical)
   - Creates preprocessing pipelines:
     - Numeric: Imputation → Standardization
     - Categorical: Imputation → OneHotEncoding

3. **Model Training**:
   - Splits data into train/test sets
   - Trains Random Forest Regressor (300 trees)
   - Evaluates on test set

4. **Model Persistence**:
   - Saves complete pipeline (preprocessing + model) as `.joblib`
   - Saves metadata as JSON

### Inference Phase

1. **User Input** (Frontend):
   - User fills form with car characteristics
   - Frontend reads metadata to know which fields to show

2. **API Request**:
   - Frontend sends POST request to `/predict` endpoint
   - Payload contains car features as JSON

3. **Prediction** (Backend):
   - API loads model and metadata
   - Validates input features
   - Creates DataFrame with single row
   - Runs through pipeline (preprocessing → prediction)
   - Returns predicted price

4. **Display** (Frontend):
   - Receives price from API
   - Displays to user with formatting

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.11+
- pip or conda
- Docker & Docker Compose (for containerized deployment)

### Option 1: Local Development

#### Step 1: Setup Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate          # Windows PowerShell
# or
source .venv/bin/activate         # Linux/Mac
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Train the Model
```bash
python pipeline/train.py \
  --csv "path/to/car-details.csv" \
  --target selling_price \
  --test-size 0.3 \
  --currency-rate 1.0 \
  --out-model models/rf_model.joblib \
  --out-meta models/metadata.json
```

**Expected Output:**
```
RMSE: 144563.7851
MAE: 71390.4369
R2: 0.9097
Model saved to: models/rf_model.joblib
Metadata saved to: models/metadata.json
```

#### Step 4: Start the API Server
```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`
- Health check: `http://localhost:8000/health`
- API docs: `http://localhost:8000/docs`

#### Step 5: Start the Frontend
Open a new terminal:
```bash
# Windows PowerShell
$env:API_URL="http://localhost:8000"
streamlit run app/frontend/streamlit_app.py

# Linux/Mac
export API_URL=http://localhost:8000
streamlit run app/frontend/streamlit_app.py
```

The frontend will be available at: `http://localhost:8501`

#### Step 6: Run Tests (Optional)
```bash
pytest tests/test_api.py -v
```

---

### Option 2: Docker Deployment

#### Step 1: Train the Model (Local)
First, train the model locally as described in Option 1, Step 3. This creates the `models/` directory with:
- `rf_model.joblib`
- `metadata.json`

#### Step 2: Build and Run with Docker Compose
```bash
docker compose up --build
```

This will:
- Build both backend and frontend images
- Start both containers
- Create a network for communication
- Mount the `models/` directory

#### Step 3: Access the Application
- **API**: `http://localhost:8000`
- **Frontend**: `http://localhost:8501`

#### Step 4: Stop the Containers
```bash
docker compose down
```

---

### Option 3: Individual Docker Containers

#### Build Backend
```bash
docker build -t carprice-backend -f Dockerfile .
docker run -p 8000:8000 -v ./models:/app/models carprice-backend
```

#### Build Frontend
```bash
docker build -t carprice-frontend -f Dockerfile.frontend .
docker run -p 8501:8501 -e API_URL=http://localhost:8000 carprice-frontend
```

---

## 🔧 Technical Details

### Machine Learning Pipeline

**Model**: Random Forest Regressor
- **n_estimators**: 300 trees
- **random_state**: 42 (for reproducibility)
- **n_jobs**: -1 (uses all CPU cores)

**Preprocessing**:
- **Numerical Features**:
  - Missing values → Median imputation
  - Values → StandardScaler (mean=0, std=1)
- **Categorical Features**:
  - Missing values → Most frequent imputation
  - Categories → OneHotEncoding (binary vectors)

**Evaluation Metrics**:
- **RMSE** (Root Mean Squared Error): Average prediction error
- **MAE** (Mean Absolute Error): Average absolute error
- **R²** (R-squared): Proportion of variance explained (0-1, higher is better)

### API Design

**Architecture**: RESTful API
- **Framework**: FastAPI (async-capable, auto-documentation)
- **CORS**: Enabled for all origins (development)
- **Error Handling**: HTTP status codes (400, 500)

**Endpoints**:
- `GET /health`: Health check (no authentication)
- `POST /predict`: Prediction endpoint (requires JSON body)

### Frontend Design

**Framework**: Streamlit
- **Layout**: Centered, single-page application
- **Form Handling**: Streamlit forms with submit button
- **API Communication**: `requests` library
- **Error Handling**: User-friendly error messages

### Data Flow Diagram

```
┌─────────────┐
│  CSV Data   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  train.py   │ ──► Feature Detection
└──────┬──────┘     Preprocessing Pipeline
       │            Model Training
       ▼            Evaluation
┌─────────────┐
│   Model +   │
│  Metadata   │
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   API       │ │  Frontend   │ │   Tests     │
│  (FastAPI)  │ │ (Streamlit) │ │  (pytest)   │
└──────┬──────┘ └──────┬───────┘ └─────────────┘
       │               │
       └───────┬───────┘
               │
               ▼
        ┌─────────────┐
        │   User      │
        └─────────────┘
```

### Security Considerations

**Current State** (Development):
- CORS allows all origins (`allow_origins=['*']`)
- No authentication/authorization
- No input validation beyond feature presence

**Production Recommendations**:
- Restrict CORS to specific domains
- Add authentication (API keys, JWT tokens)
- Implement rate limiting
- Add input validation (value ranges, data types)
- Use HTTPS
- Add logging and monitoring

### Performance Considerations

**Model Loading**:
- Currently loads model on each request (could be optimized with caching)
- Consider using `@lru_cache` or loading at startup

**Scalability**:
- FastAPI supports async operations
- Can be scaled horizontally with load balancer
- Model inference is CPU-bound (consider GPU for larger models)

### File Dependencies

```
train.py
  └─► Requires: CSV file
  └─► Produces: rf_model.joblib, metadata.json

main.py (API)
  └─► Requires: rf_model.joblib, metadata.json
  └─► Produces: JSON responses

streamlit_app.py (Frontend)
  └─► Requires: metadata.json (optional, for form generation)
  └─► Requires: API running
  └─► Produces: User interface

docker-compose.yml
  └─► Requires: All above files
  └─► Requires: models/ directory with trained model
```

---

## 📊 Example Usage

### Training Example
```bash
python pipeline/train.py \
  --csv car-details.csv \
  --target selling_price \
  --test-size 0.3
```

### API Request Example
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "year": 2020,
    "km_driven": 50000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage_mpg": 20.5,
    "engine_cc": 1500,
    "max_power_bhp": 120,
    "torque_nm": 200,
    "seats": 5
  }'
```

**Response:**
```json
{
  "price": 450000.50
}
```

---

## 🎓 Learning Points

This project demonstrates:
1. **End-to-end ML pipeline**: From raw data to deployed model
2. **Production-ready architecture**: Separation of concerns (training, API, UI)
3. **Containerization**: Docker for reproducible deployments
4. **API design**: RESTful principles with FastAPI
5. **User experience**: Streamlit for rapid UI development
6. **Testing**: Unit tests for API endpoints
7. **Documentation**: Comprehensive README and code comments

---

## 🔍 Troubleshooting

### Common Issues

1. **Model not found error**:
   - Solution: Train the model first using `train.py`

2. **Port already in use**:
   - Solution: Change port in docker-compose.yml or use different ports

3. **CORS errors in frontend**:
   - Solution: Ensure API_URL is correct and API is running

4. **Missing features error**:
   - Solution: Ensure input JSON contains all features listed in metadata.json

5. **Docker build fails**:
   - Solution: Ensure all files are in correct locations, check Dockerfile paths

---

## 📝 Conclusion

CarPriceML is a well-structured, production-ready ML application that demonstrates best practices in:
- Machine Learning pipeline development
- API design and deployment
- Frontend development
- Containerization
- Testing

The project is modular, maintainable, and can be easily extended with additional features like:
- Model versioning
- A/B testing
- Monitoring and logging
- User authentication
- Database integration
- Batch prediction endpoints

---

**Report Generated**: 2024
**Project Version**: 1.0.0
**Author**: Project Analysis

