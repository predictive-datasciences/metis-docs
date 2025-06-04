# Python Coding Guidelines

> Comprehensive Python coding standards for the Metis platform, focusing on ML services, FastAPI development, and data processing components.

## 📋 Overview

These guidelines ensure consistent, maintainable, and performant Python code across our ML services, data processing pipelines, and API endpoints.

**Key Principles:**
- **Readability**: Code should be self-documenting
- **Type Safety**: Use type hints extensively
- **Performance**: Optimize for ML workloads
- **Maintainability**: Follow established patterns

---

## 🐍 Python Version & Environment

### **Python Version**
- **Minimum**: Python 3.11+
- **Recommended**: Python 3.12+
- **Virtual Environment**: Always use virtual environments

### **Dependency Management**
```toml
# pyproject.toml
[tool.poetry]
name = "metis-ml"
version = "1.0.0"
description = "Metis ML Services"
authors = ["Metis Team <team@metis.com>"]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
pydantic = "^2.5.0"
numpy = "^1.25.0"
pandas = "^2.1.0"
scikit-learn = "^1.3.0"
```

---

## 📁 Code Organization

### **Project Structure**
```
cmd/ml/
├── main.py                 # FastAPI application entry point
├── ml_server/             # ML server modules
│   ├── __init__.py
│   ├── api/               # API endpoints
│   │   ├── __init__.py
│   │   ├── ocr.py
│   │   └── risk.py
│   ├── models/            # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── ocr_model.py
│   │   └── risk_model.py
│   ├── services/          # Business logic
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   └── feature_extractor.py
│   └── utils/             # Utilities
│       ├── __init__.py
│       ├── config.py
│       └── logging.py
└── tests/                 # Test files
    ├── __init__.py
    ├── test_api/
    ├── test_models/
    └── test_services/
```

### **File Organization**
```python
"""Module docstring describing the purpose and functionality."""

# Standard library imports
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local imports
from ml_server.models.base import BaseModel
from ml_server.utils.config import settings
from ml_server.utils.logging import get_logger

# Constants
DEFAULT_TIMEOUT = 30
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Module-level variables
logger = get_logger(__name__)

# Classes and functions follow...
```

---

## 🏷️ Naming Conventions

### **General Rules**
- **snake_case**: Variables, functions, modules
- **PascalCase**: Classes, exceptions
- **UPPER_CASE**: Constants
- **_private**: Private attributes/methods

### **Specific Examples**
```python
# Variables and functions
user_id = "user-123"
document_path = Path("/path/to/document")

def process_document(file_path: Path) -> Dict[str, Any]:
    """Process document and extract data."""
    pass

def calculate_risk_score(features: np.ndarray) -> float:
    """Calculate risk score from features."""
    pass

# Classes
class DocumentProcessor:
    """Handles document processing operations."""

    def __init__(self, model_path: Path):
        self._model_path = model_path
        self._model = None

    def _load_model(self) -> None:
        """Load the ML model (private method)."""
        pass

class OCRModel(BaseModel):
    """OCR model for text extraction."""
    pass

# Constants
MAX_RETRY_COUNT = 3
DEFAULT_MODEL_PATH = Path("models/ocr/latest.pkl")
SUPPORTED_FORMATS = ["pdf", "jpg", "png"]

# Exceptions
class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass

class InvalidDocumentError(ValueError):
    """Raised when document format is invalid."""
    pass
```

---

## 📝 Type Hints & Documentation

### **Type Hints**
```python
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd

# Function signatures
def extract_features(
    document: Path,
    model_type: str = "default",
    confidence_threshold: float = 0.8
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Extract features from document.

    Args:
        document: Path to the document file
        model_type: Type of model to use for extraction
        confidence_threshold: Minimum confidence for feature extraction

    Returns:
        Tuple of (features array, confidence scores)

    Raises:
        InvalidDocumentError: If document format is not supported
        ModelLoadError: If model fails to load
    """
    pass

# Class with type hints
class RiskModel:
    """Risk assessment model."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: Optional[Any] = None
        self._features: List[str] = []

    def predict(self, features: np.ndarray) -> Dict[str, Union[float, str]]:
        """Predict risk score."""
        pass
```

### **Docstring Standards**
```python
def process_loan_application(
    application_data: Dict[str, Any],
    model_version: str = "latest"
) -> Dict[str, Any]:
    """Process loan application through ML pipeline.

    This function orchestrates the complete loan processing workflow,
    including document validation, feature extraction, and risk assessment.

    Args:
        application_data: Dictionary containing application information
            - applicant_id (str): Unique applicant identifier
            - documents (List[Path]): List of document file paths
            - amount (float): Requested loan amount
        model_version: Version of the model to use for processing

    Returns:
        Dictionary containing processing results:
            - risk_score (float): Calculated risk score (0.0 to 1.0)
            - decision (str): Approval decision ("approved", "rejected", "review")
            - confidence (float): Model confidence in the decision
            - processing_time (float): Time taken for processing in seconds

    Raises:
        InvalidDocumentError: If any document is in unsupported format
        ModelLoadError: If the specified model version cannot be loaded
        ProcessingError: If processing fails due to data issues

    Example:
        >>> application = {
        ...     "applicant_id": "app-123",
        ...     "documents": [Path("pan.jpg"), Path("aadhar.pdf")],
        ...     "amount": 50000.0
        ... }
        >>> result = process_loan_application(application)
        >>> print(result["decision"])
        "approved"
    """
    pass
```

---

## 🚀 FastAPI Development

### **Application Structure**
```python
# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from ml_server.api import ocr, risk
from ml_server.utils.config import settings
from ml_server.utils.logging import setup_logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    setup_logging()
    logger.info("Starting ML API server")

    # Load models
    await load_models()

    yield

    # Shutdown
    logger.info("Shutting down ML API server")
    await cleanup_models()

app = FastAPI(
    title="Metis ML API",
    description="Machine Learning services for the Metis platform",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["OCR"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk"])

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
```

### **API Endpoints**
```python
# ml_server/api/ocr.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import asyncio

from ml_server.models.ocr_model import OCRModel
from ml_server.services.document_processor import DocumentProcessor
from ml_server.utils.dependencies import get_ocr_model, get_document_processor

router = APIRouter()

class OCRRequest(BaseModel):
    """OCR processing request."""
    document_type: str = Field(..., description="Type of document (pan, aadhar, etc.)")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)
    extract_fields: bool = Field(True, description="Whether to extract specific fields")

class OCRResponse(BaseModel):
    """OCR processing response."""
    extracted_text: str
    confidence: float
    fields: Optional[Dict[str, str]] = None
    processing_time: float

@router.post("/process", response_model=OCRResponse)
async def process_document(
    file: UploadFile = File(...),
    request: OCRRequest = Depends(),
    ocr_model: OCRModel = Depends(get_ocr_model),
    processor: DocumentProcessor = Depends(get_document_processor)
) -> OCRResponse:
    """Process document through OCR pipeline."""

    # Validate file
    if not file.content_type.startswith(('image/', 'application/pdf')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Only images and PDFs are allowed."
        )

    try:
        # Process document
        start_time = asyncio.get_event_loop().time()

        result = await processor.process_document(
            file=file,
            document_type=request.document_type,
            confidence_threshold=request.confidence_threshold
        )

        processing_time = asyncio.get_event_loop().time() - start_time

        return OCRResponse(
            extracted_text=result.text,
            confidence=result.confidence,
            fields=result.fields if request.extract_fields else None,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise HTTPException(status_code=500, detail="OCR processing failed")
```

---

## 🧠 ML Model Development

### **Base Model Class**
```python
# ml_server/models/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional
import pickle
import joblib
import numpy as np

class BaseModel(ABC):
    """Base class for all ML models."""

    def __init__(self, model_path: Path, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self._model: Optional[Any] = None
        self._is_loaded = False

    @abstractmethod
    def preprocess(self, data: Any) -> np.ndarray:
        """Preprocess input data for model."""
        pass

    @abstractmethod
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions using the model."""
        pass

    @abstractmethod
    def postprocess(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Postprocess model predictions."""
        pass

    def load_model(self) -> None:
        """Load the ML model from disk."""
        if self._is_loaded:
            return

        try:
            if self.model_path.suffix == '.pkl':
                with open(self.model_path, 'rb') as f:
                    self._model = pickle.load(f)
            elif self.model_path.suffix == '.joblib':
                self._model = joblib.load(self.model_path)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path.suffix}")

            self._is_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
```

### **Specific Model Implementation**
```python
# ml_server/models/risk_model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path

from .base import BaseModel

class RiskModel(BaseModel):
    """Risk assessment model for loan applications."""

    def __init__(self, model_path: Path, feature_config_path: Path):
        super().__init__(model_path, "risk_model")
        self.feature_config_path = feature_config_path
        self._feature_names: List[str] = []
        self._feature_config: Dict[str, Any] = {}

    def load_model(self) -> None:
        """Load model and feature configuration."""
        super().load_model()

        # Load feature configuration
        with open(self.feature_config_path, 'r') as f:
            self._feature_config = json.load(f)

        self._feature_names = self._feature_config.get('features', [])

    def preprocess(self, application_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess application data into features."""
        features = []

        for feature_name in self._feature_names:
            feature_config = self._feature_config['feature_definitions'][feature_name]

            if feature_config['type'] == 'numerical':
                value = application_data.get(feature_name, 0.0)
                # Apply scaling if configured
                if 'scaling' in feature_config:
                    value = self._apply_scaling(value, feature_config['scaling'])
                features.append(value)

            elif feature_config['type'] == 'categorical':
                value = application_data.get(feature_name, 'unknown')
                # One-hot encoding
                encoded = self._encode_categorical(value, feature_config['categories'])
                features.extend(encoded)

        return np.array(features).reshape(1, -1)

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Predict risk score and category."""
        if not self._is_loaded:
            self.load_model()

        # Get prediction
        risk_score = self._model.predict_proba(features)[0][1]  # Probability of default
        risk_category = self._categorize_risk(risk_score)

        # Get feature importance for explanation
        feature_importance = self._get_feature_importance(features)

        return {
            'risk_score': float(risk_score),
            'risk_category': risk_category,
            'confidence': self._calculate_confidence(features),
            'feature_importance': feature_importance,
            'model_version': self._get_model_version()
        }

    def postprocess(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Postprocess predictions (already handled in predict method)."""
        return predictions

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk score into risk levels."""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.7:
            return "medium"
        else:
            return "high"

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate model confidence in prediction."""
        # Implementation depends on model type
        probabilities = self._model.predict_proba(features)[0]
        return float(max(probabilities))

    def _get_feature_importance(self, features: np.ndarray) -> Dict[str, float]:
        """Get feature importance for explanation."""
        if hasattr(self._model, 'feature_importances_'):
            importance = self._model.feature_importances_
            return dict(zip(self._feature_names, importance.tolist()))
        return {}
```

---

## 🔧 Error Handling

### **Custom Exceptions**
```python
# ml_server/utils/exceptions.py
class MetisMLError(Exception):
    """Base exception for Metis ML services."""
    pass

class ModelLoadError(MetisMLError):
    """Raised when model loading fails."""
    pass

class InvalidDocumentError(MetisMLError):
    """Raised when document is invalid or unsupported."""
    pass

class ProcessingError(MetisMLError):
    """Raised when processing fails."""
    pass

class FeatureExtractionError(MetisMLError):
    """Raised when feature extraction fails."""
    pass
```

### **Error Handling Patterns**
```python
import logging
from typing import Optional
from fastapi import HTTPException

logger = logging.getLogger(__name__)

async def safe_process_document(
    file_path: Path,
    processor: DocumentProcessor
) -> Optional[Dict[str, Any]]:
    """Safely process document with error handling."""
    try:
        result = await processor.process(file_path)
        return result

    except InvalidDocumentError as e:
        logger.warning(f"Invalid document {file_path}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except ModelLoadError as e:
        logger.error(f"Model loading failed: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")

    except ProcessingError as e:
        logger.error(f"Processing failed for {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Retry decorator
import asyncio
from functools import wraps

def retry_async(max_attempts: int = 3, delay: float = 1.0):
    """Retry decorator for async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (2 ** attempt))
                        logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    else:
                        logger.error(f"All {max_attempts} attempts failed: {e}")

            raise last_exception
        return wrapper
    return decorator
```

---

## 🧪 Testing

### **Test Structure**
```python
# tests/test_models/test_risk_model.py
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from ml_server.models.risk_model import RiskModel
from ml_server.utils.exceptions import ModelLoadError

class TestRiskModel:
    """Test suite for RiskModel."""

    @pytest.fixture
    def sample_model_path(self, tmp_path):
        """Create a sample model file."""
        model_path = tmp_path / "risk_model.pkl"
        # Create mock model file
        return model_path

    @pytest.fixture
    def sample_feature_config(self, tmp_path):
        """Create sample feature configuration."""
        config_path = tmp_path / "features.json"
        config = {
            "features": ["income", "age", "employment_type"],
            "feature_definitions": {
                "income": {"type": "numerical", "scaling": {"method": "standard"}},
                "age": {"type": "numerical"},
                "employment_type": {"type": "categorical", "categories": ["salaried", "self_employed"]}
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f)
        return config_path

    def test_model_initialization(self, sample_model_path, sample_feature_config):
        """Test model initialization."""
        model = RiskModel(sample_model_path, sample_feature_config)
        assert model.model_path == sample_model_path
        assert model.model_name == "risk_model"
        assert not model.is_loaded()

    @patch('ml_server.models.risk_model.joblib.load')
    def test_model_loading_success(self, mock_load, sample_model_path, sample_feature_config):
        """Test successful model loading."""
        mock_model = Mock()
        mock_load.return_value = mock_model

        model = RiskModel(sample_model_path, sample_feature_config)
        model.load_model()

        assert model.is_loaded()
        mock_load.assert_called_once()

    def test_model_loading_failure(self, sample_feature_config, tmp_path):
        """Test model loading failure."""
        nonexistent_path = tmp_path / "nonexistent.pkl"
        model = RiskModel(nonexistent_path, sample_feature_config)

        with pytest.raises(ModelLoadError):
            model.load_model()

    def test_preprocess_valid_data(self, sample_model_path, sample_feature_config):
        """Test preprocessing with valid data."""
        model = RiskModel(sample_model_path, sample_feature_config)

        application_data = {
            "income": 50000.0,
            "age": 30,
            "employment_type": "salaried"
        }

        features = model.preprocess(application_data)
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1  # Single sample

    @pytest.mark.asyncio
    async def test_predict_integration(self, sample_model_path, sample_feature_config):
        """Integration test for prediction."""
        with patch('ml_server.models.risk_model.joblib.load') as mock_load:
            # Mock model with predict_proba method
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])
            mock_load.return_value = mock_model

            model = RiskModel(sample_model_path, sample_feature_config)

            application_data = {
                "income": 50000.0,
                "age": 30,
                "employment_type": "salaried"
            }

            features = model.preprocess(application_data)
            result = model.predict(features)

            assert "risk_score" in result
            assert "risk_category" in result
            assert "confidence" in result
            assert 0.0 <= result["risk_score"] <= 1.0
```

### **Test Configuration**
```python
# tests/conftest.py
import pytest
import asyncio
from pathlib import Path
from fastapi.testclient import TestClient

from ml_server.main import app

@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)

@pytest.fixture
def sample_document(tmp_path):
    """Create a sample document for testing."""
    doc_path = tmp_path / "sample.pdf"
    doc_path.write_bytes(b"sample pdf content")
    return doc_path

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

---

## ⚡ Performance Optimization

### **Async Programming**
```python
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from typing import List

# Async file operations
async def read_document_async(file_path: Path) -> bytes:
    """Read document asynchronously."""
    async with aiofiles.open(file_path, 'rb') as f:
        return await f.read()

# CPU-bound operations in thread pool
executor = ThreadPoolExecutor(max_workers=4)

async def process_documents_parallel(documents: List[Path]) -> List[Dict[str, Any]]:
    """Process multiple documents in parallel."""
    tasks = []

    for doc_path in documents:
        task = asyncio.create_task(process_single_document(doc_path))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {documents[i]}: {result}")
        else:
            processed_results.append(result)

    return processed_results

async def cpu_intensive_task(data: np.ndarray) -> np.ndarray:
    """Run CPU-intensive task in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, _cpu_intensive_operation, data)

def _cpu_intensive_operation(data: np.ndarray) -> np.ndarray:
    """CPU-intensive operation that runs in thread pool."""
    # Heavy computation here
    return np.dot(data, data.T)
```

### **Memory Management**
```python
import gc
from contextlib import contextmanager

@contextmanager
def memory_cleanup():
    """Context manager for memory cleanup."""
    try:
        yield
    finally:
        gc.collect()

# Efficient data processing
def process_large_dataset(data_path: Path, batch_size: int = 1000) -> None:
    """Process large dataset in batches."""
    with memory_cleanup():
        for chunk in pd.read_csv(data_path, chunksize=batch_size):
            # Process chunk
            processed_chunk = process_chunk(chunk)

            # Save results
            save_processed_chunk(processed_chunk)

            # Clear memory
            del chunk, processed_chunk
            gc.collect()
```

---

## 📊 Logging & Monitoring

### **Structured Logging**
```python
# ml_server/utils/logging.py
import logging
import json
from typing import Any, Dict
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry)

def get_logger(name: str) -> logging.Logger:
    """Get configured logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger

# Usage
logger = get_logger(__name__)

def process_with_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process data with structured logging."""
    logger.info(
        "Starting data processing",
        extra={'extra_fields': {'data_size': len(data), 'operation': 'process'}}
    )

    try:
        result = process_data(data)
        logger.info(
            "Data processing completed successfully",
            extra={'extra_fields': {'result_size': len(result)}}
        )
        return result

    except Exception as e:
        logger.error(
            "Data processing failed",
            extra={'extra_fields': {'error': str(e), 'data_keys': list(data.keys())}}
        )
        raise
```

---

## 🔍 Code Quality Tools

### **Linting Configuration**
```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

[tool.pylint.messages_control]
disable = "C0330, C0326"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=ml_server --cov-report=html"
```

### **Pre-commit Hooks**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
```

---

*These guidelines ensure consistent, maintainable, and performant Python code across the Metis ML services and data processing components.*
