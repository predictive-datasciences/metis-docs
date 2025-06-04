# Metis Platform Architecture - Part 3: Data Processing

> **Part 3 of 3**: This document focuses on **data processing architecture** within the Metis platform. For complete context, refer to [Introduction to Metis](architecture-overview.md), [Data Flow](architecture-part-1.md), and [Data Storage](architecture-part-2.md).

## ⚙️ Introduction

Data processing is the intelligence layer of the Metis platform, transforming raw data into actionable insights through machine learning models, business logic, and workflow orchestration. This document covers ML pipelines, processing patterns, and orchestration strategies that power our platform's decision-making capabilities.

Our processing architecture enables:
- **Real-time ML inference**
- **Complex workflow orchestration**
- **Scalable business logic processing**
- **Event-driven data transformation**

---

## 🏗️ Processing Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   METIS PROCESSING ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────┤
│  🧠 ML PROCESSING LAYER                                        │
│  ├─ FastAPI ML Server (Python)                                 │
│  ├─ Common Models (OCR, Classification)                        │
│  └─ Client-Specific Models (Risk, Fraud)                       │
├─────────────────────────────────────────────────────────────────┤
│  🔄 WORKFLOW ORCHESTRATION                                     │
│  ├─ Temporal Workflows                                         │
│  ├─ Business Process Management                                │
│  └─ State Machine Coordination                                 │
├─────────────────────────────────────────────────────────────────┤
│  ⚡ BUSINESS LOGIC PROCESSING                                  │
│  ├─ Core App Services (Go)                                     │
│  ├─ Rules Engine                                               │
│  └─ Decision Making Logic                                      │
├─────────────────────────────────────────────────────────────────┤
│  📊 EVENT PROCESSING                                           │
│  ├─ Event Consumers                                            │
│  ├─ Stream Processing                                          │
│  └─ Analytics Pipeline                                         │
├─────────────────────────────────────────────────────────────────┤
│  🔗 EXTERNAL INTEGRATIONS                                      │
│  ├─ Third-party APIs                                           │
│  ├─ Bureau Services                                            │
│  └─ Notification Services                                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Processing Layer

### **FastAPI ML Server Architecture**

Our ML processing is handled by a dedicated Python FastAPI server that provides scalable model inference:

#### Server Structure:
```python
# cmd/ml/main.py
from fastapi import FastAPI, HTTPException
from pylibs.models import OCRModel, RiskModel
from pylibs.preprocessing import DocumentProcessor

app = FastAPI(title="Metis ML API", version="1.0.0")

# Model initialization
ocr_model = OCRModel()
risk_model = RiskModel()
processor = DocumentProcessor()

@app.post("/ocr/process")
async def process_document(file: UploadFile):
    """Process document through OCR pipeline"""
    processed_data = await processor.preprocess(file)
    result = ocr_model.predict(processed_data)
    return {"extracted_data": result}

@app.post("/risk/score")
async def calculate_risk_score(data: RiskInput):
    """Calculate risk score for given input"""
    features = processor.extract_features(data)
    score = risk_model.predict(features)
    return {"risk_score": score, "confidence": score.confidence}
```

### **Model Categories**

#### 1. **Common Models** 🔄

**OCR Engine**:
```python
class OCRModel:
    def __init__(self):
        self.model = load_pretrained_ocr_model()

    def extract_text(self, image_data):
        """Extract text from document images"""
        preprocessed = self.preprocess_image(image_data)
        text_data = self.model.predict(preprocessed)
        return self.postprocess_text(text_data)

    def extract_fields(self, document_type, text_data):
        """Extract specific fields based on document type"""
        field_extractor = self.get_field_extractor(document_type)
        return field_extractor.extract(text_data)
```

**Document Classification**:
```python
class DocumentClassifier:
    def classify_document(self, image_data):
        """Classify document type (PAN, Aadhar, Bank Statement)"""
        features = self.extract_visual_features(image_data)
        document_type = self.classifier.predict(features)
        confidence = self.classifier.predict_proba(features)
        return {
            "document_type": document_type,
            "confidence": confidence,
            "processing_pipeline": self.get_pipeline(document_type)
        }
```

#### 2. **Client-Specific Models** 🎯

**Risk Scoring Models**:
```python
class ClientRiskModel:
    def __init__(self, client_id):
        self.client_id = client_id
        self.model = self.load_client_model(client_id)
        self.features = self.load_feature_config(client_id)

    def calculate_risk_score(self, applicant_data):
        """Calculate client-specific risk score"""
        features = self.extract_features(applicant_data)
        risk_score = self.model.predict(features)

        return {
            "risk_score": risk_score,
            "risk_category": self.categorize_risk(risk_score),
            "contributing_factors": self.explain_prediction(features),
            "model_version": self.model.version
        }
```

### **Model Deployment & Versioning**

#### Model Registry:
```python
class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.versions = {}

    def register_model(self, model_name, model_version, model_path):
        """Register new model version"""
        model_key = f"{model_name}:{model_version}"
        self.models[model_key] = self.load_model(model_path)
        self.versions[model_name] = model_version

    def get_model(self, model_name, version=None):
        """Get model by name and version"""
        if version is None:
            version = self.versions[model_name]
        return self.models[f"{model_name}:{version}"]
```

---

## 🔄 Workflow Orchestration with Temporal

### **Workflow Architecture**

Our platform uses Temporal for complex business process orchestration:

#### Workflow Definition:
```go
// pkg/workflows/loan_processing.go
func LoanProcessingWorkflow(ctx workflow.Context, input LoanApplication) error {
    logger := workflow.GetLogger(ctx)

    // Step 1: Document Processing
    var docResult DocumentProcessingResult
    err := workflow.ExecuteActivity(ctx, ProcessDocumentsActivity, input.Documents).Get(ctx, &docResult)
    if err != nil {
        return fmt.Errorf("document processing failed: %w", err)
    }

    // Step 2: Bureau Data Pull
    var bureauResult BureauDataResult
    err = workflow.ExecuteActivity(ctx, PullBureauDataActivity, input.ApplicantID).Get(ctx, &bureauResult)
    if err != nil {
        return fmt.Errorf("bureau data pull failed: %w", err)
    }

    // Step 3: Risk Assessment
    riskInput := RiskAssessmentInput{
        Documents: docResult,
        BureauData: bureauResult,
        Application: input,
    }

    var riskResult RiskAssessmentResult
    err = workflow.ExecuteActivity(ctx, RiskAssessmentActivity, riskInput).Get(ctx, &riskResult)
    if err != nil {
        return fmt.Errorf("risk assessment failed: %w", err)
    }

    // Step 4: Decision Making
    decision := makeDecision(riskResult)

    // Step 5: Notification
    return workflow.ExecuteActivity(ctx, NotifyApplicantActivity, decision).Get(ctx, nil)
}
```

#### Activity Implementation:
```go
func ProcessDocumentsActivity(ctx context.Context, documents []Document) (DocumentProcessingResult, error) {
    var result DocumentProcessingResult

    for _, doc := range documents {
        // Call ML service for OCR processing
        ocrResult, err := callMLService("/ocr/process", doc)
        if err != nil {
            return result, fmt.Errorf("OCR processing failed: %w", err)
        }

        // Validate extracted data
        validationResult := validateExtractedData(ocrResult)

        result.ProcessedDocuments = append(result.ProcessedDocuments, ProcessedDocument{
            OriginalDocument: doc,
            ExtractedData:    ocrResult,
            ValidationResult: validationResult,
        })
    }

    return result, nil
}
```

### **Workflow Patterns**

#### 1. **Sequential Processing**:
```go
// Linear workflow for simple processes
func SimpleApprovalWorkflow(ctx workflow.Context, application Application) error {
    // Step 1 -> Step 2 -> Step 3 -> Decision
    return executeSequentialSteps(ctx, application)
}
```

#### 2. **Parallel Processing**:
```go
// Parallel execution for independent tasks
func ParallelProcessingWorkflow(ctx workflow.Context, application Application) error {
    // Execute multiple activities in parallel
    futures := []workflow.Future{
        workflow.ExecuteActivity(ctx, ProcessDocuments, application.Documents),
        workflow.ExecuteActivity(ctx, PullBureauData, application.ApplicantID),
        workflow.ExecuteActivity(ctx, ValidateApplication, application),
    }

    // Wait for all to complete
    for _, future := range futures {
        if err := future.Get(ctx, nil); err != nil {
            return err
        }
    }

    return nil
}
```

#### 3. **Conditional Workflows**:
```go
// Conditional execution based on business rules
func ConditionalWorkflow(ctx workflow.Context, application Application) error {
    if application.Amount > 1000000 {
        return executeHighValueWorkflow(ctx, application)
    }
    return executeStandardWorkflow(ctx, application)
}
```

---

## ⚡ Business Logic Processing

### **Rules Engine Architecture**

Our rules engine provides flexible, configurable business logic:

#### Rule Definition:
```go
type Rule struct {
    ID          string                 `json:"id"`
    Name        string                 `json:"name"`
    Conditions  []Condition            `json:"conditions"`
    Actions     []Action               `json:"actions"`
    Priority    int                    `json:"priority"`
    TenantID    string                 `json:"tenant_id"`
}

type Condition struct {
    Field    string      `json:"field"`
    Operator string      `json:"operator"`
    Value    interface{} `json:"value"`
}

type Action struct {
    Type   string                 `json:"type"`
    Params map[string]interface{} `json:"params"`
}
```

#### Rules Engine Implementation:
```go
type RulesEngine struct {
    rules []Rule
}

func (re *RulesEngine) EvaluateRules(data map[string]interface{}) ([]Action, error) {
    var actions []Action

    // Sort rules by priority
    sort.Slice(re.rules, func(i, j int) bool {
        return re.rules[i].Priority > re.rules[j].Priority
    })

    for _, rule := range re.rules {
        if re.evaluateConditions(rule.Conditions, data) {
            actions = append(actions, rule.Actions...)
        }
    }

    return actions, nil
}

func (re *RulesEngine) evaluateConditions(conditions []Condition, data map[string]interface{}) bool {
    for _, condition := range conditions {
        if !re.evaluateCondition(condition, data) {
            return false
        }
    }
    return true
}
```

### **Decision Making Logic**

#### Credit Decision Engine:
```go
type CreditDecisionEngine struct {
    rulesEngine *RulesEngine
    riskModel   *RiskModel
}

func (cde *CreditDecisionEngine) MakeDecision(application LoanApplication, riskScore float64) Decision {
    // Combine rule-based and ML-based decisions
    ruleDecision := cde.evaluateRules(application, riskScore)
    mlDecision := cde.evaluateMLModel(application, riskScore)

    // Final decision logic
    finalDecision := cde.combineDecisions(ruleDecision, mlDecision)

    return Decision{
        Outcome:     finalDecision.Outcome,
        Confidence:  finalDecision.Confidence,
        Reasons:     finalDecision.Reasons,
        Conditions:  finalDecision.Conditions,
        Timestamp:   time.Now(),
    }
}
```

---

## 📊 Event Processing Pipeline

### **Event-Driven Architecture**

Our platform processes events in real-time for analytics and monitoring:

#### Event Consumer:
```go
type EventConsumer struct {
    queue      *rabbitmq.Queue
    processor  *EventProcessor
    clickhouse *clickhouse.Client
}

func (ec *EventConsumer) ProcessEvents() error {
    return ec.queue.Consume(func(event Event) error {
        // Process event
        processedEvent := ec.processor.Process(event)

        // Store in analytics database
        return ec.clickhouse.Insert("events", processedEvent)
    })
}
```

#### Stream Processing:
```go
type StreamProcessor struct {
    inputStream  chan Event
    outputStream chan ProcessedEvent
}

func (sp *StreamProcessor) ProcessStream() {
    for event := range sp.inputStream {
        // Real-time event processing
        processedEvent := sp.processEvent(event)

        // Trigger alerts if needed
        if sp.shouldAlert(processedEvent) {
            sp.triggerAlert(processedEvent)
        }

        sp.outputStream <- processedEvent
    }
}
```

---

## 🔗 External Integrations

### **Third-Party API Integration**

#### Bureau Integration:
```go
type BureauClient struct {
    httpClient *http.Client
    config     BureauConfig
}

func (bc *BureauClient) PullCreditReport(applicantID string) (*CreditReport, error) {
    // Implement retry logic with exponential backoff
    return retry.Do(func() (*CreditReport, error) {
        request := bc.buildRequest(applicantID)
        response, err := bc.httpClient.Do(request)
        if err != nil {
            return nil, err
        }

        return bc.parseResponse(response)
    }, retry.Attempts(3), retry.Delay(time.Second))
}
```

#### Notification Service:
```go
type NotificationService struct {
    emailClient *email.Client
    smsClient   *sms.Client
}

func (ns *NotificationService) SendDecisionNotification(applicant Applicant, decision Decision) error {
    // Send email notification
    emailErr := ns.emailClient.Send(email.Message{
        To:      applicant.Email,
        Subject: "Loan Application Decision",
        Body:    ns.buildEmailBody(decision),
    })

    // Send SMS notification
    smsErr := ns.smsClient.Send(sms.Message{
        To:   applicant.Phone,
        Body: ns.buildSMSBody(decision),
    })

    // Return combined errors
    return errors.Join(emailErr, smsErr)
}
```

---

## 📈 Performance Optimization

### **Processing Performance**

#### Async Processing:
```go
type AsyncProcessor struct {
    workerPool chan struct{}
    taskQueue  chan Task
}

func (ap *AsyncProcessor) ProcessAsync(task Task) {
    select {
    case ap.taskQueue <- task:
        // Task queued successfully
    default:
        // Queue full, handle overflow
        ap.handleOverflow(task)
    }
}

func (ap *AsyncProcessor) worker() {
    for {
        select {
        case <-ap.workerPool:
            task := <-ap.taskQueue
            ap.processTask(task)
            ap.workerPool <- struct{}{} // Return worker to pool
        }
    }
}
```

#### Caching Strategy:
```go
type ProcessingCache struct {
    cache *redis.Client
    ttl   time.Duration
}

func (pc *ProcessingCache) GetOrProcess(key string, processor func() (interface{}, error)) (interface{}, error) {
    // Try cache first
    if result, err := pc.cache.Get(key).Result(); err == nil {
        return result, nil
    }

    // Process and cache result
    result, err := processor()
    if err != nil {
        return nil, err
    }

    pc.cache.Set(key, result, pc.ttl)
    return result, nil
}
```

---

## 🔍 Monitoring & Observability

### **Processing Metrics**

#### Key Metrics:
```go
var (
    processingDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "processing_duration_seconds",
            Help: "Duration of processing operations",
        },
        []string{"operation", "tenant"},
    )

    processingErrors = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "processing_errors_total",
            Help: "Total number of processing errors",
        },
        []string{"operation", "error_type"},
    )
)
```

#### Workflow Monitoring:
```go
func MonitorWorkflow(ctx workflow.Context, workflowName string) {
    workflow.GetMetricsScope(ctx).Counter("workflow_started").Inc(1)

    defer func() {
        if workflow.IsReplaying(ctx) {
            return
        }
        workflow.GetMetricsScope(ctx).Counter("workflow_completed").Inc(1)
    }()
}
```

---

## ⚠️ Common Pitfalls & Solutions

### **1. ML Model Performance Degradation** 🧠

**Problem**: Model accuracy decreases over time
**Solution**:
- Implement model monitoring
- Regular model retraining
- A/B testing for model versions
- Performance drift detection

### **2. Workflow Timeout Issues** ⏰

**Problem**: Long-running workflows timing out
**Solution**:
- Implement heartbeat activities
- Break workflows into smaller chunks
- Use continue-as-new pattern
- Proper timeout configuration

### **3. External API Failures** 🔗

**Problem**: Third-party service unavailability
**Solution**:
- Implement circuit breakers
- Retry with exponential backoff
- Fallback mechanisms
- Service health monitoring

### **4. Processing Bottlenecks** ⚡

**Problem**: Processing queue backlog
**Solution**:
- Horizontal scaling of workers
- Load balancing strategies
- Queue monitoring and alerting
- Resource optimization

---

## 🎯 Future Enhancements

### **Immediate Improvements**
1. **Model Serving**: Implement model serving infrastructure
2. **Batch Processing**: Add batch processing capabilities
3. **Real-time Analytics**: Stream processing for real-time insights

### **Future Enhancements**
1. **AutoML**: Automated model training and deployment
2. **Edge Computing**: Edge-based model inference
3. **Federated Learning**: Privacy-preserving model training
4. **Advanced Orchestration**: Complex workflow patterns

---

*This completes the three-part architecture series. Together with [Introduction to Metis](architecture-overview.md), [Data Flow](architecture-part-1.md), and [Data Storage](architecture-part-2.md), this provides a comprehensive view of the Metis platform architecture.*
