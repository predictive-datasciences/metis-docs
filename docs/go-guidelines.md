# Go Coding Guidelines

> Comprehensive Go coding standards for the Metis platform, ensuring consistency, maintainability, and performance across the codebase.

## 📋 Overview

These guidelines reflect Go's best practices and are designed to prevent common anti-patterns while maintaining code quality across our multi-tenant SAAS platform.

**Key Principles:**
- **Simplicity**: Write clear, readable code
- **Consistency**: Follow established patterns
- **Performance**: Optimize for efficiency
- **Maintainability**: Code should be easy to modify

---

## 📁 Code Organization

### **File Structure**
- **Maximum file length**: 500 lines (excluding comments)
- **Single purpose**: Each file should have one well-defined responsibility
- **Related functionality**: Group related code together

### **File Element Order**
```go
// 1. Package declaration
package main

// 2. Imports (grouped and ordered)
import (
    // Standard library
    "context"
    "fmt"
    "time"

    // Third-party packages
    "github.com/gin-gonic/gin"
    "github.com/temporal-io/sdk-go/workflow"

    // Internal packages
    "github.com/metis/pkg/auth"
    "github.com/metis/pkg/database"
)

// 3. Constants
const (
    MaxRetryCount = 3
    DefaultTimeout = 30 * time.Second
)

// 4. Variables
var (
    ErrNotFound = errors.New("resource not found")
    ErrInvalidInput = errors.New("invalid input")
)

// 5. Types
type User struct {
    ID    string
    Email string
}

// 6. Functions and Methods
func NewUser(email string) *User {
    return &User{
        ID:    generateID(),
        Email: email,
    }
}
```

---

## 🏷️ Naming Conventions

### **General Rules**
- **Short, concise names** in small scopes
- **Descriptive names** in larger scopes
- **No underscores** in package names
- **CamelCase** for variables and functions
- **PascalCase** for exported names

### **Specific Conventions**

#### Packages
```go
// Good
package api
package config
package database

// Bad
package api_handlers
package db_utils
```

#### Functions
```go
// Good - Use verbs
func GetUser(ctx context.Context, id string) (*User, error)
func CreateOrder(order *Order) error
func ValidateInput(data map[string]interface{}) error

// Bad
func User(id string) (*User, error)
func Order(order *Order) error
```

#### Variables
```go
// Good - Use nouns
var user *User
var orderList []Order
var dbConnection *sql.DB

// Bad
var getUserResult *User
var createOrderFunc func()
```

#### Constants
```go
// Good
const (
    MaxRetryCount     = 3
    DefaultTimeout    = 30 * time.Second
    DatabaseURL       = "postgres://localhost:5432/metis"
)
```

#### Error Variables
```go
// Good - Prefix with Err
var (
    ErrNotFound      = errors.New("resource not found")
    ErrUnauthorized  = errors.New("unauthorized access")
    ErrInvalidInput  = errors.New("invalid input provided")
)
```

#### Interfaces
```go
// Good - Add 'er' suffix when appropriate
type Reader interface {
    Read([]byte) (int, error)
}

type UserRepository interface {
    GetUser(ctx context.Context, id string) (*User, error)
    CreateUser(ctx context.Context, user *User) error
}
```

---

## 📝 Documentation

### **Package Documentation**
```go
// Package auth provides authentication and authorization functionality
// for the Metis platform. It includes JWT token management, session
// handling, and multi-tenant security features.
package auth
```

### **Function Documentation**
```go
// GetUser retrieves a user by ID from the database.
// Returns ErrNotFound if the user doesn't exist.
// The context should include tenant information for proper isolation.
func GetUser(ctx context.Context, id string) (*User, error) {
    // Implementation
}
```

### **Type Documentation**
```go
// User represents a system user with authentication details.
// It includes fields for identification, tenant association,
// and access control within the multi-tenant architecture.
type User struct {
    ID       string    `json:"id" db:"id"`
    TenantID string    `json:"tenant_id" db:"tenant_id"`
    Email    string    `json:"email" db:"email"`
    Name     string    `json:"name" db:"name"`
    Created  time.Time `json:"created_at" db:"created_at"`
}
```

---

## 🔧 Function Design

### **Size and Complexity**
- **Maximum length**: 50 lines
- **Maximum nesting**: 3 levels of control structures
- **Single responsibility**: Functions should do one thing well

### **Parameters**
```go
// Good - Maximum 5 parameters
func ProcessLoan(ctx context.Context, userID, loanID string, amount float64, urgent bool) error

// Better - Use structs for more parameters
type LoanRequest struct {
    UserID   string
    LoanID   string
    Amount   float64
    Urgent   bool
    Metadata map[string]interface{}
}

func ProcessLoan(ctx context.Context, request LoanRequest) error
```

### **Return Values**
```go
// Good - Return errors as last value
func GetUser(ctx context.Context, id string) (*User, error)

// Good - Use named returns for documentation
func CalculateInterest(principal, rate float64, years int) (interest, total float64, err error) {
    if principal <= 0 || rate <= 0 || years <= 0 {
        err = ErrInvalidInput
        return
    }

    interest = principal * rate * float64(years) / 100
    total = principal + interest
    return
}
```

---

## ❌ Error Handling

### **Error Types**
```go
// Custom error types for domain-specific errors
type ValidationError struct {
    Field   string
    Message string
}

func (e ValidationError) Error() string {
    return fmt.Sprintf("validation failed for field %s: %s", e.Field, e.Message)
}

// Sentinel errors for common cases
var (
    ErrUserNotFound    = errors.New("user not found")
    ErrInvalidTenant   = errors.New("invalid tenant")
    ErrDuplicateEmail  = errors.New("email already exists")
)
```

### **Error Wrapping**
```go
func ProcessOrder(ctx context.Context, order *Order) error {
    if err := validateOrder(order); err != nil {
        return fmt.Errorf("invalid order: %w", err)
    }

    if err := saveOrder(ctx, order); err != nil {
        return fmt.Errorf("failed to save order %s: %w", order.ID, err)
    }

    return nil
}
```

### **Error Handling Patterns**
```go
// Good - Handle errors immediately
func GetUserProfile(ctx context.Context, userID string) (*UserProfile, error) {
    user, err := userRepo.GetUser(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }

    profile, err := profileRepo.GetProfile(ctx, userID)
    if err != nil {
        return nil, fmt.Errorf("failed to get profile: %w", err)
    }

    return &UserProfile{User: user, Profile: profile}, nil
}
```

---

## 🧪 Testing

### **Test Structure**
```go
func TestGetUser(t *testing.T) {
    tests := []struct {
        name    string
        userID  string
        want    *User
        wantErr error
    }{
        {
            name:    "valid user",
            userID:  "user-123",
            want:    &User{ID: "user-123", Email: "test@example.com"},
            wantErr: nil,
        },
        {
            name:    "user not found",
            userID:  "nonexistent",
            want:    nil,
            wantErr: ErrUserNotFound,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            got, err := GetUser(context.Background(), tt.userID)

            if !errors.Is(err, tt.wantErr) {
                t.Errorf("GetUser() error = %v, wantErr %v", err, tt.wantErr)
                return
            }

            if !reflect.DeepEqual(got, tt.want) {
                t.Errorf("GetUser() = %v, want %v", got, tt.want)
            }
        })
    }
}
```

### **Test Requirements**
- **Minimum coverage**: 80%
- **Test independence**: Tests should not depend on each other
- **Both paths**: Test success and failure scenarios
- **Descriptive names**: Test names should describe the scenario

---

## 📦 Package Design

### **Package Principles**
- **Single responsibility**: Each package should have one clear purpose
- **Minimal dependencies**: Reduce coupling between packages
- **Internal packages**: Use `internal/` for implementation details

### **Package Structure**
```
pkg/
├── auth/           # Authentication and authorization
├── database/       # Database operations and models
├── config/         # Configuration management
├── logging/        # Structured logging
├── workflows/      # Temporal workflow definitions
└── internal/       # Internal implementation details
    ├── models/     # Internal data models
    └── utils/      # Internal utilities
```

---

## 🔄 Concurrency

### **Goroutines**
```go
// Good - Proper goroutine lifecycle management
func ProcessBatch(ctx context.Context, items []Item) error {
    errChan := make(chan error, len(items))

    for _, item := range items {
        go func(i Item) {
            defer func() {
                if r := recover(); r != nil {
                    errChan <- fmt.Errorf("panic processing item %s: %v", i.ID, r)
                }
            }()

            errChan <- processItem(ctx, i)
        }(item)
    }

    for range items {
        if err := <-errChan; err != nil {
            return fmt.Errorf("batch processing failed: %w", err)
        }
    }

    return nil
}
```

### **Channels**
```go
// Good - Proper channel usage
func WorkerPool(ctx context.Context, jobs <-chan Job, results chan<- Result) {
    defer close(results)

    for {
        select {
        case job, ok := <-jobs:
            if !ok {
                return // Channel closed
            }

            result := processJob(job)

            select {
            case results <- result:
            case <-ctx.Done():
                return
            }

        case <-ctx.Done():
            return
        }
    }
}
```

---

## ⚡ Performance Considerations

### **Memory Management**
```go
// Good - Pre-allocate slices when size is known
func ProcessUsers(users []User) []ProcessedUser {
    processed := make([]ProcessedUser, 0, len(users))

    for _, user := range users {
        processed = append(processed, processUser(user))
    }

    return processed
}

// Good - Use sync.Pool for frequently allocated objects
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 0, 1024)
    },
}

func ProcessData(data []byte) ([]byte, error) {
    buf := bufferPool.Get().([]byte)
    defer bufferPool.Put(buf[:0])

    // Process data using buf
    return processWithBuffer(data, buf)
}
```

### **String Operations**
```go
// Good - Use strings.Builder for concatenation
func BuildQuery(fields []string, table string) string {
    var builder strings.Builder
    builder.WriteString("SELECT ")

    for i, field := range fields {
        if i > 0 {
            builder.WriteString(", ")
        }
        builder.WriteString(field)
    }

    builder.WriteString(" FROM ")
    builder.WriteString(table)

    return builder.String()
}
```

---

## 🚫 Anti-Patterns to Avoid

### **Global Variables**
```go
// Bad
var db *sql.DB

func init() {
    db = connectToDatabase()
}

// Good - Use dependency injection
type UserService struct {
    db *sql.DB
}

func NewUserService(db *sql.DB) *UserService {
    return &UserService{db: db}
}
```

### **Panic Usage**
```go
// Bad - Don't use panic for normal error handling
func GetUser(id string) *User {
    user, err := userRepo.Get(id)
    if err != nil {
        panic(err) // Bad!
    }
    return user
}

// Good - Return errors
func GetUser(ctx context.Context, id string) (*User, error) {
    user, err := userRepo.Get(ctx, id)
    if err != nil {
        return nil, fmt.Errorf("failed to get user: %w", err)
    }
    return user, nil
}
```

---

## 🔍 Code Review Checklist

### **Functionality**
- [ ] Code works as expected
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] Tests cover the functionality

### **Code Quality**
- [ ] Follows naming conventions
- [ ] Functions are appropriately sized
- [ ] Code is well-documented
- [ ] No code duplication

### **Performance**
- [ ] No obvious performance issues
- [ ] Appropriate data structures used
- [ ] Memory usage is reasonable
- [ ] Concurrency is handled correctly

### **Security**
- [ ] Input validation is present
- [ ] No sensitive data in logs
- [ ] Proper error messages (no information leakage)
- [ ] Authentication/authorization checks

---

## 🛠️ Tools & Linting

### **Required Tools**
- **golangci-lint**: Comprehensive linting
- **go fmt**: Code formatting
- **go vet**: Static analysis
- **staticcheck**: Advanced static analysis

### **Pre-commit Setup**
```bash
# Install pre-commit hooks
make pre-commit-install

# Run linting
make lint

# Run tests
make test
```

### **IDE Configuration**
Configure your IDE to:
- Run `go fmt` on save
- Show linting errors inline
- Run tests automatically
- Import organization

---

## 📊 Metrics & Monitoring

### **Code Metrics**
```go
// Good - Add metrics to important functions
func ProcessLoanApplication(ctx context.Context, app *LoanApplication) error {
    start := time.Now()
    defer func() {
        processingDuration.WithLabelValues("loan_application").Observe(time.Since(start).Seconds())
    }()

    // Processing logic
    return nil
}
```

### **Logging**
```go
// Good - Structured logging
func ProcessUser(ctx context.Context, userID string) error {
    logger := logging.FromContext(ctx).With(
        "user_id", userID,
        "operation", "process_user",
    )

    logger.Info("starting user processing")

    if err := validateUser(ctx, userID); err != nil {
        logger.Error("user validation failed", "error", err)
        return fmt.Errorf("validation failed: %w", err)
    }

    logger.Info("user processing completed successfully")
    return nil
}
```

---

*These guidelines ensure consistent, maintainable, and performant Go code across the Metis platform. Regular review and updates keep them aligned with evolving best practices.*
