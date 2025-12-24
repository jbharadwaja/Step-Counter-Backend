# 🏃‍♂️ AI-Powered Smart Step Counter

A full-stack iOS application that uses Machine Learning to provide context-aware calorie tracking. Unlike standard pedometers that use static math, this app connects to a cloud-based Random Forest model to analyze activity based on user biometrics and time-of-day intensity.

## 🏗️ Architecture

```mermaid
graph LR
    subgraph Client [iOS Client]
        A[SwiftUI Interface] -->|Fetches Data| B(HealthKit)
        B -->|Aggregated Steps| A
        A -->|POST JSON Request| C[Cloud API]
    end

    subgraph Server [Render Cloud]
        C -->|FastAPI| D[Python Backend]
        D -->|Input Vector| E{Random Forest Model}
        E -->|Prediction| D
        D -->|JSON Response| A
    end
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
