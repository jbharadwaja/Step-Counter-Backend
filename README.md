# 🏃‍♂️ AI-Powered Smart Step Counter & Analytics Engine

A full-stack iOS application that uses Machine Learning to provide context-aware calorie tracking. Unlike standard pedometers, this app connects to a cloud-based Random Forest model to analyze activity based on user biometrics and time-of-day intensity, while maintaining a real-time history log of user activity.

## 🏗️ Architecture

```mermaid
graph LR
    subgraph Client [iOS Client]
        A[SwiftUI Interface] -->|Fetches Data| B(HealthKit)
        A -->|POST Request| C[Cloud API]
        A -->|Haptic Feedback| A
    end

    subgraph Server [Render Cloud]
        C -->|FastAPI| D[Python Backend]
        D -->|Input Vector| E{Random Forest Model}
        E -->|Prediction| D
        D -->|Log Data| F[(History Log)]
        D -->|JSON Response| A
    end
    
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#dfd,stroke:#333,stroke-width:2px
