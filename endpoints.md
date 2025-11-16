# API Specification

## Overview
This document provides detailed specifications for all API endpoints used in the AI Web application. The base URL for all endpoints is `http://localhost:8090/`.

All endpoints use cookie-based authentication (credentials: "include") except for the authentication endpoints.

---

## Authentication APIs

### 1. Login - Send OTP
**Endpoint:** `POST /auth/login`

**Description:** Initiates the login process by sending an OTP to the provided phone number.

**Authentication:** Not required

**Request Body:**
```json
{
  "phone_number": "string"
}
```

**Request Body Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| phone_number | string | Yes | User's phone number |

**Response Structure:**
```json
{
  "status": 200,
  "message": {
    "data": null
  }
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| status | number | HTTP status code |
| message.data | null | No data returned on success |

**Example Response:**
```json
{
  "status": 200,
  "message": {
    "data": null
  }
}
```

---

### 2. Verify OTP
**Endpoint:** `POST /auth/verify-otp`

**Description:** Verifies the OTP sent to the user's phone and returns authentication tokens along with user information.

**Authentication:** Not required

**Request Body:**
```json
{
  "phone_number": "string",
  "otp": "string"
}
```

**Request Body Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| phone_number | string | Yes | User's phone number |
| otp | string | Yes | One-time password received via SMS |

**Response Structure:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "access_token": "string",
      "refresh_token": "string",
      "user": {
        "id": "string",
        "phone_number": "string",
        "name": "string"
      }
    }
  }
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| status | number | HTTP status code |
| message.data.access_token | string | JWT access token for authentication |
| message.data.refresh_token | string | JWT refresh token for renewing access |
| message.data.user.id | string | Unique user identifier |
| message.data.user.phone_number | string | User's masked phone number |
| message.data.user.name | string | User's display name |

**Example Response:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "access_token": "mock-token-12345",
      "refresh_token": "mock-refresh-token-67890",
      "user": {
        "id": "1",
        "phone_number": "123******90",
        "name": "Mock User"
      }
    }
  }
}
```

---

## Profile APIs

### 3. Get Profiles
**Endpoint:** `GET /profiles`

**Description:** Retrieves all available profiles for the authenticated user (e.g., parent, student).

**Authentication:** Required (Cookie-based)

**Request Body:** None

**Response Structure:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "profiles": [
        {
          "profile_type": "string",
          "profile_name": "string"
        }
      ]
    }
  }
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| status | number | HTTP status code |
| message.data.profiles | array | Array of available profile objects |
| message.data.profiles[].profile_type | string | Type of profile (e.g., "parent", "student") |
| message.data.profiles[].profile_name | string | Display name for the profile |

**Example Response:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "profiles": [
        {
          "profile_type": "parent",
          "profile_name": "Parent"
        },
        {
          "profile_type": "student",
          "profile_name": "Student"
        }
      ]
    }
  }
}
```

---

## Dashboard APIs

### 4. Get Student Progress Dashboard
**Endpoint:** `GET /progress`

**Description:** Retrieves the student's progress dashboard including learning history, subject progress, and statistics.

**Authentication:** Required (Cookie-based)

**Request Body:** None

**Response Structure:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "progress": {
        "label": "string",
        "title": "string",
        "callout": "string",
        "subjects": [
          {
            "subject": "string",
            "totalChapters": "number (optional)",
            "completedChapters": "number (optional)"
          }
        ]
      },
      "history": {
        "label": "string",
        "resume": [
          {
            "subject": "string",
            "chapter": "string"
          }
        ]
      },
      "statistics": {
        "metrics": [
          {
            "line1": "string",
            "line2": "string"
          }
        ]
      }
    }
  }
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| status | number | HTTP status code |
| message.data.progress.label | string | Section label for progress |
| message.data.progress.title | string | Greeting/title text |
| message.data.progress.callout | string | Motivational message about recent activity |
| message.data.progress.subjects | array | Array of subjects with progress information |
| message.data.progress.subjects[].subject | string | Name of the subject |
| message.data.progress.subjects[].totalChapters | number (optional) | Total number of chapters in the subject |
| message.data.progress.subjects[].completedChapters | number (optional) | Number of completed chapters |
| message.data.history.label | string | Section label for history |
| message.data.history.resume | array | Array of recently accessed chapters |
| message.data.history.resume[].subject | string | Subject name |
| message.data.history.resume[].chapter | string | Chapter name |
| message.data.statistics.metrics | array | Array of statistical metrics |
| message.data.statistics.metrics[].line1 | string | Primary metric value |
| message.data.statistics.metrics[].line2 | string | Metric description |

**Example Response:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "progress": {
        "label": "My Progress",
        "title": "Hi Participant!",
        "callout": "You have completed 5 lessons this week!",
        "subjects": [
          {
            "subject": "Mathematics"
          },
          {
            "subject": "Science",
            "totalChapters": 10,
            "completedChapters": 5
          },
          {
            "subject": "History",
            "totalChapters": 12,
            "completedChapters": 8
          }
        ]
      },
      "history": {
        "label": "Continue where you left off",
        "resume": [
          {
            "subject": "Mathematics",
            "chapter": "Quadratic Equations"
          },
          {
            "subject": "Science",
            "chapter": "Newton's Laws of Motion"
          }
        ]
      },
      "statistics": {
        "metrics": [
          {
            "line1": "17",
            "line2": "Total chapters studied"
          },
          {
            "line1": "03",
            "line2": "Chapters reading"
          },
          {
            "line1": "12",
            "line2": "Days of study streak"
          },
          {
            "line1": "9.2",
            "line2": "hours spent this week"
          }
        ]
      }
    }
  }
}
```

---

### 5. Get Parent Dashboard
**Endpoint:** `GET /dashboard`

**Description:** Retrieves the parent's dashboard with information about all children, study hours, and weekly insights.

**Authentication:** Required (Cookie-based)

**Request Body:** None

**Response Structure:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "totalChildren": "number",
      "totalStudyHours": "number",
      "studyInsight": "string",
      "weeklyInsights": [
        {
          "name": "string",
          "subject": "string",
          "insight": "string"
        }
      ]
    }
  }
}
```

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| status | number | HTTP status code |
| message.data.totalChildren | number | Total number of children linked to the parent account |
| message.data.totalStudyHours | number | Total study hours across all children |
| message.data.studyInsight | string | Comparative insight about study hours |
| message.data.weeklyInsights | array | Array of weekly insights for each child |
| message.data.weeklyInsights[].name | string | Child's name |
| message.data.weeklyInsights[].subject | string | Subject name |
| message.data.weeklyInsights[].insight | string | Detailed insight about the child's performance |

**Example Response:**
```json
{
  "status": 200,
  "message": {
    "data": {
      "totalChildren": 2,
      "totalStudyHours": 2.5,
      "studyInsight": "12% rise since last week",
      "weeklyInsights": [
        {
          "name": "Aarav",
          "subject": "Mathematics",
          "insight": "Showed 25% improvement in quadratic equations this week"
        },
        {
          "name": "Aarav",
          "subject": "Mathematics",
          "insight": "Struggling with multiplication tables - needs extra practice"
        },
        {
          "name": "Priya",
          "subject": "Science",
          "insight": "Completed all chapters in Photosynthesis with 90+ accuracy"
        }
      ]
    }
  }
}
```

---

## Common Response Structure

All API responses follow a consistent structure:

```json
{
  "status": number,
  "message": {
    "data": <response_data>
  }
}
```

### HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input parameters |
| 401 | Unauthorized - Authentication required or failed |
| 403 | Forbidden - Access denied |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error |

---

## Error Response Structure

When an error occurs, the API returns the following structure:

```json
{
  "status": <error_status_code>,ac
  "message": {
    "error": "string",
    "details": "string (optional)"
  }
}
```

---

## Authentication

### Cookie-Based Authentication
- After successful login via OTP verification, authentication cookies are set automatically
- All subsequent requests include these cookies via `credentials: "include"`
- The cookies contain the access token and are httpOnly for security

### Headers
- `Content-Type: application/json` - Required for POST requests
- `Cookie: <auth_cookies>` - Automatically included for authenticated requests

---

POST /profile/create
POST /auth/login
POST /auth/verify_otp
GET /profile/dashboard
GET /profile/progress
GET /profile/subjects
POST /convo/
GET /convo/history
