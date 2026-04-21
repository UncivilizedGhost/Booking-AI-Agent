# Workshop Booking Agent
### AI Agents in Production Engineering (Thesis Project)

This repository contains an autonomous booking agent designed to streamline workshop scheduling. The system uses the AutoGen framework and Google Gemini 2.5 Flash to handle natural language requests and task execution.



## 📋 Prerequisites

* **Python:** 3.12.7
* **API Access:** A Google Gemini API Key. You can obtain one at [Google AI Studio](https://aistudio.google.com/app/apikey).



## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/UncivilizedGhost/Booking-AI-Agent.git
```

### 2. Install Dependencies
Ensure you are using Python 3.12.7, then run:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a file named `.env` in the root directory. You must include your API key and a hashed password for admin access.

**Required `.env` parameters:**
* `GOOGLE_API_KEY`: Your unique key from Google AI Studio.
* `PASSWORD`: The **SHA-256 hash** of your chosen password.


**Example `.env` format:**
```text
GOOGLE_API_KEY=your_actual_api_key_here
PASSWORD=ef92b778ba94c0334... (your hashed string)
```



## Running the Application

Once the dependencies are installed and the `.env` file is configured, launch the agent by running:

```bash
python app.py
```




### 📝 Note on Contribution
This project was developed as part of a thesis exploring how AI Agents can be utilized in produciton engineering.
For academic queires, please contact me
