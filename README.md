# Reddit Competitive Intelligence Platform

A multi-tenant Reddit sentiment analysis and competitive intelligence platform. Analyze brand mentions, track competitors, and get AI-powered insights from Reddit discussions.

## Features

- **Sentiment Analysis**: HuggingFace transformer-based sentiment scoring
- **Competitor Tracking**: Monitor competitor mentions and switching behavior
- **LLM Deep Analysis**: AI-powered insights via OpenRouter (GPT-4o)
- **Beautiful Dashboard**: Streamlit-based UI with executive reports
- **Multi-Client Support**: Configure multiple brands/clients via JSON
- **Cloud Ready**: Deploy to Streamlit Cloud with proper authentication

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Connective3Innovation/reddit_sentiment_analyzer.git
cd reddit_sentiment_analyzer
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Reddit App (Required)

1. Go to [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
2. Click **"create another app..."**
3. Fill in:
   - **name**: `sentiment-analyzer`
   - **type**: Select **script** (important!)
   - **redirect uri**: `http://localhost:8080`
4. Click **"create app"**
5. Note your **client_id** (under app name) and **secret**

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Reddit API (Required)
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=sentiment-project (by u/your_reddit_username)

# For cloud deployment (Streamlit Cloud), also add:
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password

# LLM Analysis (Optional - for AI insights)
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openai/gpt-4o

# BigQuery Storage (Optional)
REDDIT_GCP_PROJECT=your_gcp_project_id
```

### 6. Run the App

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Usage

### Streamlit Dashboard

1. **Configure** your brand and competitors in the sidebar
2. Set **days to analyze** and **post limit**
3. Enable **LLM Deep Analysis** for AI-powered insights
4. Click **Run Analysis**

### CLI Tool

```bash
# Basic analysis
python run_analysis.py --client capital_one --days 30 --limit 500

# With LLM analysis and BigQuery storage
python run_analysis.py --client capital_one --days 30 --llm --save
```

## Deployment to Streamlit Cloud

### 1. Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set main file: `streamlit_app.py`

### 3. Configure Secrets

In Streamlit Cloud > App Settings > Secrets, add:

```toml
REDDIT_CLIENT_ID = "your_client_id"
REDDIT_CLIENT_SECRET = "your_client_secret"
REDDIT_USER_AGENT = "sentiment-project (by u/your_username)"
REDDIT_USERNAME = "your_reddit_username"
REDDIT_PASSWORD = "your_reddit_password"
OPENROUTER_API_KEY = "your_openrouter_key"
```

**Important**: For cloud deployment, you MUST include `REDDIT_USERNAME` and `REDDIT_PASSWORD`. Cloud provider IPs are blocked by Reddit for application-only OAuth.

## Project Structure

```
reddit/
├── streamlit_app.py              # Main Streamlit dashboard
├── run_analysis.py               # CLI tool
├── requirements.txt              # Python dependencies
├── .env                          # Local environment variables (git-ignored)
└── src/reddit_sentiment/
    ├── api/
    │   └── reddit_client.py      # PRAW wrapper
    ├── auth/
    │   └── reddit_auth.py        # Reddit OAuth client factory
    ├── analytics/
    │   ├── competitor_analyzer.py # Competitor mention analysis
    │   ├── opportunity_scorer.py  # Content opportunity scoring
    │   └── bq_store.py           # BigQuery storage
    ├── config/
    │   └── clients.py            # Multi-client configuration
    ├── data/
    │   ├── collector.py          # Reddit data collection
    │   └── preprocess.py         # Text cleaning
    ├── llm/
    │   ├── openrouter_client.py  # LLM API client
    │   └── prompts.py            # LLM prompt templates
    └── sentiment/
        ├── hf_engine.py          # HuggingFace sentiment
        └── vader_engine.py       # VADER sentiment (fast)
```

## Configuration

### Multi-Client Setup

Create a `clients.json` file for multiple brands:

```json
{
  "clients": [
    {
      "client_id": "capital_one",
      "client_name": "Capital One",
      "primary_brand": "capital one",
      "search_keywords": ["capital one", "capitalone", "cap one"],
      "competitors": ["chase", "amex", "discover", "citi"],
      "industry": "financial_services"
    }
  ]
}
```

Set the path in your environment:

```env
REDDIT_CLIENTS_CONFIG=./clients.json
```

## Troubleshooting

### 401 Authentication Error on Cloud

This happens when Reddit blocks cloud provider IPs. Fix:

1. Ensure your Reddit app type is **"script"** (not "web app")
2. Add `REDDIT_USERNAME` and `REDDIT_PASSWORD` to secrets
3. The username/password must belong to the **same account** that created the Reddit app

### No Data Found

- Check your search keywords are correct
- Increase `days_back` parameter
- Try broader search terms

### LLM Analysis Not Working

- Verify `OPENROUTER_API_KEY` is set
- Check your OpenRouter account has credits
- Try a different model: `OPENROUTER_MODEL=anthropic/claude-3-haiku`

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues, please open a GitHub issue or contact the maintainers.
