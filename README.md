# PlantIQX AI Backend

Flask service for AI-driven operational summaries, chat assistance, and feedback capture.

## Overview

The `ai-backend` powers the Admin UI AI workflows (for example `/iot/ai-insights`) and provides:

- Context-aware insight summaries
- Interactive operations chat responses
- Feedback endpoint for response quality signals

It runs in two modes:

- LLM mode (when `OPENAI_API_KEY` is configured)
- Local fallback mode (deterministic rules, no external model dependency)

## Stack

- Python 3
- Flask
- Flask-CORS
- python-dotenv
- LangChain + OpenAI integration (`langchain-openai`)

## API Endpoints

- `GET /` - service metadata + active provider/model
- `GET /health` - health + LLM configuration status (`mode`: `llm` or `fallback`)
- `POST /api/v1/ai/insights/summary` - summary, headline, key metrics, recommendations, predictions
- `POST /api/v1/ai/insights/anomalies` - anomaly / outlier scan across the fleet context
- `POST /api/v1/ai/insights/root-cause` - probable causes + investigation steps for a subject
- `POST /api/v1/ai/chat` - conversational assistant with dynamic follow-up suggestions
- `POST /api/v1/ai/feedback` - thumbs up/down capture

All insight/chat responses include `source` (`llm` or `fallback`), `model`, and
`latencyMs` so the UI can show whether output is live or heuristic.

## Providers

The backend is provider-agnostic — it works with any OpenAI-compatible API.
The provider is auto-detected from the API-key prefix:

| Prefix   | Provider   | Default model                                  |
|----------|------------|------------------------------------------------|
| `sk-`    | openai     | `gpt-4o-mini`                                  |
| `gsk_`   | groq       | `llama-3.3-70b-versatile`                      |
| `sk-or-` | openrouter | `openai/gpt-4o-mini`                           |
| `tgp_`   | together   | `meta-llama/Llama-3.3-70B-Instruct-Turbo`      |

Override with `AI_PROVIDER` / `AI_BASE_URL` if needed. A `gpt-*` model set
against a non-OpenAI provider is automatically swapped for that provider's
default, so a stale `AI_MODEL` never breaks the service.

## Project Files

```text
ai-backend/
|-- app.py
|-- requirements.txt
`-- .env.example
```

## Environment Variables

```env
FLASK_ENV=development
FLASK_PORT=5005
AI_MODEL=llama-3.3-70b-versatile
OPENAI_API_KEY=        # or AI_API_KEY / GROQ_API_KEY
# AI_PROVIDER=groq     # optional — force provider
# AI_BASE_URL=         # optional — custom OpenAI-compatible endpoint
```

## Local Setup

```bash
cd ai-backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Default URL: `http://localhost:5005`

## Integration with Admin UI

Frontend client file:

- `Admin UI/src/store/requests/aiApiClient.js`

Expected frontend env:

```env
VITE_AI_API_URL=http://localhost:5005
```

## Notes

- If `OPENAI_API_KEY` is missing or model init fails, the service still responds using fallback logic.
- CORS is enabled for `/api/*` routes by default.

## Related Modules

- Frontend: `../Admin UI`
- IoT backend: `../iot-backend`
- Architecture: `../ARCHITECTURE.md`
