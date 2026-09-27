# Multi-Agent-Customer-Support-Autogen

A customer support chatbot built from three agents using Microsoft's AutoGen framework, with a Streamlit user interface. For each question:

1. **Knowledge Assistant** answers from the model's own knowledge.
2. **Web Search Assistant** searches the web (Serper API) and returns the top result (title, snippet and source link).
3. **Reconciliation Entry Agent** saves the question and both answers to `answers.txt`, then merges them into one final answer for the user.

The app also has input/output guardrails and a built-in evaluation page. Everything lives in a single file, `app.py`.

## Flow

![AI Customer Support flow: input guardrails, greeting check, AutoGen round-robin team, output guardrails, Streamlit UI](docs/architecture.png)

## Project files

| File | Purpose |
|---|---|
| `app.py` | The whole application: UI, agents, guardrails and evaluation |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for the required API keys |
| `.streamlit/config.toml` | Light theme and accent colour for the UI |
| `docs/architecture.png` | Flow diagram shown above |
| `Dockerfile`, `docker-compose.yml`, `Caddyfile`, `.dockerignore` | Deployment (see [Deployment](#deployment-docker--caddy--ssl)) |
| `answers.txt` | Written by the Entry Agent for the latest chat question |
| `eval_answers.txt` | Written during evaluation runs instead of `answers.txt` (git-ignored) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in both keys:

```
OPENAI_API_KEY=...
SERPER_API_KEY=...
```

The app stops with an error message if either key is missing.

## Run

```bash
streamlit run app.py
```

The sidebar has a **💬 Chat / 📊 Evaluation** switch.

## How the agents work

- The agents run in a `RoundRobinGroupChat` using `gpt-4o-mini` (`DEFAULT_MODEL` in `app.py`).
- The Web Search Assistant calls `web_search_tool`, which returns the top Serper result (title, snippet and source link).
- The Entry Agent calls `save_to_file(query, answer1, answer2)`. It then writes the final answer in the same turn (`reflect_on_tool_use=True`) and ends with `TERMINATE`.
- The run stops when `TERMINATE` appears, or after 5 chat messages as a safety net. A normal run is 4 messages: the user question plus one message from each agent.

## Chat features

- **Greetings and small talk** get an instant reply without starting the agents. This covers "hi", "who are you", "what can you do", "thanks", "bye" and similar.
- **Vague inputs** get a request for more detail instead of an agent run. This covers inputs like "what", "ok", "hmm" or "???".
- **Suggested questions** appear on the welcome screen and can be clicked to send.
- **Each answer** appears in a "Final support answer" card, with expanders showing the Knowledge Assistant's answer and the web search findings.
- **Clear conversation** is a button in the sidebar that resets the chat.

## Guardrails

### Input (before the question reaches the agents)

- **Length limit:** messages over 2,000 characters are rejected.
- **Prompt injection:** pattern checks refuse messages like "ignore previous instructions", "reveal your system prompt" or "developer mode".
- **Harmful content:** checked with OpenAI's moderation endpoint (`omni-moderation-latest`).
- **Personal data redaction:**
  - Account numbers are detected by nearby words such as "account no", "a/c" or "acct".
  - Card numbers are redacted only when they pass the Luhn checksum, so order and tracking numbers are left alone.
  - SSNs, API keys, passwords, PINs, OTPs, CVVs, email addresses and phone numbers are also redacted. Phone numbers must contain separators to be detected.
  - The redacted text is what gets shown, stored, sent to the agents and web search, and saved to `answers.txt`. The user sees a 🛡️ note explaining what was removed.

### Output (before the answer is shown)

- **Leaked instructions:** answers that repeat the agents' system instructions are blocked.
- **Harmful content:** the final answer goes through the same moderation check.
- **Sensitive data:** account numbers, card numbers, SSNs, API keys and passwords/PINs are removed from the final answer and from both agent drafts. Emails and phone numbers are kept because support answers often contain contact details.

If the moderation API is unavailable, messages are allowed through (fail open) so the chat keeps working.

## Evaluation

The **📊 Evaluation** page runs test questions through the same path as the chat: input guardrail, then the three agents, then the output guardrail.

- **Test set:** 10 built-in questions (`EVAL_DATASET` in `app.py`), each with 4 expected key points.
- **Options:** how many questions to run, the agent model, and the judge model (default `gpt-4o`).
- **Judge:** an LLM judge grades retrieval and answers.
- **Live calls:** every run makes real OpenAI and Serper calls.

It reports four areas:

| Area | What's measured |
|---|---|
| **Retrieval quality** | How often the search runs and how often it fails. Judge scores (1–5) for how well the search query matches the question, how relevant the result is, and how trustworthy the source is. Context recall is the share of expected key points the search result supports. |
| **Answer quality (Entry Agent)** | Pass rate. Judge scores (1–5) for correctness, faithfulness, how well it merged the two inputs, and clarity. Key-point recall compared with the Knowledge Assistant alone. Whether it saved before answering, whether it ended with `TERMINATE`, and how often the output guardrail blocked it. |
| **Cost** | Tokens per agent (from AutoGen usage records), cost per question and per 1,000 questions, and the split between LLM and search cost. The judge's cost is shown separately. |
| **Latency** | Median (p50), 95th-percentile (p95) and slowest end-to-end time. Average time per stage (guardrails and each agent), and time spent waiting on the Serper API. |

Per-question results can be downloaded as CSV or JSON.

**Cost figures are estimates.** They use the prices in `MODEL_PRICES` (USD per 1M tokens) and `SEARCH_PRICE` (USD per search, default $0.001) in `app.py`. Update them to match your OpenAI and Serper pricing.

## Deployment (Docker + Caddy + SSL)

The app runs in Docker on a VPS. [Caddy](https://caddyserver.com) sits in front as a reverse proxy: it serves your domain, gets a free Let's Encrypt SSL certificate, renews it automatically, and redirects HTTP to HTTPS.

| File | Purpose |
|---|---|
| `Dockerfile` | Builds the Streamlit app image (Python 3.13, non-root user, health check) |
| `docker-compose.yml` | Runs the app container plus the Caddy container on ports 80/443 |
| `Caddyfile` | Domain, SSL and reverse-proxy settings |
| `.dockerignore` | Keeps `.env`, `.venv` and local files out of the image |

### 1. Point the domain at the VPS

In your domain's DNS settings, add an `A` record:

| Type | Name | Points to |
|---|---|---|
| A | `@`, or a subdomain such as `support` | your VPS public IP |

Wait until `ping your-domain` returns the VPS IP. Caddy can't get a certificate until DNS resolves.

### 2. Prepare the VPS (Ubuntu)

```bash
# Install Docker (skip if you chose Hostinger's Docker VPS template)
curl -fsSL https://get.docker.com | sh

# Open SSH, HTTP and HTTPS
ufw allow 22 && ufw allow 80 && ufw allow 443 && ufw enable
```

If a firewall is also enabled in Hostinger's VPS panel, allow ports 80 and 443 there too.

### 3. Copy the project and configure it

```bash
git clone <your-repo-url> support && cd support
# or copy the folder from your machine: scp -r . root@<vps-ip>:~/support

cp .env.example .env
nano .env   # set OPENAI_API_KEY, SERPER_API_KEY, DOMAIN, ACME_EMAIL
```

### 4. Start

```bash
docker compose up -d --build
docker compose logs -f caddy   # watch for "certificate obtained successfully"
```

Then open `https://your-domain`.

### Updating

```bash
git pull && docker compose up -d --build
```

### Protecting your API credits

Once deployed, anyone who finds the URL can use the chat and the Evaluation page, which spend your OpenAI and Serper credits. To require a password:

1. Run `docker compose exec caddy caddy hash-password`.
2. Paste the hash into the commented `basic_auth` block in `Caddyfile`.
3. Run `docker compose restart caddy`.

## Known limitations

- **Pattern-based checks:** injection and personal-data detection use regex patterns, so they can be worded around and may miss formats they don't know. An account number written without words like "account no" is not detected as an account number.
- **Exact-match small talk:** greetings and vague inputs are matched exactly, so longer messages always go to the agents.
- **One search result:** web search uses only the top result.
- **Judge variation:** LLM-judge scores vary between runs, so compare full runs rather than single questions.
