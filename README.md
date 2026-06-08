# Code Assistant: AI-Powered LangGraph Application

An intelligent, end-to-end Python code assistant built using **LangGraph**, **LangChain**, and **Retrieval-Augmented Generation (RAG)** technology. This system understands user intent (code generation or explanation), retrieves semantically relevant examples from the HumanEval dataset, and leverages advanced LLMs to deliver context-aware, pedagogically-sound responses.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Core Architecture](#core-architecture)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Command-Line Interface](#command-line-interface)
  - [Web Interface (Streamlit)](#web-interface-streamlit)
- [Configuration](#configuration)
- [Component Documentation](#component-documentation)
  - [Core Components](#core-components)
  - [State Machine](#state-machine)
  - [Utilities](#utilities)
  - [Database & Storage](#database--storage)
- [Dependencies](#dependencies)
- [Development & Contributing](#development--contributing)
- [Troubleshooting](#troubleshooting)

---

## 📌 Overview

**Cellula Code Assistant** is a sophisticated educational AI tutor specialized in Artificial Intelligence, Machine Learning, and Python programming. It combines multiple advanced NLP techniques:

### Key Features

- **Intent Classification**: Automatically detects whether users want to generate code or request explanations
- **Semantic Retrieval**: Leverages ChromaDB with embeddings to find contextually relevant code examples from HumanEval
- **Context-Aware Responses**: Maintains conversation history and generates responses informed by retrieved examples
- **Dual Interface**: Both CLI and web-based Streamlit interface for accessibility
- **User Management**: Session-based authentication and conversation history persistence
- **Memory Management**: Tracks conversation history across sessions

### Technology Stack

- **Framework**: LangGraph & LangChain (AI orchestration)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: ChromaDB (semantic search)
- **LLM Provider**: OpenRouter API (supports multiple models)
- **Web Framework**: Streamlit
- **Data Source**: HumanEval benchmark dataset
- **Language**: Python 3.8+

---

## 📂 Project Structure

```
project/
├── main.py                          # CLI entry point
├── app5.py                          # Streamlit web interface
├── context.py                       # Shared context dataclass
├── state_graph.py                   # State machine implementation
├── graph_builder.py                 # Graph construction logic
├── auth.py                          # User authentication utilities
├── db.py                            # Conversation persistence
├── requirements.txt                 # Project dependencies
├── user_data.json                   # User profiles (persistent)
├── users.json                       # Alternative user storage
├── README.md                        # Project documentation
│
├── states/                          # State machine states
│   ├── __pycache__/
│   ├── chat_state.py                # User input handling
│   ├── router_state.py              # Intent classification
│   ├── explain_code_state.py        # Code explanation logic
│   ├── generate_code_state.py       # Code generation logic
│   └── end_state.py                 # Response output state
│
├── utils/                           # Utility modules
│   ├── __pycache__/
│   ├── llm_client.py                # OpenRouter LLM interface
│   ├── intent_classifier.py         # Intent detection
│   ├── retriever.py                 # Semantic document retrieval
│   └── humaneval_db.py              # ChromaDB initialization & operations
│
├── chroma_db/                       # Vector database storage
│   ├── chroma.sqlite3               # ChromaDB persistent storage
│   └── 4d20436b-4f14-4fc7-be5a-9c5263402af6/  # Embedding data
│
└── __pycache__/                     # Python cache
```

---

## 🏗️ Core Architecture

### State Machine Flow

The application follows a **finite state machine (FSM)** pattern with the following states:

```
[ChatState] → [RouterState] → [ExplainCodeState / GenerateCodeState] → [EndState]
    ↑            ↓                          ↓
    └─────────────────────── Routes based on intent
```

#### State Descriptions

| State | Purpose | Actions |
|-------|---------|---------|
| **ChatState** | Initial input processing | Stores user input in context & LLM memory |
| **RouterState** | Intent classification | Determines if user wants to generate or explain code |
| **GenerateCodeState** | Code generation | Retrieves examples, generates code via LLM |
| **ExplainCodeState** | Code explanation | Retrieves examples, explains code to user |
| **EndState** | Response output | Displays assistant response to user |

### Data Flow

```
User Input
    ↓
ChatState: Store input in memory
    ↓
RouterState: Classify intent (explain/generate)
    ↓
[Split Based on Intent]
    ├→ ExplainCodeState: Retrieve examples → LLM explains code
    └→ GenerateCodeState: Retrieve examples → LLM generates code
    ↓
EndState: Display response
    ↓
Update conversation history
```

---


**Interactive commands:**

```
You: Generate a function that calculates fibonacci numbers
[ChatState] stored user input in memory
[RouterState] Inferred intent: generate
[GenerateCodeState] Retrieved 3 similar examples...
===== Assistant Response =====
[Assistant's generated code response]
==============================

You: Explain the code above
[RouterState] Inferred intent: explain
[ExplainCodeState] Retrieved 3 similar examples...
===== Assistant Response =====
[Assistant's explanation]
==============================

You: clear
[Clears conversation history and resets context]

You: exit
[Exits the application]
```

### Web Interface (Streamlit)

**Launch the Streamlit interface:**

```bash
streamlit run app5.py
```

This opens a web browser at `http://localhost:8501` with:

- **Login/Signup tabs**: User authentication with persistent storage
- **Chat interface**: Interactive messaging with the assistant
- **Session management**: Maintains conversation history per user
- **Responsive design**: Wide layout for optimal UX

**Features in Streamlit UI:**

✅ User authentication (create account or log in)
✅ Persistent conversation history
✅ Real-time intent classification display
✅ Retrieved examples display
✅ Beautiful formatted responses
✅ Session state management

---

## ⚙️ Configuration

### Model Selection

Edit `utils/llm_client.py` to change the LLM model:

```python
# Currently uses meta-llama/llama-3.3-8b-instruct:free
# Available models at https://openrouter.ai/docs#models
MODEL_ID = "your-model-id"
```

### Embedding Model

Customize the embedding model in `utils/humaneval_db.py`:

```python
model_name="all-MiniLM-L6-v2"  # Default: lightweight & fast
# Alternatives: 
# - "all-mpnet-base-v2" (more accurate, slower)
# - "paraphrase-MiniLM-L6-v2" (good balance)
```

### Retrieval Parameters

Adjust in `states/generate_code_state.py` and `states/explain_code_state.py`:

```python
ctx.retrieved_examples = self.agent.retriever.retrieve(ctx.user_input, top_k=3)
# Change top_k to retrieve more/fewer examples (default: 3)
```

### System Prompt

Customize the AI tutor's personality in `utils/llm_client.py`:

```python
SYSTEM_PROMPT = """[Your custom system prompt here]"""
```

---

## 📚 Component Documentation

### Core Components

#### **context.py** - Shared Context Object

Maintains state throughout the state machine execution:

```python
@dataclass
class Context:
    user_input: str                    # Raw user input
    intent: str                        # Classified intent (explain/generate/chat)
    retrieved_examples: List[Dict]     # Retrieved code examples from ChromaDB
    prompt: str                        # Processed prompt for LLM
    llm_response: str                  # LLM's response
    convo_history: List[Dict]          # Conversation message history
    metadata: Dict[str, Any]           # Additional metadata (user_id, etc.)
```

#### **state_graph.py** - State Machine Framework

Custom lightweight state machine implementation:

```python
class State:
    """Base state class with transitions"""
    def __init__(self, name: str, action: Callable)
    def add_transition(self, condition_fn, target_state_name)

class StateGraph:
    """Manages state execution and transitions"""
    def add_state(self, state: State, start: bool = False)
    def run(self, context: Context)
```

#### **graph_builder.py** - Graph Construction

Builds the complete state machine with transitions:

```python
def build_graph(agent) -> StateGraph:
    # Creates and connects all states
    # Returns executable StateGraph instance
```

### State Machine

#### **ChatState** (`states/chat_state.py`)

- **Purpose**: Initial input processing
- **Actions**:
  - Appends user message to conversation history
  - Saves input to LLM memory (if available)
- **Transitions**: Always → RouterState

#### **RouterState** (`states/router_state.py`)

- **Purpose**: Intent classification
- **Actions**: Classifies intent using regex-based classifier
- **Transitions**:
  - Intent = "explain" → ExplainCodeState
  - Intent = "generate" → GenerateCodeState
  - Intent = other → EndState

#### **GenerateCodeState** (`states/generate_code_state.py`)

- **Purpose**: Code generation
- **Actions**:
  1. Retrieves similar code examples from ChromaDB
  2. Constructs prompt with user request + examples
  3. Calls LLM for code generation
  4. Saves response to conversation history
- **Transitions**: Always → EndState

#### **ExplainCodeState** (`states/explain_code_state.py`)

- **Purpose**: Code explanation
- **Actions**:
  1. Retrieves similar code examples from ChromaDB
  2. Constructs prompt with user request + examples
  3. Calls LLM for explanation
  4. Saves response to conversation history
- **Transitions**: Always → EndState

#### **EndState** (`states/end_state.py`)

- **Purpose**: Response output
- **Actions**:
  - Formats and displays assistant response
  - Ensures response is stored in conversation history

### Utilities

#### **llm_client.py** - LLM Interface

Manages OpenRouter API communication:

```python
class LLMClient:
    def __init__(self)                  # Initializes with API key
    def call(prompt, retrieved_docs_texts, include_retrieved_in_output, user_id)
    def clear_memory()                  # Clears conversation memory
```

**Key Features:**
- Async HTTP requests to OpenRouter API
- Conversation memory management
- Configurable system prompt
- Support for retrieved document injection

#### **intent_classifier.py** - Intent Detection

Regex-based intent classifier:

```python
class IntentClassifier:
    def infer(self, text: str) -> str:  # Returns: "generate", "explain", or "chat"
```

**Classification Rules:**
- **Generate**: Keywords like "generate", "create", "def", "class" + code markers
- **Explain**: Keywords like "explain", "why", "how", "describe"
- **Chat**: Keywords like "hi", "help", "who are you"

#### **retriever.py** - Document Retrieval

Semantic search wrapper:

```python
class Retriever:
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        # Returns list of dictionaries with keys:
        # - prompt: Problem description
        # - canonical_solution: Reference solution
        # - task_id: HumanEval task identifier
```

#### **humaneval_db.py** - ChromaDB Management

Vector database operations:

```python
def load_humaneval_data() -> pd.DataFrame
    # Loads HumanEval dataset

def init_chroma(db_path, model_name) -> Collection
    # Creates or loads ChromaDB collection

def store_embeddings(collection)
    # Embeds HumanEval solutions and stores in ChromaDB

def retrieve_similar(collection, query, top_k) -> Tuple[List, List]
    # Semantic search returning documents and metadata
```

### Database & Storage

#### **auth.py** - User Authentication

```python
def create_users_table()          # Creates SQLite users table
def add_user(username, password)  # Registers new user (SHA256 hashed)
def verify_user(username, password)  # Validates credentials
```

#### **db.py** - Conversation Persistence

```python
def create_conversations_table()  # Creates SQLite conversations table
def save_message(username, message, response)  # Stores message pair
def load_conversation(username)   # Retrieves user's conversation history
```

#### **Data Files**

- `user_data.json`: JSON-based user profiles (Streamlit)
- `users.json`: Alternative user storage format
- `chroma_db/`: Vector database directory with persistent storage

---

## 📦 Dependencies

### Core Dependencies

```
sentence-transformers>=2.2.0        # Embedding models
scikit-learn>=1.0.0                 # ML utilities
openai>=0.27.0                      # OpenAI API client
```

### Additional Dependencies (Inferred)

```
langgraph                           # State machine framework
langchain                           # LLM orchestration
chromadb                            # Vector database
streamlit                           # Web interface
pandas                              # Data processing
datasets                            # HumanEval dataset
python-dotenv                       # Environment variables
requests                            # HTTP requests
```

### Full Installation

```bash
pip install -r requirements.txt
# Install missing dependencies if needed:
pip install langgraph langchain chromadb streamlit pandas datasets python-dotenv
```

---

## 🛠️ Development & Contributing

### Project Architecture Principles

1. **Separation of Concerns**: Each module has a single responsibility
2. **State Pattern**: Decoupled state logic for maintainability
3. **Configuration-Driven**: Easy customization via environment variables
4. **Extensibility**: Add new states or utilities without modifying core

### Adding New States

1. Create new file in `states/` directory
2. Inherit from `State` class:

```python
from state_graph import State

class MyCustomState(State):
    def __init__(self, agent):
        super().__init__("my_state", self.action)
        self.agent = agent

    def action(self, ctx):
        # Implement your logic
        pass
```

3. Register in `graph_builder.py`:

```python
my_state = MyCustomState(agent)
sg.add_state(my_state)
my_prev_state.add_transition(lambda ctx: True, "my_state")
```

### Code Quality Guidelines

- Follow PEP 8 naming conventions
- Add docstrings to functions
- Use type hints where applicable
- Handle exceptions gracefully
- Test with both CLI and Streamlit interfaces

---

## 🐛 Troubleshooting

### Common Issues

#### **"OPENROUTER_API_KEY not found" Warning**

**Solution**: Ensure `.env` file exists in project root:

```bash
echo "OPENROUTER_API_KEY=your_key" > .env
```

#### **ChromaDB Initialization Fails**

**Solution**: Clear cache and reinitialize:

```bash
rm -r chroma_db
python main.py  # Reinitializes database
```

#### **Intent Classification Always Returns "explain"**

**Possible Causes:**
- Keyword matching issue
- Check `utils/intent_classifier.py` regex patterns
- User input doesn't match classification keywords

**Solution**: Update classification rules in `IntentClassifier.mapping`

#### **Streamlit Shows "No module named..." Error**

**Solution**: Install missing dependencies:

```bash
pip install -r requirements.txt
pip install langgraph langchain chromadb
```

#### **Slow Response Times**

**Causes:**
- Large top_k value in retrieval
- Slow embedding model
- Network latency to OpenRouter

**Solutions:**
- Reduce `top_k` from 3 to 1-2
- Use faster embedding model: `paraphrase-MiniLM-L6-v2`
- Check internet connection

#### **Memory Grows During Long Sessions**

**Solution**: Use CLI `clear` command to reset memory:

```
You: clear
```

Or restart the application.
