import React, { useState } from "react";

// #region agent log
const DEBUG_ENDPOINT =
  "http://127.0.0.1:7664/ingest/53259b0f-de2b-42e7-8370-731eb9b74871";

function sendDebugLog({ runId, hypothesisId, location, message, data }) {
  fetch(DEBUG_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Debug-Session-Id": "15dc20"
    },
    body: JSON.stringify({
      sessionId: "15dc20",
      runId,
      hypothesisId,
      location,
      message,
      data,
      timestamp: Date.now()
    })
  }).catch(() => {});
}
// #endregion agent log

const API_BASE_URL = "http://localhost:8000";

function App() {
  const [threadId, setThreadId] = useState(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const url = threadId
        ? `${API_BASE_URL}/threads/${threadId}`
        : `${API_BASE_URL}/threads`;

      // #region agent log
      sendDebugLog({
        runId: "initial",
        hypothesisId: "H3",
        location: "src/App.jsx:handleSend:beforeFetch",
        message: "Sending chat request",
        data: {
          hasThreadId: !!threadId,
          messageLength: userMessage.content.length
        }
      });
      // #endregion agent log

      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userMessage.content })
      });

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const data = await res.json();
      if (!threadId) {
        setThreadId(data.thread_id);
      }

      // #region agent log
      sendDebugLog({
        runId: "initial",
        hypothesisId: "H3",
        location: "src/App.jsx:handleSend:afterSuccess",
        message: "Chat request success",
        data: {
          hasThreadId: !!(threadId || data.thread_id),
          responseLength:
            typeof data.response === "string" ? data.response.length : null,
          threadIdSet: !threadId && !!data.thread_id
        }
      });
      // #endregion agent log

      const assistantMessage = {
        role: "assistant",
        content: data.response
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      // #region agent log
      sendDebugLog({
        runId: "initial",
        hypothesisId: "H3",
        location: "src/App.jsx:handleSend:onError",
        message: "Chat request error",
        data: {
          hasThreadId: !!threadId,
          errorName: err?.name,
          errorMessage: err?.message
        }
      });
      // #endregion agent log

      console.error(err);
      setError(err.message || "Something went wrong");
      setMessages((prev) => prev.slice(0, -1)); // remove last user message on error
    } finally {
      setLoading(false);
    }
  };

  const handleNewThread = () => {
    setThreadId(null);
    setMessages([]);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>LlamaIndex Chatbot</h1>
        <div className="thread-info">
          <span>
            Thread: {threadId ? threadId : "New (not created yet)"}
          </span>
          <button onClick={handleNewThread} className="secondary">
            New Thread
          </button>
        </div>
      </header>

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              Start chatting and a new thread will be created automatically.
            </div>
          )}
          {messages.map((m, idx) => (
            <div
              key={idx}
              className={`message message-${m.role}`}
            >
              <div className="message-role">
                {m.role === "user" ? "You" : "Assistant"}
              </div>
              <div className="message-content">{m.content}</div>
            </div>
          ))}
          {loading && (
            <div className="message message-assistant">
              <div className="message-role">Assistant</div>
              <div className="message-content">Thinking...</div>
            </div>
          )}
        </div>
      </main>

      {error && <div className="error-banner">Error: {error}</div>}

      <form className="input-bar" onSubmit={handleSend}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={loading}
        />
        <button type="submit" disabled={loading}>
          Send
        </button>
      </form>
    </div>
  );
}

export default App;

