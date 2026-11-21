
import { useState } from "react";
import {
  Activity,
  HeartPulse,
  MessageSquare,
  Plus,
  Search,
  Settings,
  ShieldCheck,
  Send,
  BookOpen,
  Menu,
  Stethoscope,
  Sparkles,
  Bot,
  LoaderCircle,
  FileText,
  AlertCircle,
  ExternalLink,
  ChevronRight,
} from "lucide-react";

import Dashboard from "./pages/Dashboard";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000/api/chat";

const suggestions = [
  {
    icon: HeartPulse,
    title: "Diabetes",
    question: "What are the common symptoms of diabetes?",
  },
  {
    icon: Activity,
    title: "Blood pressure",
    question: "What is hypertension and what causes it?",
  },
  {
    icon: Stethoscope,
    title: "Health conditions",
    question: "What are the common symptoms of asthma?",
  },
  {
    icon: BookOpen,
    title: "Medical knowledge",
    question: "How does the human immune system work?",
  },
];

const pageTitles = {
  dashboard: "Dashboard",
  assistant: "Medical Assistant",
  library: "Medical Library",
};

const pageDescriptions = {
  dashboard: "Health Analytics Overview",
  assistant: "Evidence-Grounded Medical AI",
  library: "Medical Knowledge Resources",
};

function displayValue(value) {
  if (value === null || value === undefined) return "";

  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value.map(displayValue).filter(Boolean).join(", ");
  }

  return "";
}

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(false);

  // Start a new conversation.
  const startNewChat = () => {
    if (loading) return;

    setMessages([]);
    setQuestion("");
    setActivePage("assistant");
  };

  // Send a question to the existing FastAPI medical RAG backend.
  const sendMessage = async (text = question) => {
    const userQuestion = text.trim();

    if (!userQuestion || loading) return;

    setActivePage("assistant");
    setQuestion("");
    setLoading(true);

    const userMessage = {
      id: Date.now(),
      role: "user",
      content: userQuestion,
    };

    const assistantId = Date.now() + 1;

    const assistantMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      questionType: "",
      sections: [],
      sources: [],
      evidence: [],
      safety: {},
      status: "loading",
      loading: true,
    };

    setMessages((previous) => [
      ...previous,
      userMessage,
      assistantMessage,
    ]);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify({
          question: userQuestion,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || `Request failed with status ${response.status}`
        );
      }

      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: data.quick_answer || "No answer was returned.",
                questionType: data.question_type || "",
                sections: Array.isArray(data.sections)
                  ? data.sections
                  : [],
                sources: Array.isArray(data.sources)
                  ? data.sources
                  : [],
                evidence: Array.isArray(data.evidence)
                  ? data.evidence
                  : [],
                safety: data.safety || {},
                status: data.status || "answered",
                loading: false,
              }
            : message
        )
      );
    } catch (error) {
      setMessages((previous) =>
        previous.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content:
                  error.message ||
                  "Unable to connect to the MedIntel-AI backend.",
                sources: [],
                sections: [],
                evidence: [],
                safety: {},
                status: "error",
                loading: false,
              }
            : message
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  // Render an answer section.
  const renderSection = (section, index) => {
    const title =
      section.title ||
      section.heading ||
      section.name ||
      `Explanation ${index + 1}`;

    const body =
      section.content ||
      section.body ||
      section.text ||
      section.description ||
      "";

    return (
      <section className="answer-section" key={index}>
        <h3>{title}</h3>

        {typeof body === "string" ? (
          <p>{body}</p>
        ) : (
          <p>{displayValue(body)}</p>
        )}

        {Array.isArray(section.bullet_points) &&
          section.bullet_points.length > 0 && (
            <ul>
              {section.bullet_points.map((item, itemIndex) => (
                <li key={itemIndex}>{displayValue(item)}</li>
              ))}
            </ul>
          )}

        {Array.isArray(section.points) &&
          section.points.length > 0 && (
            <ul>
              {section.points.map((item, itemIndex) => (
                <li key={itemIndex}>{displayValue(item)}</li>
              ))}
            </ul>
          )}
      </section>
    );
  };

  // Render evidence supporting the answer.
  const renderEvidence = (evidence, index) => {
    const claim =
      evidence.claim ||
      evidence.statement ||
      evidence.text ||
      evidence.content ||
      "";

    const rawSourceIds =
      evidence.source_ids ||
      evidence.sources ||
      (evidence.source_id ? [evidence.source_id] : []);

    const sourceIds = Array.isArray(rawSourceIds)
      ? rawSourceIds
      : [rawSourceIds];

    return (
      <div className="evidence-item" key={index}>
        <div className="evidence-marker">
          <ShieldCheck size={16} />
        </div>

        <div className="evidence-content">
          <p>{displayValue(claim) || "Supporting evidence"}</p>

          {sourceIds.length > 0 && (
            <small>
              Supporting sources:{" "}
              {sourceIds.map(displayValue).join(", ")}
            </small>
          )}
        </div>
      </div>
    );
  };

  // Build a visual overview only from sections returned by the backend.
  // No medical values, ranges, or risk levels are invented in the frontend.
  const renderVisualOverview = (sections) => {
    if (!Array.isArray(sections) || sections.length === 0) return null;

    const visualItems = sections
      .map((section, index) => {
        const title =
          section.title ||
          section.heading ||
          section.name ||
          `Topic ${index + 1}`;

        const body =
          section.content ||
          section.body ||
          section.text ||
          section.description ||
          (Array.isArray(section.bullet_points)
            ? section.bullet_points[0]
            : Array.isArray(section.points)
              ? section.points[0]
              : "");

        return { title, preview: displayValue(body) };
      })
      .filter((item) => item.title || item.preview);

    if (visualItems.length === 0) return null;

    return (
      <section
        className="answer-visual-overview"
        aria-label="Visual answer overview"
        style={{
          margin: "18px 0",
          padding: "16px",
          border: "1px solid var(--border-color, #e5e7eb)",
          borderRadius: "14px",
          background: "var(--card-background, #ffffff)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            marginBottom: "12px",
          }}
        >
          <Activity size={17} />

          <h3 style={{ margin: 0, fontSize: "15px" }}>
            Answer overview
          </h3>

          <span
            style={{
              marginLeft: "auto",
              fontSize: "12px",
              opacity: 0.7,
            }}
          >
            {visualItems.length} topics from retrieved answer
          </span>
        </div>

        <div
          className="answer-visual-grid"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "10px",
          }}
        >
          {visualItems.map((item, index) => (
            <article
              key={`${item.title}-${index}`}
              style={{
                minWidth: 0,
                padding: "12px",
                borderRadius: "10px",
                border: "1px solid var(--border-color, #e5e7eb)",
                background: "var(--surface-background, #f8fafc)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  marginBottom: "8px",
                }}
              >
                <span
                  style={{
                    display: "grid",
                    placeItems: "center",
                    width: "25px",
                    height: "25px",
                    borderRadius: "8px",
                    background: "var(--accent-soft, #e8f1ff)",
                    color: "var(--accent-color, #2563eb)",
                    fontSize: "12px",
                    fontWeight: 700,
                    flexShrink: 0,
                  }}
                >
                  {index + 1}
                </span>

                <strong style={{ fontSize: "13px" }}>
                  {item.title}
                </strong>
              </div>

              {item.preview && (
                <p
                  style={{
                    margin: 0,
                    fontSize: "12px",
                    lineHeight: 1.55,
                    opacity: 0.78,
                  }}
                >
                  {item.preview.length > 150
                    ? `${item.preview.slice(0, 150).trim()}…`
                    : item.preview}
                </p>
              )}
            </article>
          ))}
        </div>

        <p
          style={{
            margin: "12px 0 0",
            fontSize: "11px",
            opacity: 0.65,
          }}
        >
          Visual summary of the answer sections; it does not represent a
          clinical measurement or risk score.
        </p>
      </section>
    );
  };

  // Render retrieved source references.
  const renderSource = (source, index) => {
    const sourceName =
      source.source ||
      source.document ||
      source.document_name ||
      "Medical reference";

    const page = source.page;
    const rawUrl = source.source_url || source.url;

    const isValidSourceUrl = (() => {
      if (!rawUrl || typeof rawUrl !== "string") return false;

      try {
        const parsed = new URL(rawUrl);

        return (
          ["http:", "https:"].includes(parsed.protocol) &&
          !parsed.hostname.toLowerCase().endsWith("nihseniorhealth.gov")
        );
      } catch {
        return false;
      }
    })();

    return (
      <div
        className="source-item"
        key={`${source.id || sourceName}-${page ?? "na"}-${index}`}
      >
        <FileText size={16} />

        <div className="source-details">
          <span>{sourceName}</span>

          <small>
            {page !== null && page !== undefined
              ? `Page ${page}`
              : source.document_id
                ? `Document ID: ${source.document_id}`
                : source.source_type || "Reference"}

            {source.source_type ? ` · ${source.source_type}` : ""}
          </small>

          {isValidSourceUrl ? (
            <a
              href={rawUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="source-link"
            >
              View source <ExternalLink size={12} />
            </a>
          ) : (
            <small className="source-unavailable">
              Source link unavailable
            </small>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="app-container">
      {/* SIDEBAR */}
      <aside
        className={`sidebar ${sidebarOpen ? "" : "sidebar-hidden"}`}
      >
        {/* BRAND */}
        <div className="brand">
          <div className="brand-icon">
            <HeartPulse size={25} />
          </div>

          <div>
            <h2>
              MedIntel<span>AI</span>
            </h2>
            <p>Medical Intelligence</p>
          </div>
        </div>

        {/* NEW CHAT */}
        <button
          className="new-chat"
          onClick={startNewChat}
          disabled={loading}
        >
          <Plus size={18} />
          <span>New conversation</span>
        </button>

        {/* WORKSPACE NAVIGATION */}
        <div className="sidebar-section">
          <p className="section-label">WORKSPACE</p>

          <button
            className={`nav-item ${
              activePage === "dashboard" ? "active" : ""
            }`}
            onClick={() => setActivePage("dashboard")}
          >
            <Activity size={18} />
            <span>Dashboard</span>
          </button>

          <button
            className={`nav-item ${
              activePage === "assistant" ? "active" : ""
            }`}
            onClick={() => setActivePage("assistant")}
          >
            <MessageSquare size={18} />
            <span>Medical Assistant</span>
          </button>

          <button
            className={`nav-item ${
              activePage === "library" ? "active" : ""
            }`}
            onClick={() => setActivePage("library")}
          >
            <BookOpen size={18} />
            <span>Medical Library</span>
          </button>
        </div>

        {/* RECENT CHATS */}
        <div className="sidebar-section">
          <p className="section-label">RECENT CHATS</p>

          {messages.filter((message) => message.role === "user").length ===
          0 ? (
            <p className="empty-history">
              Your conversations will appear here.
            </p>
          ) : (
            messages
              .filter((message) => message.role === "user")
              .slice(-5)
              .reverse()
              .map((message) => (
                <button
                  className="history-item"
                  key={message.id}
                  onClick={() => setActivePage("assistant")}
                  title={message.content}
                >
                  <MessageSquare size={15} />
                  <span>{message.content}</span>
                </button>
              ))
          )}
        </div>

        {/* SIDEBAR FOOTER */}
        <div className="sidebar-bottom">
          <div className="privacy-note">
            <ShieldCheck size={18} />

            <div>
              <strong>Privacy focused</strong>
              <p>Designed with responsible AI principles.</p>
            </div>
          </div>

          <button
            className="nav-item"
            onClick={() => setActivePage("library")}
          >
            <Settings size={18} />
            <span>Settings</span>
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-content">
        {/* TOPBAR */}
        <header className="topbar">
          <button
            className="icon-button"
            onClick={() => setSidebarOpen((previous) => !previous)}
            aria-label="Toggle sidebar"
          >
            <Menu size={21} />
          </button>

          <div className="topbar-title">
            <span className="status-dot"></span>
            <span>{pageTitles[activePage]}</span>
          </div>

          <div className="topbar-right">
            <span className="version-label">
              {pageDescriptions[activePage]}
            </span>
          </div>
        </header>

        {/* DASHBOARD PAGE */}
        {activePage === "dashboard" && <Dashboard />}

        {/* MEDICAL ASSISTANT PAGE */}
        {activePage === "assistant" && (
          <>
            <section className="chat-area">
              {messages.length === 0 ? (
                <div className="welcome-screen">
                  <div className="welcome-icon">
                    <Sparkles size={27} />
                  </div>

                  <p className="eyebrow">
                    YOUR MEDICAL KNOWLEDGE COMPANION
                  </p>

                  <h1>
                    Understand health.
                    <br />
                    <span>Explore knowledge.</span>
                  </h1>

                  <p className="welcome-description">
                    Ask medical knowledge questions and explore educational
                    information supported by retrieved medical references.
                  </p>

                  <div className="suggestion-grid">
                    {suggestions.map((item) => {
                      const Icon = item.icon;

                      return (
                        <button
                          className="suggestion-card"
                          key={item.title}
                          onClick={() => setQuestion(item.question)}
                        >
                          <div className="suggestion-icon">
                            <Icon size={20} />
                          </div>

                          <strong>{item.title}</strong>
                          <p>{item.question}</p>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <div className="messages-list">
                  {messages.map((message) => (
                    <div className="message-row" key={message.id}>
                      <div className="user-avatar">
                        {message.role === "user" ? (
                          "You"
                        ) : (
                          <Bot size={19} />
                        )}
                      </div>

                      <div className="message-content">
                        <strong>
                          {message.role === "user"
                            ? "You"
                            : "MedIntel-AI"}
                        </strong>

                        {message.loading ? (
                          <div className="loading-message">
                            <LoaderCircle
                              size={17}
                              className="loading-icon"
                            />
                            <span>
                              Searching medical sources and preparing an
                              answer...
                            </span>
                          </div>
                        ) : (
                          <>
                            {message.role === "user" ? (
                              <p className="answer-text">
                                {message.content}
                              </p>
                            ) : (
                              <div className="medical-answer">
                                {message.questionType && (
                                  <div className="question-type">
                                    {message.questionType}
                                  </div>
                                )}

                                <div className="quick-answer">
                                  <h3>Quick answer</h3>
                                  <p>{message.content}</p>
                                </div>

                                {renderVisualOverview(message.sections)}

                                {message.sections?.length > 0 && (
                                  <div className="sections-container">
                                    {message.sections.map(renderSection)}
                                  </div>
                                )}

                                {message.evidence?.length > 0 && (
                                  <div className="evidence-section">
                                    <h3>
                                      <ShieldCheck size={17} />
                                      Evidence and supporting claims
                                    </h3>

                                    {message.evidence.map(renderEvidence)}
                                  </div>
                                )}

                                {message.sources?.length > 0 && (
                                  <details className="sources-section">
                                    <summary className="sources-heading">
                                      <BookOpen size={16} />
                                      Medical references ({message.sources.length})
                                    </summary>

                                    <div className="sources-list">
                                      {message.sources.map(renderSource)}
                                    </div>
                                  </details>
                                )}

                                {message.safety &&
                                  (
                                    typeof message.safety === "string" ||
                                    (
                                      typeof message.safety === "object" &&
                                      Object.keys(message.safety).length > 0
                                    )
                                  ) && (
                                    <div className="safety-section">
                                      <h3>
                                        <ShieldCheck size={17} />
                                        Safety and limitations
                                      </h3>

                                      {typeof message.safety === "string" ? (
                                        <p>{message.safety}</p>
                                      ) : (
                                        Object.entries(message.safety).map(
                                          ([key, value]) => {
                                            if (
                                              value === null ||
                                              value === undefined ||
                                              value === false ||
                                              value === ""
                                            ) {
                                              return null;
                                            }

                                            return (
                                              <p key={key}>
                                                <strong>
                                                  {key
                                                    .replace(/_/g, " ")
                                                    .replace(
                                                      /\b\w/g,
                                                      (char) =>
                                                        char.toUpperCase()
                                                    )}
                                                  :{" "}
                                                </strong>

                                                {displayValue(value)}
                                              </p>
                                            );
                                          }
                                        )
                                      )}
                                    </div>
                                  )}

                                {message.status === "abstained" && (
                                  <div className="api-warning">
                                    <AlertCircle size={16} />
                                    <span>
                                      The available medical evidence is
                                      insufficient to provide a reliable
                                      answer.
                                    </span>
                                  </div>
                                )}

                                {message.status === "error" && (
                                  <div className="api-warning">
                                    <AlertCircle size={16} />
                                    <span>{message.content}</span>
                                  </div>
                                )}
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* CHAT INPUT */}
            <footer className="input-section">
              <div className="chat-input-box">
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a medical knowledge question..."
                  rows={1}
                  disabled={loading}
                />

                <button
                  className="send-button"
                  onClick={() => sendMessage()}
                  disabled={!question.trim() || loading}
                  aria-label="Send message"
                >
                  {loading ? (
                    <LoaderCircle
                      size={19}
                      className="loading-icon"
                    />
                  ) : (
                    <Send size={19} />
                  )}
                </button>
              </div>

              <div className="input-caption">
                <Search size={13} />
                <span>
                  Medical knowledge retrieval · Evidence-grounded answers
                </span>
              </div>

              <p className="disclaimer">
                <ShieldCheck size={13} />
                MedIntel-AI provides educational information, not medical
                diagnosis or personalized treatment. Consult a qualified
                healthcare professional for medical advice.
              </p>
            </footer>
          </>
        )}

        {/* MEDICAL LIBRARY PAGE */}
        {activePage === "library" && (
          <section className="chat-area">
            <div className="welcome-screen">
              <div className="welcome-icon">
                <BookOpen size={27} />
              </div>

              <p className="eyebrow">
                MEDICAL KNOWLEDGE LIBRARY
              </p>

              <h1>
                Explore medical.
                <br />
                <span>Research and references.</span>
              </h1>

              <p className="welcome-description">
                Explore medical documents and evidence sources used by
                MedIntel-AI. Document browsing and management will be
                implemented in the next module.
              </p>

              <button
                className="suggestion-card"
                onClick={() => setActivePage("assistant")}
              >
                <div className="suggestion-icon">
                  <MessageSquare size={20} />
                </div>

                <strong>Ask Medical AI</strong>

                <p>
                  Ask questions about medical knowledge and retrieved
                  references.
                </p>

                <ChevronRight size={17} />
              </button>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
