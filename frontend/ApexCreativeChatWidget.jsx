import { useEffect, useRef, useState } from "react";

/**
 * Floating AI chat widget for the Apex Creative site.
 *
 * Talks to the Flask backend's single real endpoint, POST {apiUrl}
 * ({ session_id, message } -> { answer, grounded }) — nothing about the
 * RAG/lead-capture/guardrail pipeline lives here, this component is
 * purely presentation + a fetch call. See app/api/app.py for the
 * contract.
 *
 * Usage:
 *   <ChatWidget apiUrl="https://api.apexcreative.example/api/chat" />
 * Drop it once near the root of the app (e.g. in App.jsx / layout),
 * not per-page — it's a fixed-position launcher, so mounting it more
 * than once will duplicate the button.
 */

const SESSION_STORAGE_KEY = "apexcreative_chat_session_id";
const HISTORY_STORAGE_KEY = "apexcreative_chat_history";

function getOrCreateSessionId() {
  if (typeof window === "undefined") return null;
  let sessionId = window.localStorage.getItem(SESSION_STORAGE_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    window.localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }
  return sessionId;
}

function loadStoredHistory() {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

const GREETING = {
  role: "assistant",
  content:
    "Hi! I'm the Apex Creative AI Assistant — ask me about our services, pricing, or how to get started.",
};

export default function ChatWidget({ apiUrl }) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState(() => {
    const stored = loadStoredHistory();
    return stored.length > 0 ? stored : [GREETING];
  });
  const [draft, setDraft] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [errorText, setErrorText] = useState(null);
  const sessionIdRef = useRef(null);
  const scrollAnchorRef = useRef(null);

  useEffect(() => {
    sessionIdRef.current = getOrCreateSessionId();
  }, []);

  useEffect(() => {
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(messages));
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  async function sendMessage() {
    const text = draft.trim();
    if (!text || isSending) return;

    setDraft("");
    setErrorText(null);
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setIsSending(true);

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionIdRef.current, message: text }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch {
      setErrorText(
        "Something went wrong sending that — please try again, or reach us directly at info@apexcreative.example."
      );
    } finally {
      setIsSending(false);
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
    if (event.key === "Escape") {
      setIsOpen(false);
    }
  }

  return (
    <>
      {/* Launcher */}
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-label={isOpen ? "Close chat" : "Open chat with Apex Creative AI Assistant"}
        className="fixed bottom-5 right-5 z-50 flex h-14 w-14 items-center justify-center rounded-full shadow-lg transition-transform hover:scale-105 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
        style={{
          background: "linear-gradient(135deg, #6650F2 0%, #EC61C4 100%)",
          boxShadow: "0 8px 24px rgba(102, 80, 242, 0.4)",
        }}
      >
        {isOpen ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M6 6L18 18M18 6L6 18" stroke="white" strokeWidth="2" strokeLinecap="round" />
          </svg>
        ) : (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path
              d="M4 5.5C4 4.67 4.67 4 5.5 4h13c.83 0 1.5.67 1.5 1.5v10c0 .83-.67 1.5-1.5 1.5H9l-4 4v-4H5.5C4.67 17 4 16.33 4 15.5v-10z"
              fill="white"
            />
          </svg>
        )}
      </button>

      {/* Panel */}
      {isOpen && (
        <div
          role="dialog"
          aria-label="Apex Creative AI Assistant chat"
          className="fixed bottom-24 right-5 z-50 flex h-[70vh] max-h-[560px] w-[92vw] max-w-sm flex-col overflow-hidden rounded-2xl shadow-2xl sm:w-96"
          style={{ backgroundColor: "#010131", border: "1px solid rgba(102, 80, 242, 0.35)" }}
        >
          {/* Header */}
          <div
            className="flex items-center justify-between px-4 py-3"
            style={{ background: "linear-gradient(135deg, #6650F2 0%, #010131 120%)" }}
          >
            <div>
              <p className="text-sm font-semibold text-white">Apex Creative AI Assistant</p>
              <p className="text-xs text-white/60">Usually replies in a few seconds</p>
            </div>
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              aria-label="Close chat"
              className="rounded-full p-1 text-white/70 hover:text-white focus:outline-none"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M6 6L18 18M18 6L6 18" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
            {messages.map((message, index) => (
              <div
                key={index}
                className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className="max-w-[85%] whitespace-pre-wrap rounded-2xl px-3.5 py-2 text-sm leading-relaxed"
                  style={
                    message.role === "user"
                      ? { background: "#EC61C4", color: "#010131" }
                      : { background: "rgba(255,255,255,0.08)", color: "#F2F0FF" }
                  }
                >
                  {message.content}
                </div>
              </div>
            ))}
            {isSending && (
              <div className="flex justify-start">
                <div
                  className="flex items-center gap-1 rounded-2xl px-3.5 py-2.5"
                  style={{ background: "rgba(255,255,255,0.08)" }}
                >
                  {[0, 1, 2].map((i) => (
                    <span
                      key={i}
                      className="h-1.5 w-1.5 animate-bounce rounded-full"
                      style={{ background: "#EC61C4", animationDelay: `${i * 0.12}s` }}
                    />
                  ))}
                </div>
              </div>
            )}
            {errorText && (
              <p className="rounded-lg bg-red-500/10 px-3 py-2 text-xs text-red-300">{errorText}</p>
            )}
            <div ref={scrollAnchorRef} />
          </div>

          {/* Input */}
          <div className="flex items-end gap-2 border-t border-white/10 p-3">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              placeholder="Ask about services, pricing, timelines..."
              aria-label="Message"
              className="max-h-24 flex-1 resize-none rounded-xl bg-white/5 px-3 py-2 text-sm text-white placeholder-white/40 focus:outline-none focus-visible:ring-1"
              style={{ ["--tw-ring-color"]: "#6650F2" }}
            />
            <button
              type="button"
              onClick={sendMessage}
              disabled={isSending || !draft.trim()}
              aria-label="Send message"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full disabled:opacity-40"
              style={{ background: "linear-gradient(135deg, #6650F2 0%, #EC61C4 100%)" }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path
                  d="M3 11l18-8-8 18-2-8-8-2z"
                  stroke="white"
                  strokeWidth="1.6"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
    </>
  );
}