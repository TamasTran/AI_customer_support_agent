import { useState } from "react";

type Message = { role: "user" | "assistant"; content: string };
type PendingConfirmation = {
  tool: string;
  arguments: Record<string, unknown>;
  summary: string;
  token: string;
  issued_at: number;
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);

  async function postChat(body: Record<string, unknown>) {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.message || `Request failed (${response.status})`);
      }
      const data = await response.json();
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
      setPendingConfirmation(data.pending_confirmation ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function sendMessage() {
    if (!input.trim()) return;
    const userMessage: Message = { role: "user", content: input };
    const nextMessages = [...messages, userMessage];
    setMessages(nextMessages);
    setInput("");
    await postChat({ messages: nextMessages });
  }

  async function respondToConfirmation(confirm: boolean) {
    if (!pendingConfirmation) return;
    await postChat({ messages, pending_confirmation: pendingConfirmation, confirm });
    setPendingConfirmation(null);
  }

  return (
    <div style={{ maxWidth: 640, margin: "40px auto", fontFamily: "system-ui" }}>
      <h1>AI Customer Support Agent</h1>
      <div style={{ border: "1px solid #ccc", borderRadius: 8, padding: 16, minHeight: 300 }}>
        {messages.map((m, i) => (
          <p key={i}>
            <strong>{m.role === "user" ? "You" : "Agent"}:</strong> {m.content}
          </p>
        ))}
        {loading && <p><em>Agent is thinking (local model, this can take a while)...</em></p>}
        {error && <p style={{ color: "red" }}>Error: {error}</p>}
      </div>
      {pendingConfirmation && !loading && (
        <div style={{ marginTop: 12, padding: 12, background: "#fff8e1", borderRadius: 8 }}>
          <p>Confirm: {pendingConfirmation.summary}?</p>
          <button onClick={() => respondToConfirmation(true)}>Yes, proceed</button>{" "}
          <button onClick={() => respondToConfirmation(false)}>No, cancel</button>
        </div>
      )}
      <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
        <input
          style={{ flex: 1, padding: 8 }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Ask about your order..."
          disabled={!!pendingConfirmation}
        />
        <button onClick={sendMessage} disabled={loading || !!pendingConfirmation}>
          Send
        </button>
      </div>
    </div>
  );
}
