# Frontend integration

`ChatWidget.jsx` is a self-contained floating chat widget
(launcher + panel) that talks to the Flask backend's one real endpoint:
`POST /api/chat`. It has no dependencies beyond React and Tailwind
(already in your project) — no extra npm install needed.

## Drop it in

1. Copy `ChatWidget.jsx` into your components folder.
2. Mount it **once**, near the root (e.g. `App.jsx` or your root
   layout) — it's a fixed-position launcher, so mounting it on every
   page will duplicate the button:

```jsx
import ChatWidget from "./components/ChatWidget";

function App() {
  return (
    <>
      {/* rest of the app */}
      <ChatWidget apiUrl={import.meta.env.VITE_CHAT_API_URL} />
    </>
  );
}
```

(Swap `import.meta.env.VITE_CHAT_API_URL` for `process.env.NEXT_PUBLIC_CHAT_API_URL`
if this is Next.js, or just hardcode the URL for a quick test.)

3. Set the env var to wherever `wsgi.py` ends up running, e.g.:
```
VITE_CHAT_API_URL=https://api.apexcreative.example/api/chat
```
For local dev against `python wsgi.py`, that's `http://localhost:5000/api/chat`.

## What it does on its own

- Generates a session UUID on first load (`crypto.randomUUID()`),
  stores it in `localStorage` — this is the same ID the backend's
  `sessions` table keys on, so a page refresh keeps the same
  conversation (Phase 4).
- Also caches the visible message list in `localStorage` purely for
  the widget's own UI continuity across a refresh — the backend is the
  actual source of truth for stored history, this is just so the
  visitor doesn't see an empty chat window after reloading.
- Shows a typing indicator while waiting, and a plain-language fallback
  if the request fails (network error, backend down) rather than
  breaking silently.

## What it does NOT do

- No streaming — this calls the endpoint and waits for the full
  `answer`. If you want token-by-token streaming later, that's a
  backend change first (the current `answer_query()` call is
  synchronous, not a generator) before the widget could consume it.
- No rate limiting on the frontend — that should live on the backend
  (e.g. per-session-ID or per-IP throttling in the Flask layer) since a
  client-side check is trivially bypassed.

## CORS

Make sure `FRONTEND_ORIGIN` in the backend's `.env` is set to your
real site origin (e.g. `https://apexcreative.example`) once this is live —
`*` is fine for local testing only.