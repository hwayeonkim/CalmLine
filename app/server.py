"""CalmLine demo server (FastAPI + WebSocket). Cloud Run-ready: listens on $PORT.

/                 landing page
/caller           caller console (microphone or sample calls)
/agent            agent console (delayed, filtered audio + calm cards)
/ws/caller/{room} caller PCM16 @16 kHz in
/ws/agent/{room}  agent: events + PCM16 out
/api/samples      list of sample calls (pre-rendered WAVs)
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from calmline.session import CallSession, Settings, SR
from calmline.versions import build, BEST

ROOT = os.path.dirname(os.path.abspath(__file__))
app = FastAPI(title="CalmLine")
@app.on_event("startup")
async def warm():
    loop = asyncio.get_running_loop()
    def _w():
        p = pipeline()
        from calmline.audio.stt import load_whisper
        load_whisper(Settings().whisper); load_whisper(Settings().spotter)
    await loop.run_in_executor(None, _w)


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")
PIPE = None
ROOMS: dict[str, dict] = {}


def pipeline():
    global PIPE
    if PIPE is None:
        PIPE = build(os.getenv("CALMLINE_VERSION", BEST))
        PIPE("warm up the models")
    return PIPE


EVIDENCE = os.path.join(os.path.dirname(ROOT), "evidence")


def new_session():
    p = pipeline()
    return CallSession(p, p.detector, Settings(delay_s=float(os.getenv("CALMLINE_DELAY", "2.0"))), evidence_dir=EVIDENCE)


def room(name):
    if name not in ROOMS:
        ROOMS[name] = dict(session=new_session(), agents=set(), started=None, lock=asyncio.Lock())
    return ROOMS[name]


async def reset_room(name):
    """Start a fresh call in the room but keep connected agents."""
    r = room(name)
    old = r["session"]; old.closed = True
    prot = old.s.protection
    r["session"] = new_session(); r["session"].s.protection = prot
    r["started"] = None
    await broadcast(r, {"type": "reset", "session": r["session"].id})
    return r


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "index.html"))


@app.get("/caller")
def caller():
    return FileResponse(os.path.join(ROOT, "static", "caller.html"))


@app.get("/demo")
def demo():
    return FileResponse(os.path.join(ROOT, "static", "demo.html"))


@app.get("/agent")
def agent():
    return FileResponse(os.path.join(ROOT, "static", "agent.html"))


@app.get("/api/samples")
def samples():
    path = os.path.join(ROOT, "static", "samples", "samples.json")
    return JSONResponse(json.load(open(path)) if os.path.exists(path) else [])


@app.get("/api/compare/{name}")
def compare(name: str):
    """Demo only: original vs. what the agent heard, for the last finished call in the room.
    (In production the original is sealed and only a supervisor can open it.)"""
    r = ROOMS.get(name)
    sid = r and r.get("last_done")
    if not sid:
        return JSONResponse({"ready": False})
    return {"ready": True, "session": sid, "original": f"/api/evidence/{sid}/caller_original.wav",
            "protected": f"/api/evidence/{sid}/agent_heard.wav"}


@app.get("/api/evidence/{sid}/{fname}")
def evidence(sid: str, fname: str):
    if fname not in ("caller_original.wav", "agent_heard.wav") or not sid.isalnum():
        return Response(status_code=404)
    return FileResponse(os.path.join(EVIDENCE, sid, fname), media_type="audio/wav")


@app.get("/api/tts")
def tts(text: str):
    from calmline import tts as T
    try:
        wav = T.synthesize(text[:600])
    except Exception as e:
        return JSONResponse({"error": str(e)[:200]}, status_code=502)
    if wav is None:
        return JSONResponse({"error": "no Google TTS backend configured (set GEMINI_API_KEY)"}, status_code=501)
    return Response(wav, media_type="audio/wav")


@app.get("/api/health")
def health():
    return {"ok": True, "version": os.getenv("CALMLINE_VERSION", BEST)}


async def finish(r, s):
    """End of call: wait for outstanding sentences and voice, then seal the evidence."""
    s.closed = True
    pend = r.get("pending", [])
    if pend:
        await asyncio.gather(*pend, return_exceptions=True)
    r["pending"] = []
    async with r.setdefault("voice_lock", asyncio.Lock()):
        s.save_evidence()
    r["last_done"] = s.id
    await broadcast(r, {"type": "compare_ready", "session": s.id})


async def broadcast(r, msg: dict | bytes):
    dead = []
    for ws in list(r["agents"]):
        try:
            if isinstance(msg, bytes):
                await ws.send_bytes(msg)
            else:
                await ws.send_text(json.dumps(msg, default=float))
        except Exception:
            dead.append(ws)
    for ws in dead:
        r["agents"].discard(ws)


async def _live(r):
    """One Gemini Live reader per room, connected ahead of time so the 0.7 s connect is off the critical path."""
    from calmline.live_voice import LiveReader
    if r.get("live") is None:
        r["live"] = LiveReader()
        try:
            await r["live"].connect()
        except Exception as e:  # noqa: BLE001
            print("live connect failed:", str(e)[:120], flush=True)
            r["live"] = False
    return r["live"] or None


async def speak(r, s, text):
    """Speak one verified calm sentence to the agents (binary prefix 2).
    Gemini Live verbatim reader (~0.7 s to first audio) -> streaming Gemini TTS -> browser voice on the agent page."""
    import queue, threading
    t_req = time.time()
    async with r.setdefault("voice_lock", asyncio.Lock()):
        t_lock = time.time(); first = True

        async def emit(chunk):
            nonlocal first
            if first:
                first = False
                info = dict(queued_ms=round((t_lock - t_req) * 1000), first_audio_ms=round((time.time() - t_lock) * 1000), via=via)
                s.audit.append([time.time(), "voice_first_audio", info]); print("TIMING voice", info, flush=True)
            s.voice_log.append((time.time() - (r["started"] or time.time()), chunk))
            await broadcast(r, b"\x02" + (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())

        via = "live"
        live = await _live(r)
        if live is not None:
            try:
                async for ch in live.speak(text):
                    await emit(ch)
                return
            except Exception as e:  # noqa: BLE001
                print("live failed:", str(e)[:120], flush=True)
                if not first:      # drifted mid-sentence: stop, show the text, don't re-speak
                    await broadcast(r, {"type": "voice_cut", "text": text}); return
                await live.close(); r["live"] = None
        via = "tts"
        if r.get("tts_down") and time.time() < r["tts_down"]:
            await broadcast(r, {"type": "voice_fallback", "text": text}); return
        from calmline import tts
        q: queue.Queue = queue.Queue()
        def work():
            try:
                for ch in tts.stream(text):
                    q.put(ch)
            except Exception as e:  # noqa: BLE001
                q.put(e)
            q.put(None)
        threading.Thread(target=work, daemon=True).start()
        while True:
            item = await asyncio.get_running_loop().run_in_executor(None, q.get)
            if item is None:
                break
            if isinstance(item, Exception):
                r["tts_down"] = time.time() + 600
                await broadcast(r, {"type": "voice_fallback", "text": text, "error": str(item)[:160]})
                break
            await emit(item)


async def process_utt(r, s, t0, t1):
    """Pass 2 for one sentence: ASR -> pipeline (Gemini rewrite + checks) -> card -> calm voice. Cards stay in order."""
    loop = asyncio.get_running_loop()
    async with r.setdefault("card_lock", asyncio.Lock()):
        tq = time.time()
        card = await loop.run_in_executor(None, s.process_utterance, t0, t1)
        if not card:
            return
        card["timing"]["queued_ms"] = round((time.time() - tq) * 1000) - card["timing"]["asr_ms"] - card["timing"]["rewrite_pipeline_ms"]
        print("TIMING card", card["timing"], "speech_end", round(t1, 2), "emitted_at_call_s", round(time.time() - (r["started"] or time.time()), 2), flush=True)
        await broadcast(r, card)
        if card.get("route") == "ESCALATE":
            await broadcast(r, {"type": "alert", "label": "threat"})
    if card.get("withheld") and card.get("route") != "ESCALATE":
        return
    if s.s.protection == "calm_voice":
        text = card["delivered"].split("] ", 1)[-1] if card["delivered"].startswith("[SAFETY") else card["delivered"]
        if card["route"] == "ESCALATE":
            text = "Safety alert. A supervisor is joining. " + text
        await speak(r, s, text)


def on_events(r, s, events):
    for e in events:
        if e.get("type") == "utt":
            r.setdefault("pending", []).append(asyncio.create_task(process_utt(r, s, e["t0"], e["t1"])))


async def playback_loop(name):
    """Sends caller audio to agents at real-time pace, `delay_s` behind the caller."""
    r = room(name); s = r["session"]; t_start = r["started"]
    played = 0.0; step = 0.1
    while not s.closed or played < len(s.audio) / SR:
        if r["session"] is not s:
            return
        now = time.time() - t_start
        target = now - s.s.delay_s
        avail = len(s.audio) / SR
        while played + step <= min(target, avail):
            if s.s.protection != "calm_voice":
                x = s.render_playback(played, played + step)
                await broadcast(r, b"\x01" + (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
            played += step
        if s.closed and target >= avail:
            break
        await asyncio.sleep(0.05)
    await broadcast(r, {"type": "ended"})


@app.websocket("/ws/caller/{name}")
async def ws_caller(ws: WebSocket, name: str):
    await ws.accept()
    r = room(name); s = r["session"]
    if r["started"] is None:
        r["started"] = time.time()
        asyncio.create_task(playback_loop(name))
        asyncio.create_task(_live(r))
    buf = np.zeros(0, np.float32)
    loop = asyncio.get_running_loop()
    chunk = int(s.s.chunk_s * SR)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                buf = np.concatenate([buf, np.frombuffer(msg["bytes"], "<i2").astype(np.float32) / 32768])
                while len(buf) >= chunk:
                    part, buf = buf[:chunk], buf[chunk:]
                    async with r["lock"]:
                        events = await loop.run_in_executor(None, s.ingest, part, False)
                    for e in events:
                        await broadcast(r, e)
                    on_events(r, s, events)
            elif msg.get("text"):
                m = json.loads(msg["text"])
                if m.get("type") == "end":
                    break
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    async with r["lock"]:
        events = await loop.run_in_executor(None, s.ingest, buf if len(buf) else np.zeros(1600, np.float32), True)
    for e in events:
        await broadcast(r, e)
    on_events(r, s, events)
    await finish(r, s)


@app.websocket("/ws/agent/{name}")
async def ws_agent(ws: WebSocket, name: str):
    await ws.accept()
    r = room(name); s = r["session"]
    r["agents"].add(ws)
    await ws.send_text(json.dumps({"type": "hello", "session": s.id, "delay": s.s.delay_s, "protection": s.s.protection,
                                   "cards": s.cards}, default=float))
    try:
        while True:
            m = json.loads(await ws.receive_text())
            s = r["session"]
            if m.get("type") == "protection":
                s.s.protection = m["value"]; s.audit.append([time.time(), "protection", m["value"]])
            elif m.get("type") == "end_call_request":
                s.audit.append([time.time(), "end_call_request"]); await broadcast(r, {"type": "end_call_ack"})
            elif m.get("type") == "reveal":
                s.audit.append([time.time(), "reveal", m.get("idx")])   # every reveal of an original is audited
    except WebSocketDisconnect:
        r["agents"].discard(ws)


@app.post("/api/reset/{name}")
def reset(name: str):
    ROOMS.pop(name, None)
    return {"ok": True}


@app.websocket("/ws/sample/{name}/{sid}")
async def ws_sample(ws: WebSocket, name: str, sid: str):
    """Server-side playback of a sample call into a room (for judges without a microphone)."""
    import soundfile as sf
    await ws.accept()
    r = await reset_room(name); s = r["session"]; r["started"] = time.time()
    asyncio.create_task(_live(r))
    asyncio.create_task(playback_loop(name))
    x = sf.read(os.path.join(ROOT, "static", "samples", f"{sid}.wav"))[0].astype(np.float32)
    loop = asyncio.get_running_loop(); chunk = int(s.s.chunk_s * SR)
    for k in range(0, len(x), chunk):
        part = x[k:k + chunk]
        await asyncio.sleep(max(0.0, (k + len(part)) / SR - (time.time() - r["started"])))   # real-time arrival
        async with r["lock"]:
            events = await loop.run_in_executor(None, s.ingest, part, k + chunk >= len(x))
        for e in events:
            await broadcast(r, e)
        on_events(r, s, events)
        try:
            await ws.send_text(json.dumps({"type": "progress", "t": (k + len(part)) / SR, "dur": len(x) / SR}))
        except Exception:
            pass
    await finish(r, s)
    await ws.send_text(json.dumps({"type": "done", "session": s.id}))
