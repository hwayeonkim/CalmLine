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
from fastapi.responses import FileResponse, JSONResponse
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


def new_session():
    p = pipeline()
    return CallSession(p, p.detector, Settings(delay_s=float(os.getenv("CALMLINE_DELAY", "2.0"))))


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


@app.get("/api/health")
def health():
    return {"ok": True, "version": os.getenv("CALMLINE_VERSION", BEST)}


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
            x = s.render_playback(played, played + step)
            pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
            await broadcast(r, pcm)
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
    s.closed = True
    s.save_evidence()


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
        try:
            await ws.send_text(json.dumps({"type": "progress", "t": (k + len(part)) / SR, "dur": len(x) / SR}))
        except Exception:
            pass
    s.closed = True
    s.save_evidence()
    await ws.send_text(json.dumps({"type": "done"}))
