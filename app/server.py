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
    from calmline.gemini import available
    if available():
        async def _warm_prefaces():
            for k in PREFACES:
                try:
                    await preface_audio(k)
                except Exception as e:  # noqa: BLE001
                    print("preface warm-up failed:", str(e)[:100], flush=True)
        asyncio.create_task(_warm_prefaces())


app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "static")), name="static")
PIPE = None
ROOMS: dict[str, dict] = {}


def pipeline():
    global PIPE
    if PIPE is None:
        PIPE = build(os.getenv("CALMLINE_VERSION", BEST))
        PIPE.detector("warm up the models"); PIPE.intent_clf("warm up")
    return PIPE


EVIDENCE = os.path.join(os.path.dirname(ROOT), "evidence")


SESSIONS: dict = {}


def new_session():
    p = pipeline()
    if hasattr(p, "context"):
        p.context = ""
    sess = CallSession(p, p.detector, Settings(delay_s=float(os.getenv("CALMLINE_DELAY", "2.0"))), evidence_dir=EVIDENCE)
    SESSIONS[sess.id] = sess
    return sess
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
    r["utt_seq"] = 0; r["emitted"] = 0; r["voice_end"] = 0; r["lag"] = {}; r["streamed"] = {}; r["utt_end"] = {}
    await broadcast(r, {"type": "reset", "session": r["session"].id})
    return r


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "static", "console.html"))


@app.get("/caller")
def caller():
    return FileResponse(os.path.join(ROOT, "static", "caller.html"))


@app.get("/demo")
def demo():
    return FileResponse(os.path.join(ROOT, "static", "console.html"))


@app.get("/split")
def split():
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


async def _live(r, pace="normal"):
    """Gemini Live readers per room (normal pace + fast catch-up pace), connected ahead of time so the
    0.7 s connect is off the critical path."""
    from calmline.live_voice import LiveReader
    key = "live" if pace == "normal" else "live_fast"
    if r.get(key) is None:
        r[key] = LiveReader(pace=pace)
        try:
            await r[key].connect()
        except Exception as e:  # noqa: BLE001
            print("live connect failed:", str(e)[:120], flush=True)
            r[key] = False
    if r.get("live_fast") is None and pace == "normal":
        asyncio.create_task(_live(r, "fast"))
    if r.get("live") is None and pace == "fast":
        asyncio.create_task(_live(r, "normal"))
    return r[key] or None


PREFACES = {"upset": "The caller is upset. One moment.", "wait": "One moment."}
_PREFACE_PCM: dict = {}


async def preface_audio(key):
    """Short lexical fillers, rendered once with the calm voice and kept in memory, so they play with zero latency.
    Lexical fillers mitigate perceived delay better than artificial wait indicators (virtual-agent / HRI studies)."""
    if key in _PREFACE_PCM:
        return _PREFACE_PCM[key]
    path = os.path.join(os.path.dirname(ROOT), "data", "cache", f"preface_{key}.npy")
    if os.path.exists(path):
        _PREFACE_PCM[key] = np.load(path); return _PREFACE_PCM[key]
    from calmline.live_voice import LiveReader
    rd = LiveReader(pace="fast")
    try:
        chunks = [c async for c in rd.speak(PREFACES[key])]
    finally:
        await rd.close()
    x = np.concatenate(chunks).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, x)
    _PREFACE_PCM[key] = x
    return x


async def play_preface(r, s, key, idx):
    """Fill the silence while Gemini prepares an abusive part: goes through the voice queue like any other audio."""
    x = await preface_audio(key)
    async with r.setdefault("voice_lock", asyncio.Lock()):
        t = time.time()
        await broadcast(r, {"type": "preface", "idx": idx, "text": PREFACES[key]})
        for k in range(0, len(x), 3200):
            ch = x[k:k + 3200]
            s.voice_log.append((time.time() - (r["started"] or time.time()), ch))
            r["voice_end"] = max(r.get("voice_end", 0), time.time()) + len(ch) / SR
            await broadcast(r, b"\x03" + (np.clip(ch, -1, 1) * 32767).astype("<i2").tobytes())
        if idx in r.get("lag", {}) or True:
            r.setdefault("preface_at", {})[idx] = t


async def speak(r, s, text, idx=None):
    """Speak one verified calm sentence to the agents (binary prefix 2).
    Gemini Live verbatim reader (~0.7 s to first audio) -> streaming Gemini TTS -> browser voice on the agent page."""
    import queue, threading
    t_req = time.time()
    gen = r.get("voice_gen", 0)
    async with r.setdefault("voice_lock", asyncio.Lock()):
        t_lock = time.time(); first = True
        if r.get("voice_gen", 0) != gen:      # agent pressed "Skip to now" while this sentence was queued
            return

        async def emit(chunk):
            nonlocal first
            if r.get("voice_gen", 0) != gen:
                return          # skipped by the agent: drain the reader silently, play nothing
            if first:
                first = False
                info = dict(queued_ms=round((t_lock - t_req) * 1000), first_audio_ms=round((time.time() - t_lock) * 1000), via=via)
                if idx in r.get("lag", {}):
                    end_wall, lag = r["lag"][idx]
                    lag.update(wait_previous_voice_ms=info["queued_ms"], voice_first_sound_ms=info["first_audio_ms"])
                    info["lag_ms"] = round((time.time() - end_wall) * 1000)   # caller finished -> first calm sound leaves the server
                    info["breakdown"] = lag
                    print("LAG", idx, info["lag_ms"], lag, flush=True)
                s.audit.append([time.time(), "voice_first_audio", info]); print("TIMING voice", info, flush=True)
                await broadcast(r, {"type": "voice_meta", "idx": idx, **info})
            if idx is not None:
                s.voice_by_idx = getattr(s, "voice_by_idx", {}); s.voice_by_idx.setdefault(idx, []).append(chunk)
            s.voice_log.append((time.time() - (r["started"] or time.time()), chunk))
            r["voice_end"] = max(r.get("voice_end", 0), time.time()) + len(chunk) / SR   # estimated end of agent playback
            await broadcast(r, b"\x02" + (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())

        # adaptive pace (decided after the previous sentence's audio has been fully generated):
        # if the agent is still hearing earlier sentences, read this one faster to catch up
        # Live streams roughly in real time, so a sentence that had to queue behind the previous one is "behind"
        backlog = max(r.get("voice_end", 0) - time.time(), t_lock - t_req)
        pace = "fast" if (backlog > 0.8 or os.getenv("CALMLINE_PACE", "fast") == "fast") else "normal"
        via = "live (fast pace)"
        try:
            pool = await _reader_pool(r)
            live = pool[r.get("pool_i", 0) % 2]; r["pool_i"] = r.get("pool_i", 0) + 1
        except Exception as e:  # noqa: BLE001
            live = None
            print("pool failed:", str(e)[:120], flush=True)
        if live is not None:
            try:
                async for ch in live.speak(text):
                    await emit(ch)
                return
            except Exception as e:  # noqa: BLE001
                print("live failed:", str(e)[:120], flush=True)
                if not first:      # drifted mid-sentence: stop, show the text, don't re-speak
                    await broadcast(r, {"type": "voice_cut", "text": text}); return
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


async def _turn(r, seq):
    """Wait until every earlier sentence has been emitted (sentences are processed in parallel, emitted in order)."""
    cond = r.setdefault("emit_cond", asyncio.Condition())
    async with cond:
        await cond.wait_for(lambda: r.get("emitted", 0) >= seq)


async def _done(r):
    cond = r.setdefault("emit_cond", asyncio.Condition())
    async with cond:
        r["emitted"] = r.get("emitted", 0) + 1
        cond.notify_all()


async def _reader_pool(r):
    """Two calm-voice readers used alternately: Live generates at ~1.0x real time, so while one reads phrase k the other
    already generates phrase k+1 and there is no gap between phrases."""
    from calmline.live_voice import LiveReader
    if not r.get("pool"):
        r["pool"] = [LiveReader(pace="fast"), LiveReader(pace="fast")]
        await asyncio.gather(*[x.connect() for x in r["pool"]], return_exceptions=True)
        r["pool_i"] = 0
    return r["pool"]


async def phrase_relay(r, s, text, seq, lag, end_wall):
    """v19 incremental relay: stream Gemini's JSON, cut the calm message into phrases as it is written, verify each
    phrase, and start speaking the first phrase before Gemini has finished (SpeakStream-style text-streaming TTS).
    Returns True if the part was spoken this way."""
    import re as _re
    from calmline.understand import _partial_calm, _early
    from calmline.textnorm import numbers
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    sent = {"n": 0, "flags": None}

    def on_partial(raw):
        if sent["flags"] is None:
            e = _early(raw) if '"calm_message"' in raw else None
            m = _re.search(r'"nothing_to_relay"\s*:\s*(true|false)', raw)
            t = _re.search(r'"threat"\s*:\s*(true|false)', raw)
            h = _re.search(r'"self_harm"\s*:\s*(true|false)', raw)
            if m and t and h:
                sent["flags"] = dict(threat=t.group(1) == "true", self_harm=h.group(1) == "true", nothing=m.group(1) == "true")
                loop.call_soon_threadsafe(q.put_nowait, ("flags", sent["flags"]))
        calm, done = _partial_calm(raw)
        if calm is None:
            return
        rest = calm[sent["n"]:]
        cuts = [mm.end() for mm in _re.finditer(r"[.,;!?](?=\s)", rest)]
        cut = None
        for c in cuts:
            if len(rest[:c].split()) >= 4:
                cut = c
        if done:
            cut = len(rest)
        if cut:
            ph = rest[:cut].strip()
            sent["n"] += cut
            if ph:
                loop.call_soon_threadsafe(q.put_nowait, ("phrase", ph))
        if done and sent.get("closed") is None:
            sent["closed"] = True
            loop.call_soon_threadsafe(q.put_nowait, ("end", None))

    ctx = s.current_context()
    s.ctx_for = getattr(s, "ctx_for", {}); s.ctx_for[seq] = ctx
    und = s.pipe.understanding
    fut = loop.run_in_executor(None, lambda: und.analyze_stream(text, ctx, None, on_partial))
    pool = await _reader_pool(r)
    phrases, gens = [], []

    async def gen(ph, k):
        rd = pool[r.get("pool_i", 0) % 2]; r["pool_i"] = r.get("pool_i", 0) + 1
        out: asyncio.Queue = asyncio.Queue()
        async def run():
            try:
                async for ch in rd.speak(ph):
                    await out.put(ch)
            except Exception as e:  # noqa: BLE001
                await out.put(e)
            await out.put(None)
        asyncio.create_task(run())
        return out

    # wait for the safety flags (they precede calm_message in the JSON)
    try:
        kind, val = await asyncio.wait_for(q.get(), timeout=float(os.getenv("CALMLINE_GEMINI_BUDGET", "2.5")))
    except asyncio.TimeoutError:
        return False
    if kind != "flags" or val["threat"] or val["self_harm"] or val["nothing"]:
        return False
    # wait our turn in the voice order, but start generating phrases immediately
    turn_task = asyncio.create_task(_turn(r, seq))
    k = 0
    pending = []
    first = True
    t_lock = None
    while True:
        kind, val = await q.get()
        if kind == "end":
            break
        if kind != "phrase":
            continue
        bad = s.det(val).is_toxic or (numbers(val) - numbers(text) - {"311"})
        if bad:
            lag["phrase_rejected"] = val[:60]
            break
        pending.append((val, await gen(val, k))); k += 1
        if first:
            first = False
            await turn_task
            await r.setdefault("voice_lock", asyncio.Lock()).acquire()
            t_lock = time.time()
            asyncio.create_task(_playout(r, s, seq, pending, lag, end_wall, t_lock))
    if first:          # nothing spoken (rejected first phrase / empty)
        if not turn_task.done():
            turn_task.cancel()
        return False
    pending.append(None)     # end marker for playout
    r.setdefault("streamed", {})[seq] = True
    return True


async def _playout(r, s, seq, pending, lag, end_wall, t_lock):
    """Play phrase audio in order (holds the voice lock until the whole part has been sent)."""
    try:
        i, first = 0, True
        while True:
            while i >= len(pending):
                await asyncio.sleep(0.02)
            item = pending[i]
            if item is None:
                break
            ph, qu = item
            while True:
                ch = await qu.get()
                if ch is None or isinstance(ch, Exception):
                    break
                if first:
                    first = False
                    info = dict(queued_ms=0, first_audio_ms=round((time.time() - t_lock) * 1000), via="live (phrase stream)",
                                lag_ms=round((time.time() - end_wall) * 1000))
                    lag.update(wait_previous_voice_ms=0, voice_first_sound_ms=info["first_audio_ms"])
                    info["breakdown"] = lag
                    print("LAG", seq, info["lag_ms"], lag, flush=True)
                    await broadcast(r, {"type": "voice_meta", "idx": seq, **info})
                s.voice_log.append((time.time() - (r["started"] or time.time()), ch))
                s.voice_by_idx = getattr(s, "voice_by_idx", {}); s.voice_by_idx.setdefault(seq, []).append(ch)
                r["voice_end"] = max(r.get("voice_end", 0), time.time()) + len(ch) / SR
                await broadcast(r, b"\x02" + (np.clip(ch, -1, 1) * 32767).astype("<i2").tobytes())
            i += 1
    finally:
        r["voice_lock"].release()


async def process_utt(r, s, t0, t1, detected_at=None, seq=0):
    """Pass 2 for one sentence: ASR -> pipeline (Gemini + checks) -> card -> calm voice.
    Sentences run in parallel (a slow Gemini answer no longer holds up the next sentence); cards are emitted in order."""
    loop = asyncio.get_running_loop()
    end_wall = (r["started"] or time.time()) + t1          # when the caller finished this sentence (wall clock)
    lag = {"sentence_end_detect_ms": round(((detected_at or time.time()) - end_wall) * 1000)}
    tq = time.time()
    text, asr_ms = await loop.run_in_executor(None, s.transcribe, t0, t1)
    if text and s.s.protection == "calm_voice" and r.get("filler", True):
        from calmline.pipeline import needs_gemini
        tox = await loop.run_in_executor(None, s.det, text)
        gemini_bound = tox.is_toxic or needs_gemini(text)
        idle = r.get("voice_end", 0) < time.time() + 0.15 and r.get("emitted", 0) >= seq and not tox.is_threat
        streamed_task = None
        if gemini_bound and os.getenv("CALMLINE_PHRASE_STREAM", "1") == "1" and not tox.is_threat:
            r.setdefault("lag", {})[seq] = (end_wall, lag)
            streamed_task = asyncio.create_task(phrase_relay(r, s, text, seq, lag, end_wall))
        async def _card_after_stream():
            if streamed_task is not None:
                # the stream caches Gemini's answer under the same prompt, so the card is built without a second call
                await asyncio.wait({streamed_task}, timeout=6)
            return await loop.run_in_executor(None, s.card_from_text, text, t0, t1, asr_ms, seq)
        card_fut = asyncio.ensure_future(_card_after_stream())
        if gemini_bound and idle:
            # filler only if Gemini has not answered within 0.8 s (fillers help with long waits, cost time on short ones)
            t_f = time.time()
            while time.time() - t_f < 0.8 and not card_fut.done() and r.get("voice_end", 0) < time.time():
                await asyncio.sleep(0.05)
            if not card_fut.done() and r.get("voice_end", 0) < time.time() + 0.15:
                asyncio.create_task(play_preface(r, s, "wait", seq))
                lag["preface_ms"] = round((time.time() - end_wall) * 1000)
                await asyncio.sleep(0)
        card = await card_fut
    else:
        card = await loop.run_in_executor(None, s.card_from_text, text, t0, t1, asr_ms, seq)
    t_wait = time.time()
    await _turn(r, seq)
    lag["wait_previous_sentence_ms"] = round((time.time() - t_wait) * 1000)
    if not card:
        await broadcast(r, {"type": "utt_empty"}); await _done(r)
        return
    if True:
        card["timing"]["queued_ms"] = 0
        lag.update(asr_ms=card["timing"]["asr_ms"], gemini_and_checks_ms=card["timing"]["rewrite_pipeline_ms"])
        r.setdefault("lag", {})[card["idx"]] = (end_wall, lag)
        asyncio.create_task(late_update(r, s, card["idx"]))
        print("TIMING card", card["timing"], "speech_end", round(t1, 2), "emitted_at_call_s", round(time.time() - (r["started"] or time.time()), 2), flush=True)
        if card["t"][1] > 0 and r["started"]:
            card["ready_lag_ms"] = round((time.time() - (r["started"] + card["t"][1])) * 1000)
        await broadcast(r, card)
        if card.get("route") == "ESCALATE":
            await broadcast(r, {"type": "alert", "label": "threat"})
    if card.get("withheld") and card.get("route") != "ESCALATE":
        await _done(r)
        return
    if r.get("streamed", {}).get(card["idx"]):
        await _done(r)
        return
    if s.s.protection == "calm_voice":
        r.setdefault("utt_end", {})[card["idx"]] = card["t"][1]
        text = card["delivered"].split("] ", 1)[-1] if card["delivered"].startswith("[SAFETY") else card["delivered"]
        if card["route"] == "ESCALATE":
            text = "Safety alert. A supervisor is joining. " + text
        task = asyncio.create_task(speak(r, s, text, card.get("idx")))
        await asyncio.sleep(0)          # let speak() join the voice queue before the next sentence does
        await _done(r)
        await task
    else:
        await _done(r)


async def late_update(r, s, idx):
    """Gemini answered after the latency budget: refresh the complaint type and facts on screen (no re-speaking)."""
    fut = getattr(s, "late", {}).pop(idx, None)
    if fut is None:
        return
    try:
        u = await asyncio.wrap_future(fut)
    except Exception:  # noqa: BLE001
        return
    if not u:
        return
    from calmline.intents import INTENTS_ALL
    from calmline.understand import TO_REGEX
    it = u.get("intent", "UNKNOWN")
    await broadcast(r, {"type": "card_update", "idx": idx, "intent": it, "intent_score": round(float(u.get("intent_confidence") or 0), 2),
                        "intent_label": INTENTS_ALL[it]["issue"] if it in INTENTS_ALL else None,
                        "nyc_types": INTENTS_ALL[it]["nyc"] if it in INTENTS_ALL else [],
                        "slots": {TO_REGEX.get(k, k): [v] for k, v in (u.get("slots") or {}).items()}})


async def process_typed(r, s, text):
    loop = asyncio.get_running_loop()
    async with r.setdefault("card_lock", asyncio.Lock()):
        s._typed_seq = len(s.cards)
        card = await loop.run_in_executor(None, s.process_text, text)
        await broadcast(r, {"type": "typed", "text": text})
        await broadcast(r, card)
        asyncio.create_task(late_update(r, s, card["idx"]))
        if card.get("route") == "ESCALATE":
            await broadcast(r, {"type": "alert", "label": "threat"})
    if card.get("withheld") and card.get("route") != "ESCALATE":
        return
    if s.s.protection == "calm_voice":
        t = card["delivered"].split("] ", 1)[-1]
        if card["route"] == "ESCALATE":
            t = "Safety alert. A supervisor is joining. " + t
        await speak(r, s, t, card.get("idx"))


@app.get("/api/voice/{sid}/{idx}")
def voice(sid: str, idx: int):
    """Replay the calm voice of one relayed sentence."""
    import io, soundfile as sf
    sess = SESSIONS.get(sid)
    chunks = getattr(sess, "voice_by_idx", {}).get(idx) if sess else None
    if not chunks:
        return Response(status_code=404)
    b = io.BytesIO(); sf.write(b, np.concatenate(chunks), SR, format="WAV", subtype="PCM_16")
    return Response(b.getvalue(), media_type="audio/wav")


def on_events(r, s, events):
    for e in events:
        if e.get("type") == "utt":
            e["detected_at"] = time.time()
            seq = r.get("utt_seq", 0); r["utt_seq"] = seq + 1
            r.setdefault("pending", []).append(asyncio.create_task(process_utt(r, s, e["t0"], e["t1"], e["detected_at"], seq)))


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
    r = room(name)
    if r["session"].closed:
        r = await reset_room(name)
    s = r["session"]
    asyncio.create_task(_reader_pool(r))
    clock_started = False
    buf = np.zeros(0, np.float32)
    loop = asyncio.get_running_loop()
    chunk = int(s.s.chunk_s * SR)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                if not clock_started:
                    clock_started = True
                    r["started"] = time.time()
                    asyncio.create_task(playback_loop(name))
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
                if m.get("type") == "text" and m.get("text", "").strip():
                    asyncio.create_task(process_typed(r, s, m["text"].strip()[:600]))
                    continue
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
            elif m.get("type") == "filler":
                r["filler"] = bool(m.get("value")); s.audit.append([time.time(), "filler", r["filler"]])
            elif m.get("type") == "skip_to_now":
                r["voice_gen"] = r.get("voice_gen", 0) + 1; r["voice_end"] = time.time()
                s.audit.append([time.time(), "skip_to_now"]); await broadcast(r, {"type": "skipped"})
            elif m.get("type") == "end_call_request":
                s.audit.append([time.time(), "end_call_request"]); await broadcast(r, {"type": "end_call_ack"})
            elif m.get("type") == "reveal":
                s.audit.append([time.time(), "reveal", m.get("idx")])   # every reveal of an original is audited
    except WebSocketDisconnect:
        r["agents"].discard(ws)


@app.post("/api/reset/{name}")
async def reset(name: str):
    await reset_room(name)
    return {"ok": True}


@app.websocket("/ws/sample/{name}/{sid}")
async def ws_sample(ws: WebSocket, name: str, sid: str):
    """Server-side playback of a sample call into a room (for judges without a microphone)."""
    import soundfile as sf
    await ws.accept()
    r = await reset_room(name); s = r["session"]
    asyncio.create_task(_reader_pool(r))
    try:   # start the clock when the browser actually starts playing the caller audio, so the timeline matches what you hear
        await asyncio.wait_for(ws.receive_text(), timeout=8)
    except Exception:
        pass
    r["started"] = time.time()
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
