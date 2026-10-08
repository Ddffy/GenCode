"""Session-level durable events and replayable run streams."""

import asyncio
import json
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gencode.core.runtime.workspace_context import now


class SessionEventBus:
    def __init__(self, session_id, path, redact=None):
        self.session_id = str(session_id)
        self.path = Path(path)
        self.redact = redact or (lambda value: value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._run_sequences = {}
        self._session_sequence = 0
        self._gaps = {}
        self._gap_history = {}
        self._subscribers = {}
        self._completed_runs = {}
        self._volatile_events = defaultdict(lambda: deque(maxlen=2048))
        self._active_run_id = ""
        self._degradation_handler = None
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gencode-events")
        self._load_sequences()

    def set_degradation_handler(self, handler):
        self._degradation_handler = handler

    def emit(self, event, payload=None):
        future, loop, value = self._submit(event, payload)
        if loop is None:
            record = future.result()
            self._notify_degradation(record, None)
            return record

        def observe(done):
            try:
                record = done.result()
                self._notify_degradation(record, loop)
            except Exception as exc:  # noqa: BLE001 - synchronous emit has no awaiter
                loop.call_soon_threadsafe(
                    loop.call_exception_handler,
                    {"message": "Session event persistence failed", "exception": exc},
                )

        future.add_done_callback(observe)
        return {"event": str(event), "run_id": value.get("run_id", ""), "queued": True}

    def _notify_degradation(self, record, loop):
        if not record.get("event_log_degraded") or self._degradation_handler is None:
            return
        run_id = str(record.get("run_id", ""))
        if run_id:
            self._degradation_handler(run_id, loop)

    async def publish(self, event, payload=None):
        future, _, _ = self._submit(event, payload)
        return await asyncio.shield(asyncio.wrap_future(future))

    def set_active_run(self, run_id):
        with self._lock:
            self._active_run_id = str(run_id or "")

    async def subscribe(self, run_id, after_seq=0):
        run_id = str(run_id)
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, set()).add((loop, queue))
        replay = await self.replay_async(run_id, after_seq)
        high_water = max(
            [int(item.get("run_seq", 0)) for item in replay] + [int(after_seq)]
        )
        try:
            for record in replay:
                yield record
                if record.get("event") == "turn_finished":
                    return
            with self._lock:
                if run_id in self._completed_runs:
                    return
            while True:
                record = await queue.get()
                if int(record.get("run_seq", 0)) > high_water:
                    high_water = int(record["run_seq"])
                    yield record
                    if record.get("event") == "turn_finished":
                        return
        finally:
            with self._lock:
                self._subscribers.get(run_id, set()).discard((loop, queue))

    def replay(self, run_id, after_seq=0):
        persisted = self._writer.submit(
            self._read_run_events, str(run_id), int(after_seq)
        ).result()
        return self._merge_volatile_events(run_id, after_seq, persisted)

    async def replay_async(self, run_id, after_seq=0):
        future = self._writer.submit(
            self._read_run_events, str(run_id), int(after_seq)
        )
        persisted = await asyncio.wrap_future(future)
        return self._merge_volatile_events(run_id, after_seq, persisted)

    def _merge_volatile_events(self, run_id, after_seq, persisted):
        with self._lock:
            volatile = [
                dict(record)
                for record in self._volatile_events.get(str(run_id), ())
                if int(record.get("run_seq", 0)) > int(after_seq)
            ]
        by_sequence = {
            int(record.get("run_seq", 0)): record for record in persisted
        }
        by_sequence.update(
            {int(record.get("run_seq", 0)): record for record in volatile}
        )
        return [by_sequence[seq] for seq in sorted(by_sequence)]

    def degraded(self, run_id):
        with self._lock:
            return dict(self._gap_history.get(str(run_id), {}))

    def is_completed(self, run_id):
        with self._lock:
            return str(run_id) in self._completed_runs

    def _submit(self, event, payload):
        value = dict(payload or {})
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        with self._lock:
            run_id = str(value.get("run_id", "") or self._active_run_id)
            if run_id:
                value["run_id"] = run_id
            future = self._writer.submit(self._record, str(event), value)
        return future, loop, value

    def _record(self, event, payload):
        with self._lock:
            value = dict(payload or {})
            run_id = str(value.get("run_id", "") or self._active_run_id)
            if run_id:
                value["run_id"] = run_id
            source = value.pop("source", "runtime")
            if run_id:
                seq = self._run_sequences.get(run_id, 0) + 1
                self._run_sequences[run_id] = seq
                value["run_seq"] = seq
            else:
                self._session_sequence += 1
                value["session_seq"] = self._session_sequence
            value.update(
                {
                    "event": str(event),
                    "session_id": self.session_id,
                    "source": source,
                    "created_at": now(),
                }
            )
            record = self.redact(value)
            records = []
            gap = self._gaps.get(run_id) if run_id else None
            if gap:
                marker = {
                    "event": "event_log_gap",
                    "session_id": self.session_id,
                    "run_id": run_id,
                    "run_seq": record["run_seq"],
                    "source": "runtime",
                    "created_at": now(),
                    "missing_from": gap["missing_from"],
                    "missing_to": gap["missing_to"],
                }
                records.append(marker)
                record["run_seq"] = int(marker["run_seq"]) + 1
                self._run_sequences[run_id] = record["run_seq"]
                record["event_log_degraded"] = True
                record["event_log_gap_recovered"] = True
                record["event_log_gap"] = dict(gap)
            records.append(record)
            persisted = True
            try:
                with self.path.open("a", encoding="utf-8") as file:
                    for item in records:
                        file.write(json.dumps(item, sort_keys=True) + "\n")
                    file.flush()
                if gap:
                    self._gaps.pop(run_id, None)
            except OSError:
                persisted = False
                if run_id:
                    current = int(record.get("run_seq", 0))
                    existing = self._gaps.get(run_id)
                    if existing:
                        existing["missing_to"] = current
                    else:
                        self._gaps[run_id] = {
                            "missing_from": current,
                            "missing_to": current,
                        }
                        existing = self._gaps[run_id]
                    self._gap_history[run_id] = dict(self._gaps[run_id])
                    record["event_log_degraded"] = True
                    record["event_log_gap"] = dict(self._gap_history[run_id])
                    self._volatile_events[run_id].append(dict(record))
            if run_id and event == "turn_started":
                self._active_run_id = run_id
            if run_id and event == "turn_finished":
                self._completed_runs[run_id] = int(record.get("run_seq", 0))
                if self._active_run_id == run_id:
                    self._active_run_id = ""
            dispatched = records if persisted else [record]
            for item in dispatched:
                self._dispatch(run_id, item)
            return record

    def _dispatch(self, run_id, record):
        if not run_id:
            return
        for loop, queue in tuple(self._subscribers.get(run_id, ())):
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, dict(record))

    def _read_run_events(self, run_id, after_seq):
        if not self.path.exists():
            return []
        events = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("run_id") == run_id and int(record.get("run_seq", 0)) > after_seq:
                    events.append(record)
        return events

    def _load_sequences(self):
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                run_id = str(record.get("run_id", "") or "")
                if run_id:
                    self._run_sequences[run_id] = max(
                        self._run_sequences.get(run_id, 0),
                        int(record.get("run_seq", 0)),
                    )
                    if record.get("event") == "event_log_gap":
                        self._gap_history[run_id] = {
                            "missing_from": int(record.get("missing_from", 0)),
                            "missing_to": int(record.get("missing_to", 0)),
                        }
                    elif record.get("event_log_gap"):
                        self._gap_history[run_id] = dict(record["event_log_gap"])
                    if record.get("event") == "turn_finished":
                        self._completed_runs[run_id] = int(record.get("run_seq", 0))
                else:
                    self._session_sequence = max(
                        self._session_sequence,
                        int(record.get("session_seq", 0)),
                    )
