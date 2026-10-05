import asyncio
import os
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path


class RuntimeLockManager:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runtime_locks ("
                "lock_key TEXT NOT NULL, owner_id TEXT NOT NULL, mode TEXT NOT NULL, "
                "pid INTEGER NOT NULL, heartbeat REAL NOT NULL, "
                "PRIMARY KEY(lock_key, owner_id))"
            )

    async def acquire_session(self, session_id):
        return await self.acquire(f"session:{session_id}", "write")

    async def acquire_workspace(self, root, mode):
        normalized = os.path.normcase(os.path.realpath(str(root)))
        return await self.acquire(f"workspace:{normalized}", mode)

    async def acquire(self, lock_key, mode):
        owner_id = uuid.uuid4().hex
        while True:
            task = asyncio.create_task(
                asyncio.to_thread(self._try_acquire, lock_key, owner_id, mode)
            )
            try:
                acquired = await asyncio.shield(task)
            except asyncio.CancelledError:
                acquired = await task
                if acquired:
                    await asyncio.to_thread(self._release, lock_key, owner_id)
                raise
            if acquired:
                return RuntimeLockLease(self, lock_key, owner_id)
            await asyncio.sleep(0.05)

    def _try_acquire(self, lock_key, owner_id, mode):
        now_value = time.time()
        with closing(
            sqlite3.connect(self.path, timeout=10, isolation_level=None)
        ) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT owner_id, mode, pid, heartbeat FROM runtime_locks WHERE lock_key=?",
                (lock_key,),
            ).fetchall()
            for existing_owner, _, pid, heartbeat in rows:
                if now_value - float(heartbeat) > 30 and not _pid_alive(int(pid)):
                    connection.execute(
                        "DELETE FROM runtime_locks WHERE lock_key=? AND owner_id=?",
                        (lock_key, existing_owner),
                    )
            rows = connection.execute(
                "SELECT mode FROM runtime_locks WHERE lock_key=?", (lock_key,)
            ).fetchall()
            allowed = not rows if mode == "write" else all(
                row[0] == "read" for row in rows
            )
            if allowed:
                connection.execute(
                    "INSERT INTO runtime_locks(lock_key, owner_id, mode, pid, heartbeat) "
                    "VALUES(?,?,?,?,?)",
                    (lock_key, owner_id, mode, os.getpid(), now_value),
                )
            connection.commit()
        return allowed

    def _renew(self, lock_key, owner_id):
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            connection.execute(
                "UPDATE runtime_locks SET heartbeat=? WHERE lock_key=? AND owner_id=?",
                (time.time(), lock_key, owner_id),
            )
            connection.commit()

    def _release(self, lock_key, owner_id):
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            connection.execute(
                "DELETE FROM runtime_locks WHERE lock_key=? AND owner_id=?",
                (lock_key, owner_id),
            )
            connection.commit()


class RuntimeLockLease:
    def __init__(self, manager, lock_key, owner_id):
        self.manager = manager
        self.lock_key = lock_key
        self.owner_id = owner_id
        self._heartbeat_task = None
        self._closed = False

    async def __aenter__(self):
        self._heartbeat_task = asyncio.create_task(self._heartbeat())
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self._closed = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
        await asyncio.to_thread(self.manager._release, self.lock_key, self.owner_id)

    async def _heartbeat(self):
        while not self._closed:
            await asyncio.sleep(5)
            await asyncio.to_thread(
                self.manager._renew, self.lock_key, self.owner_id
            )


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
