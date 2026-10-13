import asyncio
import threading
from contextlib import asynccontextmanager
from functools import partial, wraps

import httpx
import pytest
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("shape", ["function", "partial", "instance"])
@pytest.mark.parametrize("outcome", ["success", "http", "runtime"])
@pytest.mark.parametrize("scope", ["function", "request"])
def test_coroutine_returning_generator(kind, shape, outcome, scope):
    events = []
    worker_threads = []
    endpoint_threads = []

    def sync_dep():
        worker_threads.append(threading.get_ident())
        events.append("enter")
        try:
            yield "resource"
        except BaseException as exc:
            events.append(type(exc).__name__)
            raise
        finally:
            worker_threads.append(threading.get_ident())
            events.append("cleanup")

    async def async_dep():
        events.append("enter")
        try:
            yield "resource"
        except BaseException as exc:
            events.append(type(exc).__name__)
            raise
        finally:
            events.append("cleanup")

    inner = sync_dep if kind == "sync" else async_dep

    @wraps(inner)
    async def wrapper():
        events.append("wrapper")
        return inner()

    dep = wrapper
    if shape == "partial":
        dep = partial(wrapper)
    elif shape == "instance":

        class CallableDependency:
            if kind == "sync":

                def original(self):
                    yield from sync_dep()
            else:

                async def original(self):
                    async with asynccontextmanager(async_dep)() as value:
                        yield value

            @wraps(original)
            async def __call__(self):
                events.append("wrapper")
                return self.original()

        dep = CallableDependency()

    app = FastAPI()

    @app.get("/")
    async def endpoint(tasks: BackgroundTasks, value=Depends(dep, scope=scope)):
        endpoint_threads.append(threading.get_ident())
        assert value == "resource"
        events.append("endpoint")
        tasks.add_task(events.append, "background")
        if outcome == "http":
            raise HTTPException(418)
        if outcome == "runtime":
            raise RuntimeError("expected")
        return {"value": value}

    with TestClient(app) as client:
        if outcome == "runtime":
            with pytest.raises(RuntimeError, match="expected"):
                client.get("/")
        else:
            response = client.get("/")
            assert response.status_code == (200 if outcome == "success" else 418)
            if outcome == "success":
                assert response.json() == {"value": "resource"}
        assert events.count("cleanup") == 1
        assert events.count("wrapper") == 1
        if outcome != "success":
            assert ("HTTPException" if outcome == "http" else "RuntimeError") in events
        elif scope == "function":
            assert events.index("cleanup") < events.index("background")
        else:
            assert events.index("background") < events.index("cleanup")
        if kind == "sync":
            assert all(t != endpoint_threads[0] for t in worker_threads)


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("use_cache", [True, False])
def test_coroutine_generator_cache(kind, use_cache):
    events = []

    def sync_dep():
        events.append("enter")
        try:
            yield "resource"
        finally:
            events.append("cleanup")

    async def async_dep():
        events.append("enter")
        try:
            yield "resource"
        finally:
            events.append("cleanup")

    inner = sync_dep if kind == "sync" else async_dep

    @wraps(inner)
    async def dep():
        return inner()

    app = FastAPI()

    @app.get("/")
    async def endpoint(a=Depends(dep), b=Depends(dep, use_cache=use_cache)):
        return [a, b]

    with TestClient(app) as client:
        assert client.get("/").json() == ["resource", "resource"]
        assert (
            events.count("enter") == events.count("cleanup") == (1 if use_cache else 2)
        )


def test_coroutine_generator_cancellation():
    async def run():
        events = []
        started = asyncio.Event()

        async def async_dep():
            try:
                yield "resource"
            except BaseException as exc:
                events.append(type(exc).__name__)
                raise
            finally:
                events.append("cleanup")

        inner = async_dep

        @wraps(inner)
        async def dep():
            return inner()

        app = FastAPI()

        @app.get("/")
        async def endpoint(value=Depends(dep)):
            assert value == "resource"
            started.set()
            await asyncio.Event().wait()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(client.get("/"))
            try:
                await asyncio.wait_for(started.wait(), 2)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert events == ["CancelledError", "cleanup"]

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["sync", "async"])
@pytest.mark.parametrize("count", [0, 2])
def test_coroutine_generator_invalid_yield_count(kind, count):
    events = []

    def sync_dep():
        try:
            for _ in range(count):
                yield "resource"
        finally:
            events.append("cleanup")

    async def async_dep():
        try:
            for _ in range(count):
                yield "resource"
        finally:
            events.append("cleanup")

    inner = sync_dep if kind == "sync" else async_dep

    @wraps(inner)
    async def dep():
        return inner()

    app = FastAPI()

    @app.get("/")
    async def endpoint(value=Depends(dep)):
        return value

    message = "generator didn't yield" if count == 0 else "generator didn't stop"
    with TestClient(app) as client:
        with pytest.raises(RuntimeError, match=message):
            client.get("/")
        assert events == ["cleanup"]


def test_coroutine_sync_generator_does_not_block_event_loop():
    async def run():
        entered = threading.Event()
        release = threading.Event()

        def inner():
            entered.set()
            assert release.wait(5), "event loop did not release the worker"
            yield "resource"

        @wraps(inner)
        async def dep():
            return inner()

        app = FastAPI()

        @app.get("/resource")
        async def resource(value=Depends(dep)):
            return value

        @app.get("/ping")
        async def ping():
            return "pong"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            task = asyncio.create_task(client.get("/resource"))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                response = await asyncio.wait_for(client.get("/ping"), 2)
                assert response.json() == "pong"
            finally:
                release.set()
                response = await task
            assert response.json() == "resource"

    asyncio.run(run())
