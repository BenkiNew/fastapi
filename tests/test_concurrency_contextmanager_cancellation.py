import threading
from contextlib import contextmanager

import anyio
import pytest
from fastapi.concurrency import contextmanager_in_threadpool


@pytest.mark.parametrize("backend", ["asyncio", "trio"])
def test_contextmanager_in_threadpool_cancel(backend):
    async def run():
        events = []
        threads = []
        loop_thread = threading.get_ident()
        started = anyio.Event()
        cancelled = anyio.get_cancelled_exc_class()

        @contextmanager
        def dependency():
            threads.append(threading.get_ident())
            events.append("enter")
            try:
                yield "resource"
            except cancelled:
                events.append("cancelled")
                raise
            finally:
                threads.append(threading.get_ident())
                events.append("cleanup")

        async def request():
            async with contextmanager_in_threadpool(dependency()) as value:
                assert value == "resource"
                started.set()
                await anyio.sleep_forever()

        async with anyio.create_task_group() as group:
            group.start_soon(request)
            await started.wait()
            group.cancel_scope.cancel()
        assert events == ["enter", "cancelled", "cleanup"]
        assert all(thread != loop_thread for thread in threads)

    anyio.run(run, backend=backend)


def test_contextmanager_in_threadpool_asyncio_task_cancel():
    import asyncio

    async def run():
        events = []
        started = asyncio.Event()

        @contextmanager
        def dependency():
            try:
                yield "resource"
            except asyncio.CancelledError:
                events.append("cancelled")
                raise
            finally:
                events.append("cleanup")

        async def request():
            async with contextmanager_in_threadpool(dependency()):
                started.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(request())
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["cancelled", "cleanup"]

    asyncio.run(run())


@pytest.mark.parametrize("suppress", [True, False])
def test_contextmanager_in_threadpool_base_exception(suppress):
    class Abort(BaseException):
        pass

    async def run():
        events = []
        error = Abort("expected")

        @contextmanager
        def dependency():
            try:
                yield "resource"
            except Abort as caught:
                assert caught is error
                events.append("caught")
                if not suppress:
                    raise
            finally:
                events.append("cleanup")

        async def request():
            async with contextmanager_in_threadpool(dependency()):
                raise error

        if suppress:
            await request()
        else:
            with pytest.raises(Abort) as caught:
                await request()
            assert caught.value is error
        assert events == ["caught", "cleanup"]

    anyio.run(run)
