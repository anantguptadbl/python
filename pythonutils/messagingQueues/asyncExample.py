import asyncio
from asyncio import Queue, sleepasync def producer(q, n):
    for item in range(n):
        await q.put(item)
        await sleep(1)
        print(f"[Produced] {item}")
    print("[Producer] Done")
    await q.put(None)async def consumer(q):
    while True:
        item = await q.get()
        if item is None:
            break
        print(f"[Consumer] {item}")
        await sleep(1)
    print("[Consumer] Done")n = 10
q = asyncio.Queue()
loop = asyncio.get_event_loop()
loop.create_task(producer(q, n))
loop.create_task(consumer(q))
