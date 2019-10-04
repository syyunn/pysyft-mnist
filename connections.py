import syft as sy


def connect_to_workers(hook, n_workers):
    workers = []
    for i in range(n_workers):
        worker = sy.VirtualWorker(hook, id=f"worker{i+1}")
        workers.append(worker)

    return workers


def connect_to_crypto_provider(hook):
    return sy.VirtualWorker(hook, id="crypto_provider")
