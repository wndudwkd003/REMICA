from config.config import Config


def make_plan(config: Config) -> list:
    gpus = config.able_gpus
    total = config.multi_worker
    n_gpus = len(gpus)

    base = total // n_gpus
    rem = total % n_gpus

    plan = []
    for i, gid in enumerate(gpus):
        n = base + (1 if i < rem else 0)
        if n > 0:
            plan.append((gid, n))

    return plan
