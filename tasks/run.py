import os

os.environ["OMP_NUM_THREADS"] = "1"

from utils.commons.hparams import hparams, set_hparams
import importlib


def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])
    cls_name = hparams["task_cls"].split(".")[-1]
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    # print(type(task_cls))
    # print(f"Class reference: {task_cls}") # 打印类引用
    # print(f"Class name: {task_cls.__name__}") # 打印类名
    # return
    task_cls.start()


if __name__ == '__main__':
    set_hparams()
    print("sdz param final")
    print(hparams['load_db_to_memory'])
    run_task()
