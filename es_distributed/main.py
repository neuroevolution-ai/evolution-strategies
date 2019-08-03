import errno
import json
import logging
import os
import sys

import click
from multiprocessing import Process, Queue, Lock, Manager

from es_distributed.es import run_master, run_worker, SharedNoiseTable

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


@click.group()
def cli():
    logging.basicConfig(
        format='[%(asctime)s pid=%(process)d] %(message)s',
        level=logging.INFO,
        stream=sys.stderr)


@cli.command()
@click.option('--exp_str') #configuration file
@click.option('--exp_file')
@click.option('--num_workers')
@click.option('--log_dir')
def master(exp_str, exp_file, num_workers, log_dir):
    """
    Starts the master which will listen for workers on the redis server.

    "master" must be used as first parameter when starting main.py, then this method gets called. In addition,
    exp_str xor exp_file must be provided, which is the configuration file and master_socket_path, the socket
    for the redis server is also mandatory.

    :param exp_str: A JSON formatted string containing the configuration
    :param exp_file: A JSON file containing the configuration
    :param master_socket_path: Path to the unixsocket used by redis
    :param log_dir: Optional to specify where to save logs
    :return: None
    """

    # Start the master
    # JSON String XOR JSON File
    assert (exp_str is None) != (exp_file is None), 'Must provide exp_str xor exp_file to the master'
    if exp_str:
        exp = json.loads(exp_str)
    elif exp_file:
        with open(exp_file, 'r') as f:
            exp = json.loads(f.read())
    else:
        assert False
    log_dir = os.path.expanduser(log_dir) if log_dir else '/tmp/es_master_{}'.format(os.getpid())
    mkdir_p(log_dir)

    noise = SharedNoiseTable()  # Workers share the same noise so less data needs to be interchanged
    num_workers = num_workers if num_workers else os.cpu_count()

    workers = []

    manager = Manager()
    tasks = manager.list()
    result_queue = Queue()

    lock = Lock()

    master_p = Process(target=run_master, args=(exp, tasks, result_queue, lock, log_dir,))
    master_p.start()

    for _ in range(int(num_workers)):
        worker_p = Process(target=run_worker, args=(noise, exp, tasks, result_queue, lock,))
        workers.append(worker_p)
        worker_p.start()


    for worker in workers:
        worker.join()

    master_p.join()

if __name__ == '__main__':
    cli()
