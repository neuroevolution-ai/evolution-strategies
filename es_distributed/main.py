import errno
import json
import logging
import os
import sys

import click
from multiprocessing import Process, Pool, Queue

#from es_distributed.dist import RelayClient
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
#@click.option('--master_socket_path', required=True)
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

    task_queue = Queue()
    result_queue = Queue()

    with Pool(int(num_workers)) as pool:
        pool.apply_async(run_worker, args=(exp, noise, task_queue, result_queue))

    # master_p = Process(target=run_master, args=(exp, task_queue, result_queue, log_dir))
    # master_p.start()
    #
    # for _ in range(int(num_workers)):
    #     worker_p = Process(target=run_worker, args=(exp, noise, task_queue, result_queue))
    #     worker_p.start()



@cli.command()
#@click.option('--master_host', required=True)
#@click.option('--master_port', default=6379, type=int)
#@click.option('--relay_socket_path', required=True)
@click.option('--num_workers', type=int, default=0)
def workers(exp):
    """
    Starts a batch of workers, delivering work to the redis server.

    The defined number of workers gather the configuration from the master over the redis server and start working.
    Then their results are pushed onto the server where the master can gather the results.
    master_host and relay_socket_path are required. master_port is optional, the default ist 6379. num_workers is
    also optional and defaults to os.cpu_count()

    :param master_host: Location of the master server (e.g., if running locally 127.0.0.1)
    :param master_port: Port of the location of the master server
    :param relay_socket_path: Path to the unix socket of the redis server
    :param num_workers: Number of workers
    :return: None
    """

    # # Start the relay
    # master_redis_cfg = {'host': master_host, 'port': master_port}
    # relay_redis_cfg = {'unix_socket_path': relay_socket_path}
    # if os.fork() == 0:
    #     #RelayClient(master_redis_cfg, relay_redis_cfg).run()
    #     return
    # # Start the workers
    noise = SharedNoiseTable()  # Workers share the same noise so less data needs to be interchanged
    #num_workers = num_workers if num_workers else os.cpu_count()
    logging.info('Spawning {} workers'.format(num_workers))
    for _ in range(num_workers):
        if os.fork() == 0:
            run_worker(relay_redis_cfg, exp=exp, noise=noise)
            return
    os.wait()


if __name__ == '__main__':
    cli()
