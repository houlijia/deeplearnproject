import os
import random
import socket
import time

from fasteners.process_lock import InterProcessLock


class BindFreePort(object):
    def __init__(self, start, stop):
        self.port = None
        self.sock = socket.socket()
        while True:
            port = random.randint(start, stop)
            try:
                self.sock.bind(('127.0.0.1', port))
                self.port = port
                time.sleep(0.01)
                break
            except Exception:
                continue
    def release(self):
        assert self.port is not None
        self.sock.close()
class FreePort(object):
    used_ports = set()
    def __init__(self, start=4000, stop=6000):
        self.lock = None
        self.bind = None
        self.port = None
        pid = os.getpid()
        while True:
            bind = BindFreePort(start, stop)
            print(f'{pid} got port : {bind.port}', end="\n")
            if bind.port in self.used_ports:
                print(f'{pid} will release port : {bind.port}')
                bind.release()
                continue
            lock = InterProcessLock(path='/tmp/socialdna/port_{}_lock'.format(bind.port))
            success = lock.acquire(blocking=False)
            if success:
                self.lock = lock
                self.port = bind.port
                self.used_ports.add(bind.port)
                bind.release()
                break
            bind.release()
            time.sleep(0.01)
    def release(self):
        assert self.lock is not None
        assert self.port is not None
        self.used_ports.remove(self.port)
        self.lock.release()
