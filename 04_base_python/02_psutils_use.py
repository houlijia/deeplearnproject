import os
import platform
import signal
import psutil


def get_pids_by_process_name(name):
    """
    :param name: process name
    :return: process pids
    """
    if not name:
        return

    pids_list = []
    pids_name_list = []
    pids = psutil.pids()
    for pid in pids:
        try:
            p = psutil.Process(pid)
            process_name = p.name()
            if name == process_name:
                pids_list.append(pid)
                pids_name_list.append(process_name)
        except Exception as e:
            print(f"error: {e}")
            continue
    return pids_list, pids_name_list


class SystemInfo(object):
    SYSTEM_NAME = platform.system().lower()
    IS_WINDOWS = True if SYSTEM_NAME == "windows" else False
    IS_MAC = True if SYSTEM_NAME == "drawn" else False


def kill_process_by_name(name):
    """
    :param name:  process name
    :return:
    """
    if not name:
        return

    pids, pids_names = get_pids_by_process_name(name)
    if len(pids) == 0:
        return

    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL if SystemInfo.IS_MAC else signal.SIGINT)
        except psutil.NoSuchProcess as e:
            print(f"error: {e}")


kill_process_by_name("语雀")
