import os
import sys
import glog as log
from datetime import datetime
import numpy as np
import json


def set_log_file(fname):
    # set log file
    # simple tricks for duplicating logging destination in the logging module such as:
    # logging.getLogger().addHandler(logging.FileHandler(filename))
    # does NOT work well here, because python Traceback message (not via logging module) is not sent to the file,
    # the following solution (copied from : https://stackoverflow.com/questions/616645) is a little bit
    # complicated but simulates exactly the "tee" command in linux shell, and it redirects everything
    import subprocess
    # sys.stdout = os.fdopen(sys.stdout.fileno(), 'wb', 0)
    tee = subprocess.Popen(['tee', fname], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), sys.stdout.fileno())
    os.dup2(tee.stdin.fileno(), sys.stderr.fileno())

def modify_attack_success_rate(root_dir):

    log_path = "{}/modify_attack_success_rate_{}.log".format(root_dir,  datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    set_log_file(log_path)
    for dir_name in os.listdir(root_dir):
        if os.path.isdir(root_dir+"/"+dir_name):
            for file_name in os.listdir(root_dir+'/'+dir_name):
                file_path = root_dir + '/' + dir_name + '/' + file_name
                if file_name.endswith(".json") and not file_name.startswith("tmp") and not file_name.startswith("args.json"):
                    with open(file_path, "r") as file_obj:
                        data = json.loads(file_obj.read())
                    try:
                        orig_not_done = data["avg_not_done"]
                        not_done_all = np.array(data["not_done_all"]).astype(np.float32)
                        correct_all = np.array(data["correct_all"]).astype(np.int32)
                        avg_not_done = np.mean(not_done_all[np.nonzero(correct_all)[0]]).item()
                        if abs(avg_not_done - orig_not_done) >= 0.001:
                            log.info("{} Change {} to {}".format(file_path, orig_not_done, avg_not_done))
                            data["avg_not_done"] = avg_not_done
                            with open(file_path, "w") as result_file_obj:
                                json.dump(data, result_file_obj, sort_keys=True)
                    except Exception:
                        log.info("error in process {}".format(file_path))
                        raise

if __name__ == "__main__":
    modify_attack_success_rate("/home1/machen/meta_perturbations_black_box_attack/logs")