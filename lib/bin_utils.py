import subprocess
import time
from process_perception import trigger_new_perception


def empty_bin_check(state):
    if state == 'No Object':
        print('state')
        subprocess.run('bash/testUSBCAN')
        # os.system('./testUSBCAN')

        time.sleep(5)
        trigger_new_perception()
        return True
    else:return False
