import paramiko
from scp import SCPClient
router_ip="10.5.12.223"
router_username="taqiaden"
router_password="taqiaden"

ssh=paramiko.SSHClient()


def run_command_on_device(ip_address, username, password, command):
    """ Connect to a device, run a command, and return the output."""

    # Load SSH host keys.
    ssh.load_system_host_keys()
    # Add SSH host key when missing.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    total_attempts = 3
    for attempt in range(total_attempts):
        try:
            print("Attempt to connect: %s" % attempt)
            # Connect to router using username/password authentication.
            ssh.connect(router_ip,
                        username=router_username,
                        password=router_password,
                        look_for_keys=False)
            # Run command.
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)
            # Read output from command.
            output = ssh_stdout.readlines()
            # Close connection.
            ssh.close()
            return output

        except Exception as error_message:
            print("Unable to connect")
            print(error_message)

if __name__ == "__main__":

    # Run function
    router_output = run_command_on_device(router_ip, router_username, router_password, "show ip route")

    # Analyze show ip route output
    # Make sure we didn't receive empty output.
    if router_output != None:
        for line in router_output:
            if "0.0.0.0/0" in line:
                print("Found default route:")
                print(line)


    '''scp equivalent functions'''

    sftp = ssh.open_sftp()
    sftp.put('<Source>', '<Destination>')

    '''alternative scp'''
    scp = SCPClient(ssh.get_transport())
