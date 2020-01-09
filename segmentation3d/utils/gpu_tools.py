import subprocess


def get_gpu_memory(gpu_id):
    """Get the gpu memory usage.
    :param gpu_id the gpu id
    :return the gpu memory used
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])

    # convert lines into a dictionary
    gpu_memory = [int(x) for x in result.decode().strip().split('\n')]
    gpu_memory_dict = dict(list(zip(list(range(len(gpu_memory))), gpu_memory)))

    return gpu_memory_dict[gpu_id]