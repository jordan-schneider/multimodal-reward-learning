STATUS_FIELDS = ["VmRSS", "VmHWM", "VmSize", "VmPeak"]
SMAPS_FIELDS = ["Private_Clean", "Private_Dirty"]


def get_memory():
    """
    returns the current and peak, real and virtual memories
    used by the calling linux python process, in Bytes
    """

    # read in process info
    with open("/proc/self/status", "r") as file:
        lines = file.read().split("\n")

    # container of memory values (_FIELDS)
    values = {}

    # check all process info fields
    for line in lines:
        if ":" in line:
            name, val = line.split(":")

            # collect relevant memory fields
            if name in STATUS_FIELDS:
                values[name] = int(val.strip().split(" ")[0])  # strip off "kB"
                values[name] *= 1000  # convert to B

    # check we collected all info
    assert len(values) == len(STATUS_FIELDS)

    with open("/proc/self/smaps", "r") as file:
        lines = file.read().split("\n")

    uss = 0
    for line in lines:
        if ":" in line:
            name, val = line.split(":")

            # collect relevant memory fields
            if name in SMAPS_FIELDS:
                kbs = int(val.strip().split(" ")[0])  # strip off "kB"
                uss += kbs * 1000  # convert to B
    values["uss"] = uss

    return values
