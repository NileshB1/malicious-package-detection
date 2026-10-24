"""Column groups of the processed QUT-DV25 table.

The grouping follows the six trace sources described in the dataset card. Count columns are
integers; the rest are text that needs parsing before modelling (Phase 4).
"""

FILESYSTEM_COUNTS = [
    "Root_DIR_Installation",
    "Temporary_DIR_Installation",
    "Home_DIR_Installation",
    "User_Access",
    "Sys_Access",
    "Etc_DIR_Installation",
    "Other_DIR_Installation",
]

NETWORK_COUNTS = [
    "Local_IP_Address_Access",
    "Remote_IP_Address_Access",
    "Local_Port_Access",
    "Remote_Port_Access",
]

SYSCALL_COUNTS = [
    "File_Operations",
    "Network_Operations",
    "Process_Management_Operations",
    "IO_Operations",
    "Time_Operations",
    "Security_Operations",
]

TCP_TEXT = ["State_Transition"]

PROCESS_TEXT = [
    "Read_Processes",
    "Write_Processes",
    "Read_Data_Transfer_Processes",
    "Write_Data_Transfer_Processes",
    "File_Access_Processes",
]

DEPENDENCY_TEXT = ["Total_Dependencies", "Direct_Dependencies", "Indirect_Dependencies"]

PATTERN_TEXT = [f"Pattern_{i}" for i in range(1, 11)]

COUNT_COLUMNS = FILESYSTEM_COUNTS + NETWORK_COUNTS + SYSCALL_COUNTS
TEXT_COLUMNS = TCP_TEXT + PROCESS_TEXT + DEPENDENCY_TEXT + PATTERN_TEXT
FEATURE_COLUMNS = COUNT_COLUMNS + TEXT_COLUMNS

# The process lists are filetop/opensnoop "top N" lists for the whole machine, so most tokens
# are not the package at all. Built from the token frequencies per class (see
# reports/tables/process_tokens_by_class.csv). Names are truncated to 15 chars by the kernel,
# hence things like "gnome-system-mo".

# the tracing tools themselves
MONITORING_PROCESSES = {"opensnoop-bpfcc", "filetop-bpfcc", "tcpstates-bpfcc", "strace", "pgrep"}

# Desktop and background services on the sandbox. Several are almost class-exclusive
# (tracker-miner-f only in malicious runs, chrome/ThreadPoolForeg only in benign runs), which
# says the two classes were recorded in different sessions - a collection artefact, not behaviour.
DESKTOP_PROCESSES = {
    "gnome-system-mo",
    "gnome-shell",
    "gnome-terminal-",
    "nautilus",
    "clear",
    "systemd-journal",
    "systemd-logind",
    "systemd-oomd",
    "snapd",
    "fwupd",
    "apt-check",
    "tracker-miner-f",
    "pool-tracker-mi",
    "pool-org.gnome.",
    "glean.dispatche",
    "chrome",
    "ThreadPoolForeg",
    "MemoryInfra",
    "Socket",
    "StreamTrans",
    "StreamT~ns",
    "StreamT~s",
    "sqldb:c~lite",
    "Classif~date",
    "Classif~ate",
    "Cache2",
    "Indexed~IO",
}

# How the installer was launched. pip3 appears in a third of benign runs and almost never in
# malicious ones, so even this differs by class and can't be trusted as a signal.
INSTALLER_PROCESSES = {"pip", "pip3", "python", "sudo"}

ENVIRONMENT_PROCESSES = MONITORING_PROCESSES | DESKTOP_PROCESSES | INSTALLER_PROCESSES
