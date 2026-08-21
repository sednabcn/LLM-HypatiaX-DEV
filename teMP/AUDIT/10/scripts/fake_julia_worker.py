# Stands in for "python -c _SUBPROCESS_WORKER" from the real harness.
# Mimics the exact orphaning shape: this wrapper spawns ITS OWN OS
# subprocess (standing in for Julia) rather than calling a library
# in-process, then blocks waiting on it.
import subprocess, sys, time, os

# Spawn the "julia" stand-in: a long-running process with a distinctive,
# greppable name so we can check if it's still alive after the parent dies.
child = subprocess.Popen(
    [sys.executable, "-c",
     "import time; open('/tmp/fake_julia.pid','w').write(str(__import__('os').getpid())); time.sleep(300)"]
)
print(f"wrapper pid={os.getpid()} spawned fake-julia child pid={child.pid}", flush=True)
child.wait()  # block here, just like the real worker blocking on Julia
