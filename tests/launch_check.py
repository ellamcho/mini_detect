"""Start the installed `mini-detect` command and check that it stays running.

Usage: python tests/launch_check.py <folder containing the mini-detect command>
"""
import os
import subprocess
import sys
import time

folder = sys.argv[1]
names = [n for n in os.listdir(folder) if os.path.splitext(n)[0] == "mini-detect"]
assert names, f"no mini-detect command in {folder}: {os.listdir(folder)}"
command = os.path.join(folder, names[0])
print("starting", command)
process = subprocess.Popen([command])
time.sleep(15)
code = process.poll()
if code is not None:
    sys.exit(f"mini-detect exited straight away with code {code}")
print("mini-detect is still running after 15 s")
process.terminate()
