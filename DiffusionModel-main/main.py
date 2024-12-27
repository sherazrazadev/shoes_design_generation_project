import subprocess
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
subprocess.call(["python", "train.py", "-test", "-ts", timestamp])
subprocess.call(["python", "inference.py", "-d", f"training_{timestamp}"])
