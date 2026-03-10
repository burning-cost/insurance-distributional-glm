# Databricks notebook source
# MAGIC %pip install git+https://github.com/burning-cost/insurance-distributional-glm.git pytest

# COMMAND ----------

import subprocess
import sys
import shutil
import os

# Copy tests to /tmp (workspace doesn't support __pycache__)
src = "/Workspace/insurance-distributional-glm/tests"
dst = "/tmp/gamlss_tests"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"Tests copied to {dst}")
print("Test files:", os.listdir(dst))

# COMMAND ----------

result = subprocess.run(
    [
        sys.executable, "-m", "pytest",
        dst,
        "-v", "--tb=short", "--no-header",
        "-p", "no:cacheprovider",
    ],
    capture_output=True,
    text=True,
)
output = result.stdout
if len(output) > 12000:
    print(output[:4000])
    print("\n...[middle truncated]...\n")
    print(output[-8000:])
else:
    print(output)
if result.stderr:
    print("STDERR:", result.stderr[:500])
print("Return code:", result.returncode)

dbutils.notebook.exit(f"rc={result.returncode}\n" + output[-4000:])
