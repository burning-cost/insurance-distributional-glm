"""
Submit test job to Databricks via Jobs API.

Usage:
  export DATABRICKS_HOST=https://...
  export DATABRICKS_TOKEN=dapi...
  python notebooks/submit_test_job.py
"""
import os
import sys
import time
import json
import urllib.request
import urllib.error

HOST = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
TOKEN = os.environ.get("DATABRICKS_TOKEN", "")

if not HOST or not TOKEN:
    print("Set DATABRICKS_HOST and DATABRICKS_TOKEN environment variables.")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}


def api_call(method, path, payload=None, params=None):
    url = f"{HOST}{path}"
    if params:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{qs}"
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# Submit a one-time notebook job using serverless compute
job_payload = {
    "run_name": "insurance-distributional-glm tests",
    "tasks": [{
        "task_key": "run_tests",
        "notebook_task": {
            "notebook_path": "/Workspace/insurance-distributional-glm/run_tests",
            "source": "WORKSPACE",
        },
    }],
    "queue": {"enabled": True},
}

print("Submitting job...")
result = api_call("POST", "/api/2.1/jobs/runs/submit", payload=job_payload)
run_id = result["run_id"]
print(f"Submitted run_id: {run_id}")
print(f"Track at: {HOST}/#job/runs/{run_id}")

# Poll for completion
for i in range(80):
    time.sleep(30)
    data = api_call("GET", "/api/2.1/jobs/runs/get", params={"run_id": run_id})
    life_cycle = data["state"]["life_cycle_state"]
    result_state = data["state"].get("result_state", "")
    print(f"  [{i*30}s] State: {life_cycle} / {result_state}")
    if life_cycle in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print(f"\nFinal state: {life_cycle} / {result_state}")

# Get task output
tasks = data.get("tasks", [])
task_run_id = tasks[-1]["run_id"] if tasks else None

if task_run_id:
    try:
        output_data = api_call("GET", "/api/2.1/jobs/runs/get-output",
                               params={"run_id": task_run_id})
        nb = output_data.get("notebook_output", {})
        result_text = nb.get("result", "")
        if result_text:
            print("\n=== Notebook output ===")
            print(result_text)
        error = output_data.get("error", "")
        if error:
            print(f"\nError: {error[:1000]}")
    except Exception as e:
        print(f"Could not get output: {e}")

if result_state != "SUCCESS":
    print(f"\nRun FAILED: {result_state}")
    sys.exit(1)
else:
    print("\nAll tests PASSED.")
