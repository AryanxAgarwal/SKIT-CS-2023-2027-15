
import time

print("Workload started")

for i in range(15):
    print(f"Processing task {i + 1}/15")
    time.sleep(1)

print("Workload completed")
