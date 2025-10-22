# worker_simulation/simulated_iphone.py
#useless for now not using it for anything
import asyncio
import websockets
import json
import numpy as np

async def worker():
    # set static domain name for ngrok tunnel
    # This won't change unless you change it in the ngrok command.
    DOMAIN_NAME = "metal-forge-demo.ngrok-free.app"

    # The WebSocket URL is derived from the domain name
    uri = f"wss://{DOMAIN_NAME}/ws/iphone_simulator_01"
    
    print(f"Attempting to connect to broker at: {uri}")
    # code below is just some matrix multiplication to simulate the ngrok tunneling later on we need to change this to real iphone metal code and stuff
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Yippee! Worker connected to broker successfully!")
            while True:
                job_str = await websocket.recv()
                job = json.loads(job_str)
                job_id = job.get("job_id")
                print(f"Received job: {job_id}")

                # matrix multiplication simulation
                data = job.get("data")
                matrix_a = np.array(data.get("matrix_a"))
                matrix_b = np.array(data.get("matrix_b"))
                
                result_matrix = np.dot(matrix_a, matrix_b)
                
                result_payload = {
                    "job_id": job_id,
                    "data": result_matrix.tolist()
                }
                
                await websocket.send(json.dumps(result_payload))
                print(f"Sent back result for job {job_id}")
    except Exception as e:
        print(f"🥀 Error connecting to broker: {e}")

if __name__ == "__main__":
    asyncio.run(worker())