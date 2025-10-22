# client_notebooks/aetherforge_lib.py

# System & Setup
import os
import threading
import time
import json
import base64
import io

# Web Server & Networking
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
import uvicorn
from pyngrok import ngrok
import requests

# User Interface & Data
import qrcode
from IPython.display import display

# Machine Learning (PyTorch/ONNX Path)
import torch
import torch.nn as nn
import onnx
class MetalForgeSession:
    """Manages a compute session by handling the server, tunnel, and worker connection."""
    
    def __init__(self, ngrok_authtoken: str):
        """Initializes the session with the user's ngrok Authtoken."""
        if not ngrok_authtoken:
            raise ValueError("An ngrok Authtoken is required to create a public URL.")
        
        self.ngrok_authtoken = ngrok_authtoken
        self._app = FastAPI()
        self._active_worker: WebSocket = None
        self._job_results = {}
        self.public_url = None
        
        self._setup_routes()

    
    def _setup_routes(self):
        """Defines the API endpoints for the broker server."""
        
        # Determine the absolute path to the web_worker directory
        lib_dir = os.path.dirname(os.path.abspath(__file__))
        web_worker_path = os.path.join(lib_dir, '..', 'web_worker')
        # Serve the index and static files from the web_worker folder.
        # Return FileResponse explicitly so FastAPI sends the correct file content and headers.
        @self._app.get("/", response_class=FileResponse)
        async def serve_index():
            return FileResponse(os.path.join(web_worker_path, 'index.html'))

        @self._app.get("/{filename}")
        async def serve_static(filename: str):
            # Guard against path traversal
            safe_path = os.path.normpath(os.path.join(web_worker_path, filename))
            if not safe_path.startswith(os.path.normpath(web_worker_path)):
                return FileResponse(os.path.join(web_worker_path, 'index.html'))

            # If file doesn't exist (e.g., browser requests /favicon.ico), return 204 No Content
            if not os.path.exists(safe_path):
                # Log missing static file for debugging
                print(f"Static file not found: {safe_path} -> returning 204")
                return Response(status_code=204)

            return FileResponse(safe_path)

        # Single websocket endpoint to receive results from the worker
        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._active_worker = websocket
            print("✅ Worker has connected!")
            try:
                while True:
                    result_data = await websocket.receive_text()
                    result = json.loads(result_data)
                    job_id = result.get('job_id')
                    if job_id:
                        self._job_results[job_id] = result.get('data')
                        print(f"Result received for job {job_id}")
            except WebSocketDisconnect:
                print("Worker disconnected.")
                self._active_worker = None

        # Single POST endpoint to submit jobs to the active worker
        @self._app.post("/submit_job")
        async def submit_job(job_data: dict):
            if not self._active_worker:
                return {"error": "No worker connected"}
            
            job_id = str(len(self._job_results) + 1)
            job = {"job_id": job_id, "data": job_data}
            try:
                print(f"Sending job {job_id} to worker: {job}")
                await self._active_worker.send_text(json.dumps(job))
            except Exception as e:
                # More helpful error for debugging if send fails
                print(f"Failed to send job {job_id} to worker: {e}")
                return {"error": f"Failed to send job to worker: {e}"}

            self._job_results[job_id] = "pending"
            print(f"Job {job_id} queued; waiting for result...")
            return {"message": "Job sent to worker", "job_id": job_id}

    def start(self):
        """Starts the server, creates the ngrok tunnel, and provides the worker link."""
        os.environ['NGROK_AUTHTOKEN'] = self.ngrok_authtoken
        
        thread = threading.Thread(target=lambda: uvicorn.run(self._app, host="127.0.0.1", port=8000, log_level="warning"))
        thread.daemon = True
        thread.start()
        time.sleep(4) 

        self.public_url = ngrok.connect(8000)
        print(f" Broker is up!")

        print(f" Click here to launch the Web Worker: {self.public_url.public_url}")

        # 2. QR Code for Mobile App
        websocket_url = self.public_url.public_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        qr_img = qrcode.make(websocket_url)
        print("\n📱 For Mobile Worker (Future App): Scan the QR code below.")
        display(qr_img)
    '''
    def load_model_on_worker_tf(self, model: tf.keras.Model):
        """Converts a Keras model to TF.js format and sends it to the worker."""
        if not self._active_worker:
            print(" Error: No worker is connected.")
            return

        print("Converting Keras model to TensorFlow.js format")
        # Save the model to a temporary in-memory path
        tfjs.converters.save_keras_model(model, './tfjs_model')
        
        # Read the model architecture file
        with open('./tfjs_model/model.json', 'r') as f:
            model_json = json.load(f)
            
        # Read the binary weights file and encode it as a Base64 string for safe transfer
        with open('./tfjs_model/group1-shard1of1.bin', 'rb') as f:
            weights_data = base64.b64encode(f.read()).decode('utf-8')
        
        print("Sending model to worker")
        # Use the existing 'run' method to send the 'load_model' command
        self.run(
            operation="load_model",
            data={
                "model_json": model_json,
                "weights_data_b64": weights_data
            },
            wait_for_result=False # Don't wait for a reply for this special command
        )
        print(" Model sent. The worker is now ready for inference.")
    '''
    #pytorch model loader
    def load_onnx_model_on_worker(self, model: nn.Module, dummy_input: torch.Tensor):
        """Converts a PyTorch model to ONNX format and sends it to the worker."""
        if not self._active_worker:
            print(" Error: No worker is connected.")
            return

        print("Converting PyTorch model to ONNX format")
        # Use an in-memory buffer to avoid writing to disk
        f = io.BytesIO()
        try:
            # New exporters in torch may require onnxscript; try the default first
            torch.onnx.export(model, dummy_input, f)
        except ModuleNotFoundError as e:
            # Known failure: missing onnxscript in some environments
            if 'onnxscript' in str(e) or 'onnx' in str(e):
                print("onnxscript not available in this environment; retrying export with dynamo=False fallback.")
                try:
                    # Retry using the legacy exporter by disabling dynamo
                    torch.onnx.export(model, dummy_input, f, dynamo=False)
                except Exception as e2:
                    print("Fallback export also failed:", e2)
                    print("To fix, install the optional dependencies: pip install onnx onnxscript")
                    raise
            else:
                raise
        onnx_model_bytes = f.getvalue()
        
        # Encode the binary ONNX model as a Base64 string for safe JSON transfer
        onnx_b64 = base64.b64encode(onnx_model_bytes).decode('utf-8')
        
        print("Sending ONNX model to worker")
        self.run(
            operation="load_onnx_model",
            data={"onnx_model_b64": onnx_b64},
            wait_for_result=False
        )
        print(" ONNX Model sent. The worker is now ready for inference.")

    def run(self, operation: str, data: dict,wait_for_result=True):
        """Submits a job to the connected worker and waits for the result."""
        if not self._active_worker:
            print(" Error: No worker is connected. Please connect a worker before running a job.")
            return None
        
        # Package the job data
        payload = {"operation": operation, **data}

        # Make a simple HTTP request to our new endpoint
        url = f"{self.public_url.public_url}/submit_job"
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            print(f" Error submitting job: {response.text}")
            return None
            
        response_data = response.json()
        job_id = response_data.get("job_id")
        print(f"Job {job_id} submitted to worker...")

        if not wait_for_result:
            return None

        # Poll for the result
        for _ in range(10): # Try for 20 seconds
            if self._job_results.get(job_id) and self._job_results.get(job_id) != "pending":
                print("✅ Result received!")
                return self._job_results[job_id]
            time.sleep(2)
        
        print(" Error: Timed out waiting for result.")
        return None

    def close(self):
        """Shuts down the ngrok tunnel and cleans up the session."""
        if self.public_url:
            ngrok.disconnect(self.public_url)
            print("Tunnel closed.")