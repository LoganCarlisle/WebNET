// main.js

console.log("main.js script started!");

const statusEl = document.getElementById('status-box');
const modelStatusEl = document.getElementById && document.getElementById('model-status');
const infraStatusEl = document.getElementById && document.getElementById('infra-status');
const logBox = document.getElementById && document.getElementById('log-box');

let model; // Global variable for our ML model
let inferenceSession = null; // ONNX inference session

function appendLog(msg) {
    const ts = new Date().toISOString();
    const line = `[${ts}] ${msg}`;
    console.log(line);
    if (logBox) {
        logBox.textContent = line + '\n' + logBox.textContent;
    }
}

// job router
// This object maps operation names to their handler functions.
const jobHandlers = {
    "load_model": handleLoadModel,
    "load_onnx_model": handleLoadOnnxModel,
    "inference": handleInference,
    "onnx_inference": handleOnnxInference,
    "square_array": handleSquareArray
};


// Automatically determine the WebSocket URL and connect
const host = window.location.host;
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${host}/ws`;
const socket = new WebSocket(wsUrl);

socket.onerror = function(err) {
    console.error('WebSocket error:', err);
};

socket.onclose = function(event) {
    console.warn('WebSocket closed:', event);
    statusEl.textContent = " Disconnected";
    statusEl.className = "disconnected";
};

socket.onopen = function() {
    statusEl.textContent = " Connected! Waiting for a job";
    statusEl.className = "connected";
};

// The onmessage function 
socket.onmessage = async function(event) {
    console.log('Received message from broker:', event.data);
    const job = JSON.parse(event.data);
    const operation = job.data.operation;

    // Look up the correct handler in our router and call it
    const handler = jobHandlers[operation];
    if (handler) {
        try {
            console.log(`Handling operation: ${operation}, job id: ${job.job_id}`);
            await handler(job);
            console.log(`Finished operation: ${operation}, job id: ${job.job_id}`);
        } catch (err) {
            console.error(`Error handling job ${job.job_id}:`, err);
            // send back an error payload so the broker doesn't wait forever
            const errPayload = { job_id: job.job_id, error: String(err) };
            try { socket.send(JSON.stringify(errPayload)); } catch (e) { console.error('Failed to send error payload:', e); }
        }
    } else {
        console.error(`Unknown operation: ${operation}`);
    }
};

// handle functions for each job type

async function handleLoadOnnxModel(job) {
    appendLog('Receiving and loading ONNX model...');
    statusEl.textContent = " Receiving and loading ONNX model";
    statusEl.className = "processing";
    if (infraStatusEl) infraStatusEl.textContent = 'Infra: loading model';

    try {
        // Decode the Base64 string from the Colab notebook back into binary data
        const onnxB64 = job.data.onnx_model_b64;
        const onnxBytes = base64ToArrayBuffer(onnxB64);

        // Create an ONNX inference session from the model bytes
        inferenceSession = new onnx.InferenceSession();
        await inferenceSession.loadModel(onnxBytes);

        if (modelStatusEl) modelStatusEl.textContent = 'Model: loaded (ONNX)';
        if (infraStatusEl) infraStatusEl.textContent = 'Infra: ready';
        appendLog('ONNX model loaded successfully.');

        statusEl.textContent = " ONNX Model loaded Ready for inference.";
        statusEl.className = "connected";
    } catch (error) {
        appendLog('Error loading ONNX model: ' + String(error));
        console.error("Error loading ONNX model:", error);
        statusEl.textContent = ` Error: ${error.message}`;
        statusEl.className = "disconnected";
        if (infraStatusEl) infraStatusEl.textContent = 'Infra: error';
    }
}

async function handleOnnxInference(job) {
    if (!inferenceSession) {
        appendLog('ONNX inference requested but no model loaded');
        console.error("Inference called but no ONNX model is loaded.");
        statusEl.textContent = " Error: No model loaded.";
        statusEl.className = "disconnected";
        return;
    }
    appendLog(`Running ONNX inference for job ${job.job_id}`);
    statusEl.textContent = " Running ONNX inference";
    statusEl.className = "processing";
    if (infraStatusEl) infraStatusEl.textContent = 'Infra: running inference';

    try {
        const inputData = job.data.input_data;
        const dims = [inputData.length, inputData[0].length]; // e.g., [2, 4]
        const inputTensor = new onnx.Tensor(new Float32Array(inputData.flat()), 'float32', dims);
        
        // Run the model and get the output
        const outputMap = await inferenceSession.run([inputTensor]);
        const outputTensor = outputMap.values().next().value; // Get the first output tensor
        
        const resultPayload = {
            job_id: job.job_id,
            data: Array.from(outputTensor.data)
        };
        socket.send(JSON.stringify(resultPayload));
        appendLog(`ONNX inference complete for job ${job.job_id}, result length=${resultPayload.data.length}`);
        if (infraStatusEl) infraStatusEl.textContent = 'Infra: ready';
        
        statusEl.textContent = " Result sent! Waiting for next job";
        statusEl.className = "connected";
    } catch (error) {
        appendLog('Error during ONNX inference: ' + String(error));
        console.error("Error during ONNX inference:", error);
        statusEl.textContent = ` Error: ${error.message}`;
        statusEl.className = "disconnected";
        if (infraStatusEl) infraStatusEl.textContent = 'Infra: error';
        const errPayload = { job_id: job.job_id, error: String(error) };
        try { socket.send(JSON.stringify(errPayload)); } catch (e) { console.error('Failed to send error payload:', e); }
    }
}

async function handleLoadModel(job) {
    statusEl.textContent = " Receiving and loading model";
    statusEl.textContent = " Model loaded Ready for inference.";
}

async function handleInference(job) {
    if (!model) {
        console.error("Inference called but no model is loaded.");
        return;
    }
    statusEl.textContent = " Running inference";
    // model.predict() logic 
    // send result back via WebSocket
    statusEl.textContent = " Result sent! Waiting for next job";
}

async function handleSquareArray(job) {
    console.log(`Square array job received: ${job.job_id}, length=${job.data.array.length}`);
    statusEl.textContent = " Processing array on GPU";
    const inputArray = new Float32Array(job.data.array);
    let resultArray;
    try {
        resultArray = await runSquareArrayOnGpu(inputArray); // Call your WebGPU function
    } catch (err) {
        console.error('GPU processing failed:', err);
        statusEl.textContent = ` Error: ${err.message}`;
        statusEl.className = "disconnected";
        const errPayload = { job_id: job.job_id, error: String(err) };
        try { socket.send(JSON.stringify(errPayload)); } catch (e) { console.error('Failed to send error payload:', e); }
        return;
    }

    const resultPayload = {
        job_id: job.job_id,
        data: Array.from(resultArray)
    };
    console.log(`Sending result for job ${job.job_id}, bytes=${resultPayload.data.length}`);
    try {
        socket.send(JSON.stringify(resultPayload));
        statusEl.textContent = " Result sent! Waiting for next job";
        statusEl.className = "connected";
    } catch (err) {
        console.error('Failed to send result payload:', err);
        statusEl.textContent = ` Error sending result: ${err.message}`;
        statusEl.className = "disconnected";
    }
}

// other GPU Utility Functions 

function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}


async function runSquareArrayOnGpu(inputArray) {
    if (!navigator.gpu) throw new Error("WebGPU not supported on this browser.");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No appropriate GPUAdapter found.");
    const device = await adapter.requestDevice();

    const shaderCode = `
        @group(0) @binding(0) var<storage, read> input_array: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output_array: array<f32>;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index : u32 = global_id.x;
            if (index >= arrayLength(&input_array)) {
                return;
            }
            let value : f32 = input_array[index];
            output_array[index] = value * value;
        }
    `;

    const elementSize = 4; // bytes per f32
    const length = inputArray.length;
    const bufferSize = length * elementSize;

    // Input buffer (upload)
    const inputBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    new Float32Array(inputBuffer.getMappedRange()).set(inputArray);
    inputBuffer.unmap();

    // Output buffer (gpu only)
    const outputBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    // Buffer to copy results to and map for reading
    const readBuffer = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const shaderModule = device.createShaderModule({ code: shaderCode });

    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: shaderModule,
            entryPoint: "main"
        }
    });

    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } }
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupSize = 64;
    const workgroupCount = Math.ceil(length / workgroupSize);
    passEncoder.dispatchWorkgroups(workgroupCount);

    passEncoder.end();

    // Copy output to a mappable buffer
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, bufferSize);

    const commands = commandEncoder.finish();
    device.queue.submit([commands]);

    // Read back results
    await readBuffer.mapAsync(GPUMapMode.READ);
    const arrayBuffer = readBuffer.getMappedRange().slice(0);
    readBuffer.unmap();

    return new Float32Array(arrayBuffer);
}