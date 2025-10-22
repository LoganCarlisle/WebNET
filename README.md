# WebNet

**A framework to offload compute from cloud notebooks (Colab, Kaggle) to a distributed fleet of devices using webnet.**

This project aims to build a connection between various research notebooks and the computing power of newer iphone chips, and other hardware, done via WebGPU.
Early develoment is focused on using single chips, however hopefully in the future clusters of iphones/chips can be used for federated inferance then maybe for finetuning or maybe even training.

---

## The Vision & Architecture

The system consists of three main components:
1.  **The Client:** A Python library in a cloud notebook that sends compute jobs.
2.  **The Broker:** A central server that routes jobs and results. Might use Ngrok?
3.  **The Worker:** For now since I dont have a Mac to build out the metal code I will be using WebGPU as a worker, the main idea in the future is to engine swap the WebGPU worker with a Metal worker



```
add diagram eventuallu
```
### How to Run the Demo

1.  **Get an ngrok Authtoken:** Sign up for a free account at [ngrok.com](https://ngrok.com) and get your token.
2.  **Open the Colab Notebook:** The entire demo is contained in a single, self-contained notebook.
    * **[Click here to open the example notebook.](client_notebooks/Colab_to_WebGPU_infra.ipynb)**
3.  **add token to secrets** add tokens to secret section in colab or just paste it if its your own private notebook
4.  **Launch the Worker:** Click the link to open the worker in your browser on a device. Working amd gen 5 and iphone 17 from my own testings.
5.  **Run the Job:** Go back to the notebook and run the final cells to see it work!

---

## Project Status

got some infra working, need to scale up now to multiple devices safely ngrok is broker needs to be swapped, and webgpu probaly needs a engine swap to metal for iphones ect, then various functions to train with other probaly need to implement a credit system then but first get a couple devices connected and running infra together or sum.

### How to Run the Demo


---

## 🤝 Looking for an iOS/Metal Collaborator!

**The backend is ready. Now we need to build the high-performance iOS worker.**

My skillset is more in ai research and such, so building out infrastructure and the javasript side is not a focus of mine, and I also do not have access to a Mac to build the native iOS app. I am looking for a collaborators who are passionate about high-performance computing and wants to help out on building the core of this project. To make this program more efficient as webgpu only code will be a bottleneck and limit output of chips. Why use something not efficiently right?

### The Role:
* Architect and build up the native iOS worker app in Swift.
* Write high-performance Metal compute kernels for various math operations (starting with matrix multiplication) normal proof of concept stuff.
* Implement the WebSocket client to communicate with the existing Broker.
* And share any ideas or develoments your interested in!

**If your interested, Open an issue or reach out! will appriciate all the help I can get!**
