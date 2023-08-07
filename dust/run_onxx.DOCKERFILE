ARG VERSION_TAG=21.03-py3

FROM nvcr.io/nvidia/pytorch:${VERSION_TAG}

RUN pip3 install brtdevkit --extra-index-url https://artifactory.bluerivertech.com/artifactory/api/pypi/prod-mesa-libraries/simple
RUN pip3 install onnx onnxruntime-gpu transformers scikit-learn timm wandb matplotlib pycuda

# in CLI
# $ docker build -f ~/git/scripts/dust/run_onxx.DOCKERFILE ~/git/scripts/dust/ -t onnx_tag
# Sending build context to Docker daemon  12.29kB
# Step 1/4 : ARG VERSION_TAG=21.03-py3
# Step 2/4 : FROM nvcr.io/nvidia/pytorch:${VERSION_TAG}
#  ---> ad8c213c76c5
# Step 3/4 : RUN pip3 install brtdevkit --extra-index-url https://artifactory.bluerivertech.com/artifactory/api/pypi/prod-mesa-libraries/simple
#  ---> Using cache
#  ---> c8e33ba1e7c0
# Step 4/4 : RUN pip3 install onnx onnxruntime-gpu transformers scikit-learn timm wandb matplotlib pycuda
#  ---> Using cache
#  ---> 4843ec21e016
# Successfully built 4843ec21e016
# Successfully tagged onnx_tag:latest
# $ sudo docker run -it --rm --gpus all -v ~/data/dust_val_bestmodel.onnx:/models onnx_tag:latest
# docker: Error response from daemon: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running hook #0: error running hook: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'
# nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown.