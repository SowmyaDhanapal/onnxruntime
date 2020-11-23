### To Build the MetaWareNN Execution Provider in ONNXRuntime
    * cd /path/to/onnxruntime
    * ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_metawarenn
### Set Library Path 
    export LD_LIBRARY_PATH=/path/to/onnxruntime/build/Linux/RelWithDebInfo:$LD_LIBRARY_PATH
### Command to Build the Executable to Run Inference 
    * cd /path/to/onnxruntime/onnxruntime/core/providers/metawarenn/inference
    * g++ -L/path/to/onnxruntime/build/Linux/RelWithDebInfo -o inference inference.cpp -I/path/to/onnxruntime/include/onnxruntime/core/providers/metawarenn -I/path/to/onnxruntime/include/onnxruntime/core/session -lonnxruntime -std=c++14
### Run the Executable
    ./inference
#### Note:
Update line number: 51 in inference.cpp with model path
