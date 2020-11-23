## Build onnxruntime MetaWareNN Execution Provider

### Prerequisites 
   1. `git clone --recursive https://github.com/SowmyaDhanapal/onnxruntime.git`  
   2. Install cmake-3.13 or higher
   3. Install gcc/g++ 7

### Steps to build
   1. ` cd /path/to/onnxruntime`  
   2. `./build.sh --config RelWithDebInfo --build_shared_lib --parallel --use_metawarenn`  
   3. Set library path  
      `export LD_LIBRARY_PATH=/path/to/onnxruntime/build/Linux/RelWithDebInfo:$LD_LIBRARY_PATH`  
    
## Compile and run the Inference Script 
   1. Download the model at https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx  
   2. `cd /path/to/onnxruntime/onnxruntime/core/providers/metawarenn/inference`
   3. Update the model path in script `inference.cpp` at line 51 
   4. Build script  
      `g++ -L/path/to/onnxruntime/build/Linux/RelWithDebInfo -o inference inference.cpp -I/path/to/onnxruntime/include/onnxruntime/core/providers/metawarenn -I/path/to/onnxruntime/include/onnxruntime/core/session -lonnxruntime -std=c++14`  
   5. Run the executable  
      `./inference`  
