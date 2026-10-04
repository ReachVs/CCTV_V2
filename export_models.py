import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

try:
    import tf2onnx  # type: ignore[import-untyped, import-not-found] # pyright: ignore[reportMissingImports]
except ImportError:
    tf2onnx = None  # type: ignore

try:
    import tensorrt as trt  # type: ignore[import-untyped, import-not-found] # pyright: ignore[reportMissingImports]
except ImportError:
    trt = None  # type: ignore

def export_yolo_to_tensorrt(model_path="yolov8n.pt"):
    print(f"[Export] Loading YOLO model from '{model_path}'...")
    try:
        from ultralytics import YOLO
        import torch
        
        model = YOLO(model_path)
        
        if torch.cuda.is_available():
            print("[Export] CUDA is available. Exporting YOLO to TensorRT format...")
            # format="engine" is TensorRT format in Ultralytics YOLOv8
            export_path = model.export(format="engine")
            print(f"[Export] YOLO successfully exported to TensorRT at: {export_path}")
        else:
            print("[Export] CUDA not available (Apple Silicon or CPU environment).")
            if torch.backends.mps.is_available():
                print("[Export] macOS MPS (Metal Performance Shaders) is available. Keeping PyTorch format optimized for MPS.")
            else:
                print("[Export] Falling back to standard PyTorch/CPU format.")
    except Exception as e:
        print(f"[Export] Error exporting YOLO model: {e}")

def export_arcface_to_tensorrt(model_name="ArcFace"):
    print(f"[Export] Loading {model_name} model from DeepFace...")
    try:
        from deepface import DeepFace
        keras_model = DeepFace.build_model(model_name)
        
        # In a GPU/TensorRT environment, we convert the Keras model to ONNX, then to TensorRT.
        # Let's check if CUDA is available.
        import torch
        if torch.cuda.is_available():
            print("[Export] CUDA detected. Converting ArcFace Keras model to ONNX...")
            if tf2onnx is None:
                print("[Export] tf2onnx module not installed. Skipping TensorRT conversion.")
                return
            
            import tensorflow as tf
            
            # Export to ONNX
            input_signature = [tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype, name="input_1")]
            onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)
            
            onnx_path = "arcface.onnx"
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"[Export] ArcFace successfully saved to ONNX at: {onnx_path}")
            
            # Convert ONNX to TensorRT engine
            print("[Export] Compiling ONNX to TensorRT engine...")
            if trt is None:
                print("[Export] TensorRT module not installed. Skipping TensorRT engine compilation.")
                return

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("Failed to parse the ONNX file")
            
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1GB
            
            # Build engine
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                raise RuntimeError("Failed to build the TensorRT engine")
                
            engine_path = "arcface.engine"
            with open(engine_path, "wb") as f:
                f.write(serialized_engine)
            print(f"[Export] ArcFace successfully exported to TensorRT engine at: {engine_path}")
        else:
            print("[Export] CUDA not available. Skipping TensorRT engine compilation for ArcFace.")
            print("[Export] Using standard Keras/TensorFlow backend for representation.")
    except Exception as e:
        print(f"[Export] Error exporting ArcFace model: {e}")

if __name__ == "__main__":
    export_yolo_to_tensorrt()
    export_arcface_to_tensorrt()
