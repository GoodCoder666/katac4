## Katac4 Inference Engine

The same Explorer GUI, but running on the [OpenVINO](https://openvino.ai/) backend.

This provides a lightweight solution for users to try out the model. It performs best on modern Intel hardware, and should also work on non-Intel CPUs.

### How to use

1. Clone the repo and switch to this branch:

   ```bash
   git clone https://github.com/GoodCoder666/katac4.git
   git switch inference
   ```

2. Place the ONNX model as `model.onnx` under the root directory. Either download one trained by me from the [Releases](https://github.com/GoodCoder666/katac4/releases), or manually convert from a PT model with `export_onnx.py`.

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the main program:

   ```bash
   python main.py
   ```

