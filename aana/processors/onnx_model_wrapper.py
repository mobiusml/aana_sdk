import onnxruntime

import numpy as np
import logging


class ONNXModel:
    def __init__(
        self,
        model,
        batch_size,
        input_size,
        output_order=None,
        **kwargs,
    ):
        self.det_model = onnxruntime.InferenceSession(
            model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        logging.info("Detector started")
        # Check if CUDA is available in the session
        available_providers = self.det_model.get_providers()
        print("Available providers:", available_providers)

        self.input = self.det_model.get_inputs()[0]
        self.input_dtype = self.input.type
        if self.input_dtype == "tensor(float)":
            self.input_dtype = np.float32
        else:
            self.input_dtype = np.uint8

        self.output_order = output_order
        self.out_shapes = None
        self.input_shape = (
            batch_size,
            3,
            input_size,
            input_size,
        )  # input_size should be in [640, 1280]

    # warmup
    def prepare(self, **kwargs):
        # import pdb; pdb.set_trace()
        logging.info("Warming up face detection ONNX Runtime engine...")
        if self.output_order is None:
            self.output_order = [e.name for e in self.det_model.get_outputs()]
        self.out_shapes = [e.shape for e in self.det_model.get_outputs()]

        self.det_model.run(
            self.output_order,
            {
                self.det_model.get_inputs()[0].name: [
                    np.zeros(tuple(self.input_shape[1:]), self.input_dtype)
                ]
            },
        )

    def run(self, input):
        net_out = self.det_model.run(self.output_order, {self.input.name: input})
        return net_out
