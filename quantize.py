# extracted from https://github.com/xenova/transformers.js/blob/main/scripts/convert.py
import sys
import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from typing import Optional, Set
import json


def get_operators(model: onnx.ModelProto) -> Set[str]:
    operators = set()

    def traverse_graph(graph):
        for node in graph.node:
            operators.add(node.op_type)
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    subgraph = attr.g
                    traverse_graph(subgraph)

    traverse_graph(model.graph)
    return operators


def quantize(model_names_or_paths, **quantize_kwargs):
    quantize_config = dict(**quantize_kwargs, per_model_config={})

    for model in model_names_or_paths:
        directory_path = os.path.dirname(model)
        file_name_without_extension = os.path.splitext(os.path.basename(model))[0]

        loaded_model = onnx.load_model(model)
        op_types = get_operators(loaded_model)
        weight_type = QuantType.QUInt8 if "Conv" in op_types else QuantType.QInt8

        quantize_dynamic(
            model_input=model,
            model_output=os.path.join(
                directory_path, f"{file_name_without_extension}_quantized.onnx"
            ),
            weight_type=weight_type,
            optimize_model=False,
            extra_options=dict(EnableSubgraph=True),
            **quantize_kwargs,
        )

        quantize_config["per_model_config"][file_name_without_extension] = dict(
            op_types=list(op_types),
            weight_type=str(weight_type),
        )

    # Save quantization config
    with open(os.path.join(directory_path, "..", "quantize_config.json"), "w") as fp:
        json.dump(quantize_config, fp, indent=4)


if __name__ == "__main__":
    onnx_dir = sys.argv[-1]

    models = [
        os.path.join(onnx_dir, m) for m in os.listdir(onnx_dir) if m.endswith(".onnx")
    ]

    quantize(models)
