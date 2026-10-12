import numpy as np
import qonnx.core.data_layout as DataLayout
import warnings

from onnx import TensorProto
from onnx import helper as oh
from onnx import helper

from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.data_layout as DataLayout

from finn.transformation.fpgadataflow.specialize_layers import _determine_impl_style
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from shutil import copy

from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.util.fpgadataflow import is_hls_node, is_rtl_node
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from qonnx.transformation.general import SortGraph
from finn.util.basic import (
    make_build_dir,
    pynq_native_port_width,
    pynq_part_map,
    resolve_xilinx_tool,
)

class RevertNCHWElementwiseToONNX(Transformation):
    """
    Search for ElementwiseAdd and ElementwiseMul nodes with NCHW format on inputs and outputs to corresponding ONNX Add and Mul nodes.
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        for idx, n in enumerate(graph.node):
            if n.op_type in ["ElementwiseMul", "ElementwiseAdd"]:
                # Find the input and output tensor layouts for the ElementwiseAdd or ElementwiseMul node
                inp_layouts = [model.get_tensor_layout(inp) for inp in n.input]
                inp_layouts = [layout for layout in inp_layouts if layout is not None]  # Remove None values from the list of input layouts
                out_layout = model.get_tensor_layout(n.output[0])

                # Check if the input and output tensor layouts are all NCHW
                is_nchw = all(in_layout == DataLayout.NCHW for in_layout in inp_layouts) and (out_layout == DataLayout.NCHW)
                if not is_nchw:
                    warnings.warn(
                        "%s: Not all input and output tensor layouts are NCHW. Can't operate transformation on node." % n.name
                    )
                    continue

                # If all checks passed, we can proceed to convert the ElementwiseAdd or ElementwiseMul node to the corresponding ONNX Add or Mul node
                base_op = n.op_type.replace("Elementwise", "")
                
                new_node = helper.make_node(
                    base_op,
                    n.input,
                    n.output,
                    name=n.name
                )
                # Substitui no mesmo índice para manter a topologia
                graph.node.remove(n)
                graph.node.insert(idx, new_node)
                graph_modified = True
                break

        return (model, graph_modified)


class MakeElementwiseAddNHWC(Transformation):
    """
    Converts the inputs and outputs for all ElementwiseAdd Layers from NCHW to NHWC.
    Only proceeds if the Concat all producers and consumers have op_type "Transpose".
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "ElementwiseAdd":
                consumer = model.find_consumer(n.output[0])
                producers = [model.find_producer(inp) for inp in n.input]

                # Verify if all producers and the consumer are not None and have op_type "Transpose"
                if not all(producer is not None and producer.op_type == "Transpose" for producer in producers):
                    warnings.warn(
                        "%s: Not all producers are Transpose. Can't operate transformation on node." % n.name
                    )
                    continue
                if consumer is None or consumer.op_type != "Transpose":
                    warnings.warn(
                        "%s: Consumer is not Transpose. Can't operate transformation on node." % n.name
                    )
                    continue

                # Verify if all producers have (N, H, W, C) -> (N, C, H, W) transpose pattern
                if not all(list(get_by_name(producer.attribute, "perm").ints) == [0, 3, 1, 2] for producer in producers):
                    warnings.warn(
                        "%s: Not all producers have (N, H, W, C) -> (N, C, H, W) transpose pattern. Can't operate transformation on node." % n.name
                    )
                    continue
                # Verify if the consumer have (N, C, H, W) -> (N, H, W, C) transpose pattern
                if not list(get_by_name(consumer.attribute, "perm").ints) == [0, 2, 3, 1]:
                    warnings.warn(
                        "%s: Consumer does not have (N, C, H, W) -> (N, H, W, C) transpose pattern. Can't operate transformation on node." % n.name
                    )
                    continue

                # If all checks passed, we can proceed to convert the ElementwiseAdd node from NCHW to NHWC
                new_nhwc_shape = model.get_tensor_shape(consumer.output[0])
                for attr in n.attribute:
                    # Set the inputs and outputs to the NHWC layout
                    if attr.name in ["lhs_shape", "rhs_shape", "out_shape"]:
                        inst = getCustomOp(n)
                        inst.set_nodeattr(attr.name, new_nhwc_shape)

                # And then proceed to erase the Transpose nodes and rewire the graph accordingly
                start_names = [producer.input[0] for producer in producers]
                end_name = consumer.output[0]
                n.output[0] = end_name
                for i in range(len(producers)):
                    n.input[i] = start_names[i]

                graph.node.remove(consumer)
                for producer in producers:
                    graph.node.remove(producer)

                # Break the loop after modifying the graph to avoid issues with iterating over a modified list of nodes
                graph_modified = True
                break

        return (model, graph_modified)


class InsertFIFODepths(Transformation):
    """Insert deep (=tensor size or =fixed value) FIFOs between all fpgadataflow nodes
    Implementation based on the FINN InsertAndSetFIFODepths() transformation

    Constructor arguments:
    :parameter clk_ns: clock period (used for IP preparation)
    :parameter fixed_depth: the fixed depth of the FIFOs to be inserted
    :parameter ram_style: the preferred RAM style to be used for the FIFOs to be inserted

    Assumed input graph properties:
    - all nodes are fpgadataflow nodes
    - no FIFOs inserted,
    - (inFIFODepths/outFIFODepths attrs will be ignored)

    Output:
    - graph with estimated-depth FIFOs inserted"""

    def __init__(
        self,
        fpgapart,
        fixed_depth=None,
        ram_style="auto"
    ):
        super().__init__()
        self.fpgapart = fpgapart
        self.fixed_depth = fixed_depth
        self.ram_style = ram_style

    def apply(self, model):
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        for node in model.graph.node:
            # verify assumptions
            assert is_hls_node(node) or is_rtl_node(node), "Found non-fpgadataflow node: " + str(
                node
            )
            op_type = node.op_type
            assert not op_type.startswith("StreamingFIFO"), "Found existing StreamingFIFO node"
            node = getCustomOp(node)
            ifd = node.get_nodeattr("inFIFODepths")
            ofd = node.get_nodeattr("outFIFODepths")
            if self.fixed_depth is not None:
                ifd = [self.fixed_depth] * len(ifd)
                ofd = [self.fixed_depth] * len(ofd)
            else:
                # set each FIFO to its tensor size
                # (except stream width hence the :-1)
                for i in range(len(ifd)):
                    # safe guard that for very small tensors depth is not set to 1
                    tensor_size = np.prod(node.get_folded_input_shape(i)[:-1])
                    ifd[i] = tensor_size if tensor_size > 1 else 2
                for o in range(len(ofd)):
                    # safe guard that for very small tensors depth is not set to 1
                    tensor_size = np.prod(node.get_folded_output_shape(o)[:-1])
                    ofd[o] = tensor_size if tensor_size > 1 else 2
            # set node attribute and ensure that it gets saved as list of integers
            node.set_nodeattr("inFIFODepths", [int(fifo) for fifo in ifd])
            node.set_nodeattr("outFIFODepths", [int(fifo) for fifo in ofd])

        # insert stream infrastructure (DWC/FIFO)
        model = model.transform(InsertDWC())
        model = model.transform(InsertFIFO(create_shallow_fifos=True, ram_style=self.ram_style))
        model = model.transform(SpecializeLayers(self.fpgapart))
        model = model.transform(SortGraph())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(AnnotateCycles())

        return (model, False)


class RedefineImplStyle(Transformation):
    """Convert hls to rlt or vice versa based on the init parameters. Based on the FINN SpecializeLayers transformation."""

    def __init__(self, impl_style, fpgapart):
        super().__init__()

        if impl_style not in ["hls", "rtl"]:
            raise ValueError("impl_style must be either 'hls' or 'rtl'")

        self.new_impl_style = impl_style
        self.old_impl_style = 'hls' if impl_style == 'rtl' else 'rtl'
        self.fpgapart = fpgapart

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        for node in graph.node:

            # Skip nodes that are not the old implementation style layers
            if not node.domain == f"finn.custom_op.fpgadataflow.{self.old_impl_style}":
                node_ind += 1
                continue
            node_inst = getCustomOp(node)
            # If the node already has the new preferred impl style, but is still in the old impl style, we skip it.
            if node_inst.get_nodeattr("preferred_impl_style") == self.new_impl_style:
                node_ind += 1
                continue

            # First we set the preferred_impl_style attribute to the new implementation style, so that the node can be re-specialized to the new implementation style
            node_inst.set_nodeattr("preferred_impl_style", self.new_impl_style)

            # The name, op_type and domain are set to hls or rtl, so we replace it with base_op_type, so that the node can be re-specialized to the new implementation style
            base_name = node.name.replace(f"_{self.old_impl_style}", "")
            node.name = base_name
            base_op_type = node.op_type.replace(f"_{self.old_impl_style}", "")
            node.op_type = base_op_type
            node.domain = "finn.custom_op.fpgadataflow"

            # For shuffle nodes the specialisation happens after
            # the ShuffleDecomposition transformation with a
            # dedicated InferInnerOuterShuffle transformation
            if node.op_type == "Shuffle":
                node_ind += 1
                continue
            impl_style = _determine_impl_style(node, self.fpgapart, model)
            optype = node.op_type + "_" + impl_style

            new_node = helper.make_node(
                optype,
                node.input,
                node.output,
                name=node.name,
                domain="finn.custom_op.fpgadataflow." + impl_style,
            )

            # List of old attributes that must be removed from the new node, since they are not used anymore in the new implementation style
            stale_attrs = [
                "code_gen_dir_ipgen", 
                "ip_path", 
                "ip_vlnv", 
                "ipgen_path"
            ]

            # add all attributes
            for attribute in node.attribute:
                if attribute.name in stale_attrs:
                    # Create a new attribute with an empty string as its value
                    clean_attr = helper.make_attribute(attribute.name, "")
                    new_node.attribute.append(clean_attr)
                else:
                    new_node.attribute.append(attribute)
            graph.node.insert(node_ind, new_node)
            # remove old nodes
            graph.node.remove(node)
            # Stop the loop since we changed the ONNX graph size.
            # This is necessary to avoid issues with the iteration over the graph nodes.
            return (model, True)
        return (model, False)