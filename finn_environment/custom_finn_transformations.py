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
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import (
    make_build_dir,
    pynq_native_port_width,
    pynq_part_map,
    resolve_xilinx_tool,
)


class MoveMulPastAvgPool(Transformation):
    """
    Based on the FINN MoveMulPastMaxPool transformation
    Move non-negative scalar or channelwise mul operations past avg pool operations.
    We want to have muls next to each other such that they can be collapsed into a
    single mul.
    """

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Mul" and not model.is_fork_node(n) and not model.is_join_node(n):
                consumer = model.find_consumer(n.output[0])
                if (
                    consumer is not None
                    and consumer.op_type == "AveragePool"
                    and not model.is_join_node(consumer)
                ):
                    mul_weight_name = n.input[1]
                    A = model.get_initializer(mul_weight_name)
                    if A is None:
                        warnings.warn(
                            """Mul weight tensor is not set. If it is a constant,
                                please use set_initializer to set the tensor."""
                        )
                        continue
                    avgpool_node = consumer
                    mul_node = n
                    start_name = mul_node.input[0]
                    avgpool_in_name = avgpool_node.input[0]
                    avgpool_in_shape = model.get_tensor_shape(avgpool_in_name)
                    ifm_ch = avgpool_in_shape[1]
                    avgpool_out_name = avgpool_node.output[0]
                    avgpool_out_shape = model.get_tensor_shape(avgpool_out_name)

                    # do not support non-2D avgpool
                    kernel_shape = list(get_by_name(avgpool_node.attribute, "kernel_shape").ints)
                    if len(kernel_shape) != 2:
                        continue

                    # do not move negative multiplication factor(s)
                    if (A < 0).any():
                        continue

                    if all(x == 1 for x in A.shape) or A.shape == (1, ifm_ch, 1, 1):
                        # if the mul is scalar or channelwise,
                        # we can simply swap the order of ops
                        # rewire mul input to be avgpool input
                        avgpool_node.input[0] = start_name
                        model.set_tensor_shape(start_name, avgpool_in_shape)
                        model.set_tensor_datatype(start_name, DataType["FLOAT32"])
                        # use old avgpool input tensor as avgpool output
                        avgpool_node.output[0] = avgpool_in_name
                        model.set_tensor_shape(avgpool_in_name, avgpool_out_shape)
                        model.set_tensor_datatype(avgpool_in_name, DataType["FLOAT32"])
                        # use new avgpool output as new mul node input
                        mul_node.input[0] = avgpool_in_name
                        # use old avgpool output as new mul node output
                        mul_node.output[0] = avgpool_out_name
                        model.set_tensor_datatype(avgpool_out_name, DataType["FLOAT32"])
                        # move mul node past avgpool node
                        graph.node.remove(mul_node)
                        graph.node.insert(node_ind, mul_node)
                        graph_modified = True
        model = model.transform(InferShapes())
        return (model, graph_modified)


class MoveScalarLinearPastConcat(Transformation):
    """
    Modification of the FINN MoveLinearPastEltwiseAdd transformation.
    Move Scalar Add and Scalar Mul operations past Concat operations.
    """

    def move_node(self, graph, model, n, prods, node_ind):

        # found! move one of the muls to output, remove the others
        lin0_inputs = [prod.input[0] for prod in prods]
        in0 = n.input[0]
        out = n.output[0]

        # Store the original Concat output input shape to set the shape of the new input tensor after moving the Mul node past the Concat node
        concat_out_shape = model.get_tensor_shape(out)

        # connect the Concat inputs to mul inputs
        for i in range(len(prods)):
            n.input[i] = lin0_inputs[i]

        # connect mul0 output to concat output
        prods[0].output[0] = out

        # Change the input of the future input of mul0 and output of Concat to the original input of Concat
        if concat_out_shape is not None:
            model.set_tensor_shape(in0, concat_out_shape)

        # connect the input of mul0 and output of Concat together
        n.output[0] = in0
        prods[0].input[0] = in0

        # move prods[0] node past Concat node, and remove the others
        for prod in prods:
            graph.node.remove(prod)
        graph.node.insert(node_ind - 2, prods[0])

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        nodes = [n for n in graph.node]
        for n in nodes:
            node_ind += 1
            if n.op_type == "Concat":

                # check for tensors on all Concat inputs
                inputs = [input for input in n.input]
                if len(inputs) < 2:
                    continue
                if any(input is None for input in inputs):
                    continue
                in_inits = [model.get_initializer(input) for input in inputs]
                if any(init is not None for init in in_inits):
                    continue

                # check for mul with same initializer on all inputs
                prods = [model.find_producer(input) for input in inputs]

                # check if any branches are empty (i.e., no producer)
                if any(prod is None for prod in prods):
                    continue

                # check if branches come from the same node (i.e., prod0 == prod1)
                ref_prod = prods[0]
                if any(prod == ref_prod for prod in prods[1:]):
                    continue

                # check if all producers inputs are valid (i.e., have at least 2 inputs)
                if any(len(prod.input) < 2 for prod in prods):
                    continue
                inits = [model.get_initializer(prod.input[1]) for prod in prods]

                # if any initializer is None, skip
                if any(init is None for init in inits):
                    continue

                # Check if all initializer are scalars
                if any(init.size != 1 for init in inits):
                    continue

                # check if all producers are Mul and equal or Add and equal. If so move one of the Mul nodes past the Concat and remove the others.
                if all(prod.op_type == "Mul" for prod in prods) or all(prod.op_type == "Add" for prod in prods):
                    ref_init = inits[0]
                    if all(np.array_equal(init, ref_init) for init in inits[1:]):
                        self.move_node(graph, model, n, prods, node_ind)
                        node_ind -= 1
                        graph_modified = True
                else:
                    continue
        model = model.transform(InferShapes())
        return (model, graph_modified)


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
    

class MakeConcatNHWC(Transformation):
    """
    Converts the inputs and outputs for all Concat Layers from NCHW to NHWC.
    Only proceeds if the Concat all producers and consumers have op_type "Transpose".
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Concat":
                ishape = model.get_tensor_shape(n.input[0])
                old_axis = get_by_name(n.attribute, "axis").i

                # Verify if the Concat axis is not on the last dimension (C) for NCHW layout
                if old_axis == -1 or old_axis == len(ishape) - 1:
                    warnings.warn(
                        "%s: Concat axis is on the last dimension (C) for NCHW layout. Can't operate transformation on node." % n.name
                    )
                    continue

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

                # If all checks passed, we can proceed to convert the Concat node from NCHW to NHWC
                for attr in n.attribute:
                    if attr.name == "axis":
                        attr.i = len(ishape) - 1  # Set the axis to the last dimension (C) for NHWC layout

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

        return (model, False)


class RemoveUselessMultiThresholds(Transformation):
    """
    DO NOT USE: THIS TRANSFORMATION CANNOT BE DONE BECAUSE IT WILL BREAK THE MODEL. IT IS KEPT HERE FOR FUTURE REFERENCE.
    When there is two (or more) MultiThreshold nodes in a row and the the first one consumer is exclusively the second one, the first MultiThreshold nodes is useless and can be removed.
    This can reduce significantly the LUT usage in the final design, since MultiThreshold nodes are implemented with LUTs.
    """

    def apply(self, model):
        nodes_to_remove = []
        graph_modified = False

        for node in model.graph.node:
            if node.op_type == "MultiThreshold":

                # Check if the producer is also a MultiThreshold node
                producer = model.find_producer(node.input[0])
                if producer is None or producer.op_type != "MultiThreshold":
                    #warnings.warn(f"RemoveUselessMultiThresholds: Skipping node {node.name} because its producer is not a MultiThreshold node.")
                    continue

                # Check if the producer has one consumer
                prod_consumers = model.find_consumers(producer.output[0])
                if len(prod_consumers) > 1:
                    warnings.warn(f"RemoveUselessMultiThresholds: Skipping node {node.name} because its producer has more than one consumer.")
                    continue

                ## If we reach this point, then proceed to remove the current MultiThreshold node and rewire the graph accordingly

                # Rewires the input of the current node with the input of the repeated MultiThreshold (producer) node
                node.input[0] = producer.input[0]
                # Add the producer node to the list of nodes to remove
                nodes_to_remove.append(producer)
                # Set the graph_modified flag to True since we are modifying the graph
                graph_modified = True

        # Remove the useless MultiThreshold nodes from the graph
        for node in nodes_to_remove:
            model.graph.node.remove(node)
        return model, graph_modified


class StrictRoundAndClipThresholds(Transformation):
    """
    THIS IS A COPY OF THE FINN MOST RECENT RoundAndClipThresholds FUNCTION ON GITHUB. THE ONE STORES IN MY CONTAINER IS OUTDATED AND DOES NOT WORK PROPERLY.
    For MultiThreshold, Thresholding, MVAU, and VVAU nodes operating on integer inp/accumulators,
    round up (ceil) threshold values to the nearest integer and clip to valid range.
    Type-casts thresholds (back) to the float32 container type (this is separate from the
    quantization annotation). Runs InferDataTypes() afterward to propagate any changes to the
    quantization data types."""

    def apply(self, model: ModelWrapper):  # noqa
        graph = model.graph
        graph_modified = False
        for index, node in enumerate(graph.node):
            op_type = node.op_type
            if op_type == "MultiThreshold" or op_type.startswith("Thresholding"):
                thresholds = model.get_initializer(node.input[1])
                if thresholds is None:
                    continue
                dtype = model.get_tensor_datatype(node.input[0])
                # This transformation only applies to thresholding operations
                # operating on integer inputs
                if not (dtype.is_integer() or dtype.is_fixed_point()):
                    continue
                if dtype.is_integer():
                    # Round thresholds up to nearest integer and clip thresholds
                    # outside the input range
                    #   Note: This might promote the thresholds to float64 and
                    #   introduce extra inaccuracies due to large integers not being
                    #   exactly representable in floating-point representation.
                    #   See for example: np.ceil(np.float32(16777217)) == 16777216
                    new_thresholds = np.clip(np.ceil(thresholds), dtype.min(), dtype.max() + 1)
                    # Convert back to the preferred float32 container type
                    new_thresholds = new_thresholds.astype(np.float32)
                    # Insert the rounded and clipped thresholds back into the model
                    model.set_initializer(node.input[1], new_thresholds)
                    # The rounded and clipped thresholds now fit into a data type
                    # that is one bit bigger than the input datatype
                    # Determine new max_value
                    max_val = dtype.max() + 1
                    if not dtype.signed():
                        tdt = DataType.get_smallest_possible(max_val)
                    else:
                        tdt = DataType.get_smallest_possible(-(max_val) - 1)
                elif dtype.is_fixed_point():
                    # Round thresholds up to nearest representable value
                    # of the input datatype
                    new_thresholds = np.clip(
                        np.ceil(thresholds / dtype.scale_factor()) * dtype.scale_factor(),
                        dtype.min(),
                        dtype.max() + dtype.scale_factor(),
                    )
                    new_thresholds = new_thresholds.astype(np.float32)
                    model.set_initializer(node.input[1], new_thresholds)
                    # find smallest underlying integer representation for the thresholds
                    max_val = dtype.max() / dtype.scale_factor() + 1
                    tdt_int = DataType.get_smallest_possible(-max_val - 1)
                    tdt = DataType[
                        f"FIXED<{tdt_int.bitwidth()},{tdt_int.bitwidth() - dtype.frac_bits()}>"
                    ]

                model.set_tensor_datatype(node.input[1], tdt)
                # If hw op we need to set the weight data type attribute as well
                if op_type.startswith("Thresholding"):
                    inst = getCustomOp(node)
                    inst.set_nodeattr("weightDataType", tdt.name)
                # ones
                if np.any(new_thresholds != thresholds):
                    # Track the graph has been modified to inform the transform
                    # container to exhaustively repeat this transformation until
                    # no changes are possible
                    graph_modified = True
                    # Immediately exit here to propagate the data type changes
                    # before considering the next node
                    break

            # Handle MVAU and VVAU nodes with thresholds (noActivation=0)
            elif op_type.startswith("MVAU") or op_type.startswith("VVAU"):
                inst = getCustomOp(node)
                # Only process if node has thresholds (noActivation=0)
                if inst.get_nodeattr("noActivation") == 0 and len(node.input) > 2:
                    thresholds = model.get_initializer(node.input[2])
                    if thresholds is None:
                        continue

                    # Get accumulator datatype (should be set by MinimizeAccumulatorWidth)
                    acc_dt = DataType[inst.get_nodeattr("accDataType")]

                    # This transformation only applies to integer accumulators
                    if not acc_dt.is_integer():
                        continue

                    # Round thresholds up to nearest integer and clip to accumulator range
                    new_thresholds = np.clip(np.ceil(thresholds), acc_dt.min(), acc_dt.max() + 1)
                    # Convert back to the preferred float32 container type
                    new_thresholds = new_thresholds.astype(np.float32)
                    # Insert the rounded and clipped thresholds back into the model
                    model.set_initializer(node.input[2], new_thresholds)
                    # The rounded and clipped thresholds now fit into a data type
                    # that is one bit bigger than the accumulator datatype
                    # Determine new max_value
                    max_val = acc_dt.max() + 1
                    if not acc_dt.signed():
                        tdt = DataType.get_smallest_possible(max_val)
                    else:
                        tdt = DataType.get_smallest_possible(-(max_val) - 1)

                    
                    model.set_tensor_datatype(node.input[2], tdt)

                    if np.any(new_thresholds != thresholds):
                        # Track the graph has been modified to inform the transform
                        # container to exhaustively repeat this transformation until
                        # no changes are possible
                        graph_modified = True
                        # Immediately exit here to propagate the data type changes
                        # before considering the next node
                        break

        model = model.transform(InferDataTypes())
        return model, graph_modified


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


class SetGlobalFIFODepths(Transformation):
    def __init__(self, depth=64):
        super().__init__()
        self.depth = depth

    def apply(self, model):
        for node in model.graph.node:
            prefix = node.name + "_"
            if "fpgadataflow" in node.domain:
                inst = getCustomOp(node)
                try:
                    num_in = len(inst.get_nodeattr("inFIFODepths"))
                    inst.set_nodeattr("inFIFODepths", [self.depth] * num_in)
                except KeyError:
                    pass
                try:
                    num_out = len(inst.get_nodeattr("outFIFODepths"))
                    inst.set_nodeattr("outFIFODepths", [self.depth] * num_out)
                except KeyError:
                    pass
        return (model, False)


class PrepareModelForFPGADesign(Transformation):
    '''
    Partially runs the ZynqBuild Transform to make it possible tu run estimates before actual build.
    '''
    def __init__(self, platform="Pynq-Z2"):
        super().__init__()
        self.fpga_part = pynq_part_map[platform]
        self.axi_port_width = pynq_native_port_width[platform]
        
    def apply(self, model):
        # first infer layouts
        model = model.transform(InferDataLayouts())
        # prepare at global level, then break up into kernels
        prep_transforms = [
            InsertIODMA(self.axi_port_width),
            InsertDWC(),
            SpecializeLayers(self.fpga_part),
            Floorplan(),
        ]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        # Build each kernel individually
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        for sdp_node in sdp_nodes:
            prefix = sdp_node.name + "_"
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_model = kernel_model.transform(InsertFIFO())
            kernel_model = kernel_model.transform(SpecializeLayers(self.fpga_part))
            kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
        return (model, False)