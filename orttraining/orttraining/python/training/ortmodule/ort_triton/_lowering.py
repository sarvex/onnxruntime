# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import itertools
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import sympy
from onnx import NodeProto

from ._common import TensorInfo
from ._ir import (
    ComputeNode,
    DropoutNode,
    ElementwiseKernelNode,
    IONode,
    KernelNode,
    ModuleNode,
    OffsetCalculator,
    RecomputeEnd,
    RecomputeStart,
    ReduceKernelNode,
    ReduceNode,
    TensorArg,
)
from ._op_config import is_reduction_node
from ._sorted_graph import SortedGraph
from ._utils import get_reduce_info, to_numpy_array


class NodeGroup:
    def __init__(self, node: NodeProto, reduce_axis: int, node_arg_infos: Dict[str, TensorInfo]):
        self._node_arg_infos = node_arg_infos
        self.nodes_groups: List[Any] = [node]
        self.target_shape: List[sympy.Expr] = self._get_target_shape(node)
        self.reduce_axis = reduce_axis
        reduce_dim = self.target_shape[self.reduce_axis] if self.reduce_axis != -1 else None
        self.recompute = reduce_dim is not None and (not reduce_dim.is_number or reduce_dim > sympy.Integer(1024))

    # Check if shape can be broadcasted to target_shape.
    def _compatable_shape(self, shape: List[sympy.Expr]) -> bool:
        if len(shape) > len(self.target_shape):
            return False
        shape = [sympy.Integer(1)] * (len(self.target_shape) - len(shape)) + shape
        for axis in range(len(shape)):
            if shape[axis] != self.target_shape[axis] and (
                not shape[axis].is_number or shape[axis] != sympy.Integer(1)
            ):
                return False
        return True

    def _get_target_shape(self, node):
        name = node.input[0] if is_reduction_node(node) else node.output[0]
        return self._node_arg_infos[name].shape

    def compatible(self, node: NodeProto, keep_dims: int, reduce_axis: int) -> bool:
        target_shape = self._get_target_shape(node)
        if is_reduction_node(node):
            if keep_dims != 1:
                return False
            if (self.reduce_axis != -1 and self.reduce_axis != reduce_axis) or self.target_shape != target_shape:
                return False
            return True
        return self._compatable_shape(target_shape)

    def add_node(self, node: NodeProto, reduce_axis):
        if is_reduction_node(node):
            group = NodeGroup(node, reduce_axis, self._node_arg_infos)
            self.nodes_groups.append(group)
            if self.reduce_axis == -1:
                self.reduce_axis = group.reduce_axis
                reduce_dim = self.target_shape[self.reduce_axis]
                self.recompute = not reduce_dim.is_number or reduce_dim > sympy.Integer(1024)
            return group
        self.nodes_groups.append(node)
        return self

    def dependent_nodes(self, keep_reduce_node: bool):
        node_map = dict()
        reduce_node = None
        for idx, item in enumerate(self.nodes_groups):
            if isinstance(item, NodeGroup):
                node_map.update(item.dependent_nodes(keep_reduce_node)[0])
            elif keep_reduce_node or (idx != 0 or not is_reduction_node(item)):
                node_map[item.name] = item
            else:
                reduce_node = item
        return node_map, reduce_node

    def flatten(self, sorted_nodes: List[NodeProto]) -> Tuple[List[NodeProto], List[List[int]]]:
        if self.recompute:
            layers = []
            group_layer = [self]
            while len(group_layer) > 0:
                node_map = dict()
                reduce_nodes = []
                next_layer = []
                for group in group_layer:
                    sub_node_map, reduce_node = group.dependent_nodes(False)
                    node_map.update(sub_node_map)
                    if reduce_node is not None:
                        reduce_nodes.append(reduce_node)
                    next_layer.extend([item for item in group.nodes_groups if isinstance(item, NodeGroup)])
                layers.append((node_map, reduce_nodes))
                group_layer = next_layer
            nodes = []
            layer_indices = []
            for i in range(len(layers) - 1, -1, -1):
                sub_nodes = list(layers[i][0].values())
                sub_nodes.sort(key=sorted_nodes.index)
                nodes.extend(sub_nodes)
                sub_layer_indices = []
                for node in layers[i][1]:
                    nodes.append(node)
                    sub_layer_indices.append(len(nodes) - 1)
                layer_indices.append(sub_layer_indices)
            return nodes, layer_indices
        node_map, _ = self.dependent_nodes(True)
        nodes = list(node_map.values())
        nodes.sort(key=sorted_nodes.index)
        return nodes, []


class KernelIO:
    def __init__(self):
        self.module_inputs: List[str] = []
        self.cross_kernel_inputs: List[str] = []
        self.constants: List[str] = []
        self.module_outputs: List[str] = []
        self.cross_kernel_outputs: [str] = []
        self.internal_args: List[str] = []


class GraphLowering:
    def __init__(self, sorted_graph: SortedGraph):
        self._sorted_graph: SortedGraph = sorted_graph
        self._node_arg_infos: Dict[str, TensorInfo] = sorted_graph.node_arg_infos
        self._module_inputs: List[TensorArg] = []
        self._module_outputs: List[TensorArg] = []
        self._module_constants: List[TensorArg] = []
        self._module_input_names: Set[str] = set()
        self._module_output_names: Set[str] = set()
        self._module_constant_names: Set[str] = set()
        self._tensor_args: Dict[str, TensorArg] = {}
        self._extract_module_io()

        self._groups: List[NodeGroup] = []
        self._group_nodes()

        self._kernel_nodes: List[KernelNode] = []
        self._kernel_io_list: List[KernelIO] = []
        self._lower()

    def _extract_module_io(self):
        graph = self._sorted_graph.original_graph
        self._module_inputs = [TensorArg(input.name, self._node_arg_infos[input.name]) for input in graph.input]
        self._module_input_names = set(arg.name for arg in self._module_inputs)
        self._module_outputs = [TensorArg(output.name, self._node_arg_infos[output.name]) for output in graph.output]
        self._module_output_names = set(arg.name for arg in self._module_outputs)
        for initializer in graph.initializer:
            data = to_numpy_array(initializer)
            self._module_constants.append(TensorArg(initializer.name, data=data))
        for const_node in self._sorted_graph.const_nodes:
            data = to_numpy_array(const_node)
            self._module_constants.append(TensorArg(const_node.output[0], data=data))
        self._module_constant_names = set(arg.name for arg in self._module_constants)
        self._tensor_args = dict(
            (arg.name, arg)
            for arg in itertools.chain(self._module_inputs, self._module_outputs, self._module_constants)
        )

    def _get_reduce_info(self, node) -> Tuple[int, int]:
        assert is_reduction_node(node)
        input_rank = len(self._node_arg_infos[node.input[0]].shape)
        keep_dims, axes = get_reduce_info(node, self._sorted_graph.original_graph, input_rank)
        assert len(axes) == 1
        axis = axes[0]
        if axis < 0:
            axis += input_rank
        return keep_dims, axis

    def _process_node(self, node: NodeProto, precessors: Dict[str, List[NodeProto]], group: NodeGroup):
        dependent_nodes = set()
        dependent_nodes.add(node.name)
        for precessor in precessors[node.name]:
            if precessor.name in dependent_nodes:
                continue
            keep_dims = 1
            reduce_axis = -1
            if is_reduction_node(precessor):
                keep_dims, reduce_axis = self._get_reduce_info(precessor)
            if group.compatible(precessor, keep_dims, reduce_axis):
                next_group = group.add_node(precessor, reduce_axis)
                dependent_nodes.update(self._process_node(precessor, precessors, next_group))
        return dependent_nodes

    def _group_nodes(self):
        producers = dict()
        precessors = defaultdict(list)
        processed = set()
        sorted_nodes = self._sorted_graph.sorted_nodes
        for node in sorted_nodes:
            for output in node.output:
                producers[output] = node
            for input in node.input:
                if input in producers:
                    precessors[node.name].append(producers[input])
        for _, value in precessors.items():
            value.sort(key=sorted_nodes.index, reverse=True)
        for idx in range(len(sorted_nodes) - 1, -1, -1):
            node = sorted_nodes[idx]
            if node.name not in processed:
                reduce_axis = -1
                if is_reduction_node(node):
                    _, reduce_axis = self._get_reduce_info(node)
                self._groups.append(NodeGroup(node, reduce_axis, self._node_arg_infos))
                processed.update(self._process_node(node, precessors, self._groups[-1]))

    def _get_node_io(self, node: NodeProto) -> Tuple[List[TensorArg], List[TensorArg]]:
        input_args = []
        for input in node.input:
            if input in self._tensor_args:
                input_args.append(self._tensor_args[input])
            else:
                input_args.append(TensorArg(input, self._node_arg_infos[input]))
                self._tensor_args[input] = input_args[-1]
        output_args = []
        for output in node.output:
            if output in self._tensor_args:
                output_args.append(self._tensor_args[output])
            else:
                output_args.append(TensorArg(output, self._node_arg_infos[output]))
                self._tensor_args[output] = output_args[-1]
        return input_args, output_args

    def _extract_kernel_io(self, nodes: List[NodeProto]) -> KernelIO:
        kernel_io = KernelIO()
        input_set = set()
        output_set = set()
        for node in nodes:
            for input in node.input:
                if input in input_set:
                    continue
                elif input in self._module_constant_names:
                    kernel_io.constants.append(input)
                elif input in self._module_input_names:
                    kernel_io.module_inputs.append(input)
                elif input not in output_set:
                    kernel_io.cross_kernel_inputs.append(input)
                input_set.add(input)
            for output in node.output:
                if output in output_set:
                    continue
                if output in self._module_output_names:
                    kernel_io.module_outputs.append(output)
                else:
                    kernel_io.internal_args.append(output)
                output_set.add(output)
        return kernel_io

    def _to_compute_node(self, node: NodeProto, offset_calc: OffsetCalculator):
        inputs, outputs = self._get_node_io(node)
        op_type = node.op_type
        if op_type == "Dropout":
            return DropoutNode(inputs, outputs, offset_calc)
        if is_reduction_node(node):
            return ReduceNode(op_type, inputs, outputs)
        return ComputeNode(op_type, inputs, outputs)

    def _analyze_kernel_io_list(self):
        cross_kernel_inputs = set()
        for kernel_io in self._kernel_io_list:
            cross_kernel_inputs.update(kernel_io.cross_kernel_inputs)
        for kernel_io in self._kernel_io_list:
            kernel_io.cross_kernel_outputs = [arg for arg in kernel_io.internal_args if arg in cross_kernel_inputs]
            kernel_io.internal_args = [
                arg for arg in kernel_io.internal_args if arg not in kernel_io.cross_kernel_outputs
            ]

    def _insert_load_and_store(self, kernel_node: KernelNode):
        input_names = [input.name for input in kernel_node.inputs]
        output_name_map = dict()
        for output in kernel_node.outputs:
            output_name_map[output.name] = 0
        for node in kernel_node.sub_nodes:
            for output in node.outputs:
                if output.name in output_name_map:
                    output_name_map[output.name] += 1
        sub_nodes = kernel_node.sub_nodes
        new_sub_nodes = []
        cur = 0
        nxt = 0
        reduce_store_nodes = []
        while True:
            while nxt < len(sub_nodes) and not isinstance(sub_nodes[nxt], RecomputeEnd):
                nxt += 1
            load_cache = set()
            load_nodes = []
            store_nodes = []
            for idx in range(cur, nxt):
                for input in sub_nodes[idx].inputs:
                    if input.name in kernel_node.constants or input.name in input_names:
                        if (input.data is not None and input.data.size == 1) or input.name in load_cache:
                            continue
                        load_nodes.append(IONode(input, kernel_node.offset_calc, True))
                        load_cache.add(input.name)
                for output in sub_nodes[idx].outputs:
                    if output.name in output_name_map:
                        output_name_map[output.name] -= 1
                        if output_name_map[output.name] == 0:
                            store_nodes.append(IONode(output, kernel_node.offset_calc, False))
            if isinstance(sub_nodes[cur], RecomputeStart):
                new_sub_nodes.append(sub_nodes[cur])
                cur += 1
            if nxt < len(sub_nodes):
                assert isinstance(sub_nodes[nxt], RecomputeEnd)
                for reduce_node in sub_nodes[nxt].reduce_nodes:
                    input = reduce_node.inputs[0]
                    if input.name in kernel_node.constants or input.name in input_names:
                        if (input.data is not None and input.data.size == 1) or input.name in load_cache:
                            continue
                        load_nodes.append(IONode(input, kernel_node.offset_calc, True))
                        load_cache.add(input.name)
            new_sub_nodes.extend(load_nodes)
            new_sub_nodes.extend(sub_nodes[cur:nxt])
            new_sub_nodes.extend(store_nodes)
            if nxt < len(sub_nodes):
                assert isinstance(sub_nodes[nxt], RecomputeEnd)
                for reduce_node in sub_nodes[nxt].reduce_nodes:
                    if reduce_node.outputs[0].name in output_name_map:
                        reduce_store_nodes.append(IONode(reduce_node.outputs[0], kernel_node.offset_calc, False))
                new_sub_nodes.append(sub_nodes[nxt])
                nxt += 1
            cur = nxt
            if cur >= len(sub_nodes):
                break
        new_sub_nodes.extend(reduce_store_nodes)
        kernel_node.sub_nodes = new_sub_nodes

    def _lower(self):
        for group in self._groups:
            is_reduction_kernel = group.reduce_axis != -1
            target_shape = group.target_shape
            # The inputs and outputs will be initialized later.
            kernel_node = (
                ReduceKernelNode([], [], target_shape, group.reduce_axis)
                if is_reduction_kernel
                else ElementwiseKernelNode([], [], target_shape)
            )
            self._kernel_nodes.append(kernel_node)
            sub_nodes = []
            nodes, layer_indices = group.flatten(self._sorted_graph.sorted_nodes)
            self._kernel_io_list.append(self._extract_kernel_io(nodes))
            if group.recompute:
                r_numel = group.target_shape[group.reduce_axis]
                start = 0
                for indices in layer_indices:
                    reduce_nodes = [self._to_compute_node(nodes[idx], kernel_node.offset_calc) for idx in indices]
                    assert all(isinstance(node, ReduceNode) for node in reduce_nodes)
                    sub_nodes.append(RecomputeStart(reduce_nodes, r_numel))
                    end = indices[0] if len(indices) > 0 else len(nodes)
                    for idx in range(start, end):
                        node = nodes[idx]
                        assert not is_reduction_node(node)
                        sub_nodes.append(self._to_compute_node(node, kernel_node.offset_calc))
                        if node.op_type == "Dropout":
                            self._kernel_nodes[-1].has_dropout = True
                    if len(indices) > 0:
                        sub_nodes.append(RecomputeEnd(reduce_nodes))
                    start = indices[len(indices) - 1] + 1 if len(indices) > 0 else len(nodes)
            else:
                for node in nodes:
                    sub_nodes.append(self._to_compute_node(node, kernel_node.offset_calc))
                    if node.op_type == "Dropout":
                        self._kernel_nodes[-1].has_dropout = True
            self._kernel_nodes[-1].sub_nodes = sub_nodes

        self._analyze_kernel_io_list()
        cross_kernel_arg_map = dict()
        for idx, kernel_io in enumerate(self._kernel_io_list):
            for output in itertools.chain(kernel_io.cross_kernel_outputs, kernel_io.module_outputs):
                cross_kernel_arg_map[output] = idx
        dependency = defaultdict(set)
        for idx, kernel_io in enumerate(self._kernel_io_list):
            for input in kernel_io.cross_kernel_inputs:
                dependency[cross_kernel_arg_map[input]].add(idx)
        visited = set()
        sorted_indices = []

        def _topological_soft_internal(idx):
            visited.add(idx)
            for next_idx in dependency[idx]:
                if next_idx not in visited:
                    _topological_soft_internal(next_idx)
            sorted_indices.insert(0, idx)

        for idx in range(len(self._kernel_io_list)):
            if idx not in visited:
                _topological_soft_internal(idx)

        self._kernel_nodes = [self._kernel_nodes[idx] for idx in sorted_indices]
        self._kernel_io_list = [self._kernel_io_list[idx] for idx in sorted_indices]
        cross_kernel_arg_map.clear()
        for idx, kernel_io in enumerate(self._kernel_io_list):
            for arg in kernel_io.cross_kernel_inputs:
                if arg not in self._module_output_names:
                    cross_kernel_arg_map[arg] = idx

        self._cross_kernel_args = [(self._tensor_args[key], value) for key, value in cross_kernel_arg_map.items()]

        for idx, kernel_node in enumerate(self._kernel_nodes):
            kernel_io = self._kernel_io_list[idx]
            kernel_node.internal_args.update(kernel_io.internal_args)
            kernel_node.inputs = [
                self._tensor_args[name]
                for name in itertools.chain(kernel_io.module_inputs, kernel_io.cross_kernel_inputs)
            ]
            kernel_node.outputs = [
                self._tensor_args[name]
                for name in itertools.chain(kernel_io.module_outputs, kernel_io.cross_kernel_outputs)
            ]
            for name in kernel_io.constants:
                kernel_node.constants[name] = self._tensor_args[name]
            self._insert_load_and_store(kernel_node)
            kernel_node.gen_variable_names()

    def module_node(self, func_name: str):
        return ModuleNode(
            func_name,
            self._module_inputs,
            self._module_outputs,
            self._module_constants,
            self._cross_kernel_args,
            self._kernel_nodes,
        )


def lower(func_name: str, sorted_graph: SortedGraph) -> ModuleNode:
    return GraphLowering(sorted_graph).module_node(func_name)
