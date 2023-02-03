// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CUDA

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

struct TritonFusionConfig {
  struct OpInfo {
    std::string domain;
    std::vector<ONNX_NAMESPACE::OperatorSetVersion> versions;
    bool is_no_op;
    std::unordered_map<std::string, std::string> conditions;
  };

  TritonFusionConfig(std::string_view config_json = "{}");

  bool IsSupported(const Graph& graph, const Node& node) const;
  bool IsNoOp(const Node& node) const;

  std::unordered_map<std::string, OpInfo> ops;
  std::string initializer = "none";
};

class TritonFusion : public GraphTransformer {
 public:
  TritonFusion(std::string_view config_json = "{}",
               const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("TritonFusion", compatible_execution_providers), config_(config_json) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  bool IsSupportedNode(const Graph& graph, const Node& node) const;

  TritonFusionConfig config_;
};

}  // namespace onnxruntime

#endif  // USE_CUDA
