
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "hecate/Support/Support.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_SMUEMBEDDING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct SMUEmbeddingPass
    : public hecate::earth::impl::SMUEmbeddingBase<SMUEmbeddingPass> {
  SMUEmbeddingPass() {}

  void runOnOperation() override {
    auto func = getOperation();
    auto builder = OpBuilder(func);

    auto smu = getAnalysis<hecate::ScaleManagementUnit>();
    markAllAnalysesPreserved();

    smu.attach();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
