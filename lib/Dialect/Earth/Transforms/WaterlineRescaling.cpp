
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_WATERLINERESCALING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct WaterlineRescalingPass
    : public hecate::earth::impl::WaterlineRescalingBase<
          WaterlineRescalingPass> {
  WaterlineRescalingPass() {}

  WaterlineRescalingPass(hecate::earth::WaterlineRescalingOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  void runOnOperation() override {

    auto func = getOperation();

    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    SmallVector<mlir::Type, 4> inputTypes;
    // Set function argument types
    for (auto argval : func.getArguments()) {
      argval.setType(
          argval.getType().dyn_cast<RankedTensorType>().replaceSubElements(
              [&](hecate::earth::HEScaleTypeInterface t) {
                return t.switchScale(waterline);
              }));
      inputTypes.push_back(argval.getType());
    }

    // Apply waterline rescaling for the operations
    func.walk([&](hecate::earth::ForwardMgmtInterface sop) {
      builder.setInsertionPointAfter(sop.getOperation());
      sop.processOperandsEVA(waterline);
      inferTypeForward(sop);
      sop.processResultsEVA(waterline);
    });
    hecate::earth::refineReturnValues(func, builder, inputTypes, waterline,
                                      output_val);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
