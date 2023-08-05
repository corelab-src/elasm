

#include <fstream>

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "llvm/Support/Debug.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_SNRRESCALING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "hecate_snr"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct SNRRescalingPass
    : public hecate::earth::impl::SNRRescalingBase<SNRRescalingPass> {

  SNRRescalingPass() {}

  SNRRescalingPass(hecate::earth::SNRRescalingOptions ops) {
    this->waterline = ops.waterline;
    this->output_val = ops.output_val;
  }

  int64_t calcWaterline(hecate::ScaleManagementUnit &smu, Value v) {
    while (smu.getID(v) < 0) {
      // Backward movement for -1 value because it means scale management
      // operations or apply_schedule op.
      v = v.getUses().begin().getUser()->getResult(0);
    }
    return smu.inNoisyGroup(v) ? waterline + 4 : waterline;
  }
  int64_t calcWaterline(hecate::ScaleManagementUnit &smu, Operation *op) {
    return calcWaterline(smu, op->getResult(0));
  }

  void runOnOperation() override {

    auto func = getOperation();

    hecate::ScaleManagementUnit smu =
        getAnalysis<hecate::ScaleManagementUnit>();

    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);
    SmallVector<mlir::Type, 4> inputTypes;
    // Set function argument types
    for (auto argval : func.getArguments()) {
      argval.setType(
          argval.getType().dyn_cast<RankedTensorType>().replaceSubElements(
              [&](hecate::earth::HEScaleTypeInterface t) {
                return t.switchScale(calcWaterline(smu, argval));
              }));

      inputTypes.push_back(argval.getType());
    }
    func.setFunctionType(builder.getFunctionType(
        inputTypes, func.getFunctionType().getResults()));

    func.walk([&](hecate::earth::ForwardMgmtInterface sop) {
      builder.setInsertionPoint(sop.getOperation());
      sop.processOperandsSNR(calcWaterline(smu, sop.getOperation()));
      inferTypeForward(sop);
      builder.setInsertionPointAfter(sop.getOperation());
      sop.processResultsSNR(calcWaterline(smu, sop.getOperation()));
    });
    hecate::earth::refineReturnValues(func, builder, inputTypes, waterline,
                                      output_val);

    std::error_code EC;

    LLVM_DEBUG(llvm::dbgs() << __FILE__ << ":" << __LINE__ << "\n");
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
