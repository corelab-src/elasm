
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_UPSCALEBUBBLING
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate
  //
#define DEBUG_TYPE "hecate_ub"

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct UpscaleBubblingPass
    : public hecate::earth::impl::UpscaleBubblingBase<UpscaleBubblingPass> {
  UpscaleBubblingPass() {}

  void runOnOperation() override {

    auto func = getOperation();
    markAnalysesPreserved<hecate::ScaleManagementUnit>();

    mlir::OpBuilder builder(func);
    mlir::IRRewriter rewriter(builder);

    SmallVector<mlir::Type, 4> inputTypes;

    auto &&bb = func.getBody().getBlocks().front();
    for (auto iter = bb.rbegin(); iter != bb.rend(); ++iter) {
      if (auto op = dyn_cast<hecate::earth::HEScaleOpInterface>(*iter)) {
        // Gather the users and finds the minimum "upFactor"
        if (op.isConsume() && !op.isSingle()) {
          continue;
        }

        uint64_t minUpFactor = -1;
        for (auto &&oper : op->getResult(0).getUses()) {
          if (auto oop = dyn_cast<hecate::earth::UpscaleOp>(oper.getOwner())) {
            minUpFactor = std::min(minUpFactor, oop.getUpFactor());
          } else {
            minUpFactor = 0;
          }
        }

        // Check that every user needs the "upFactored"ed scale
        if (!minUpFactor) {
          continue; // Go to next operation
        }

        // Move the modswitch
        if (auto oop = dyn_cast<hecate::earth::UpscaleOp>(op.getOperation())) {
          // Upscale movement can be absorbed into upscale
          oop.setUpFactor(oop.getUpFactor() + minUpFactor);
          oop.getResult().setType(
              oop.getScaleType().switchScale(oop.getScale() + minUpFactor));
        } else if (!op.isConsume()) {
          // Upscale is moved to the opreands
          for (auto &&i = 0; i < op->getNumOperands(); i++) {
            auto oper = op->getOperand(i);
            builder.setInsertionPoint(op);
            auto newOper = builder.create<hecate::earth::UpscaleOp>(
                op->getLoc(), oper, minUpFactor);
            op->setOperand(i, newOper);
          }
          op->getResult(0).setType(
              op.getScaleType().switchScale(op.getScale() + minUpFactor));
        } else if (op.isConsume() && op.isSingle()) {
          // Upscale can be moved to ciphertext operand
          for (auto &&i = 0; i < op->getNumOperands(); i++) {
            if (op.isOperandCipher(i)) {
              auto oper = op->getOperand(i);
              builder.setInsertionPoint(op);
              auto newOper = builder.create<hecate::earth::UpscaleOp>(
                  op->getLoc(), oper, minUpFactor);
              op->setOperand(i, newOper);
            }
          }
          op->getResult(0).setType(
              op.getScaleType().switchScale(op.getScale() + minUpFactor));
        }

        // Change the user modswitch downFactors
        for (auto &&oper : op->getResult(0).getUsers()) {
          if (auto oop = dyn_cast<hecate::earth::UpscaleOp>(oper)) {
            oop.setUpFactor(oop.getUpFactor() - minUpFactor);
          }
        }
      }
    }
    LLVM_DEBUG(llvm::dbgs() << __FILE__ << ":" << __LINE__ << "\n");
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
