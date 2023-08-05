

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_REMOVELEVEL
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct RemoveLevelPass
    : public hecate::ckks::impl::RemoveLevelBase<RemoveLevelPass> {
  RemoveLevelPass() {}

  void runOnOperation() override {
    markAllAnalysesPreserved();
    auto &&func = getOperation();
    mlir::OpBuilder builder(func);

    SmallVector<int64_t, 4> level_in;
    SmallVector<int64_t, 4> level_out;
    for (auto &&arg : func.getArguments()) {
      level_in.push_back(
          arg.getType().dyn_cast<hecate::ckks::PolyTypeInterface>().getLevel());
    }
    func->setAttr("arg_level", builder.getDenseI64ArrayAttr(level_in));
    for (auto &&restype : func.getResultTypes()) {
      level_out.push_back(
          restype.dyn_cast<hecate::ckks::PolyTypeInterface>().getLevel());
    }
    func->setAttr("res_level", builder.getDenseI64ArrayAttr(level_out));

    for (auto value : func.getArguments()) {
      auto &&tt = value.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
      value.setType(tt.switchLevel(0));
    }
    func.walk([&](Operation *op) {
      for (auto value : op->getResults()) {
        auto &&tt = value.getType().dyn_cast<hecate::ckks::PolyTypeInterface>();
        value.setType(tt.switchLevel(0));
      }
    });

    auto funcType = func.getFunctionType();
    llvm::SmallVector<Type, 4> inputTys;
    llvm::SmallVector<Type, 4> outputTys;
    for (auto &&ty : funcType.getInputs()) {
      auto &&tt = ty.dyn_cast<hecate::ckks::PolyTypeInterface>();
      inputTys.push_back(tt.switchLevel(0));
    }
    for (auto &&ty : funcType.getResults()) {
      auto &&tt = ty.dyn_cast<hecate::ckks::PolyTypeInterface>();
      outputTys.push_back(tt.switchLevel(0));
    }
    func.setFunctionType(builder.getFunctionType(inputTys, outputTys));
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::ckks::CKKSDialect>();
  }
};
} // namespace
