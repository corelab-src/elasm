
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include <fstream>

namespace hecate {
namespace earth {
#define GEN_PASS_DEF_ELIDECONSTANT
#include "hecate/Dialect/Earth/Transforms/Passes.h.inc"
} // namespace earth
} // namespace hecate

using namespace mlir;

namespace {
/// Pass to bufferize Arith ops.
struct ElideConstantPass
    : public hecate::earth::impl::ElideConstantBase<ElideConstantPass> {
  ElideConstantPass() {}
  ElideConstantPass(hecate::earth::ElideConstantOptions ops) {
    this->name = ops.name;
  }

  void runOnOperation() override {
    auto func = getOperation();
    SmallVector<SmallVector<double, 4>, 4> save_data;

    mlir::OpBuilder builder(func.getOperation());

    func.walk([&](hecate::earth::ConstantOp cop) {
      SmallVector<double, 4> datas(
          cop.getValue().dyn_cast<DenseElementsAttr>().getValues<double>());
      save_data.push_back(datas);
      cop.setValueAttr(builder.getI64IntegerAttr(save_data.size() - 1));
    });

    name = name + (func.getName() + ".cst").str();
    llvm::errs() << name << "\n";
    std::ofstream of(name, std::ios::binary);
    int64_t a;
    a = save_data.size();
    of.write((char *)&a, sizeof(int64_t));
    for (auto k : save_data) {
      a = k.size();
      of.write((char *)&a, sizeof(int64_t));
      for (auto d : k) {
        of.write((char *)&d, sizeof(double));
      }
    }
    of.close();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hecate::earth::EarthDialect>();
  }
};
} // namespace
