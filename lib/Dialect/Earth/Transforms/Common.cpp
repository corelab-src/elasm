#include "hecate/Dialect/Earth/Transforms/Common.h"

using namespace mlir;

void hecate::earth::refineReturnValues(mlir::func::FuncOp func,
                                       mlir::OpBuilder builder,
                                       SmallVector<mlir::Type, 4> inputTypes,
                                       int64_t waterline, int64_t output_val) {

  int64_t max_required_level = 0;
  // Reduce the level of the resulting values to reduce the size of returns
  //
  auto rop = dyn_cast<func::ReturnOp>(func.getBlocks().front().getTerminator());
  /* func.walk([&](func::ReturnOp rop) { */
  builder.setInsertionPoint(rop);

  int64_t acc_scale_max = 0;
  int64_t rescalingFactor = hecate::earth::EarthDialect::rescalingFactor;
  for (auto v : rop.getOperands()) {
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale = st.getLevel() * rescalingFactor + st.getScale();
    acc_scale_max = std::max(acc_scale_max, acc_scale);
  }

  max_required_level =
      (acc_scale_max + output_val + rescalingFactor - 1) / rescalingFactor;

  for (size_t i = 0; i < rop.getNumOperands(); i++) {
    auto v = rop.getOperand(i);
    auto st = v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
    auto acc_scale = st.getLevel() * rescalingFactor + st.getScale();
    int64_t required_level =
        (acc_scale + output_val + rescalingFactor - 1) / rescalingFactor;
    int64_t level_diff = max_required_level - required_level;
    rop.setOperand(i, builder.create<hecate::earth::ModswitchOp>(
                          rop.getLoc(), v, level_diff));
  }

  // Remap the return types
  func.setFunctionType(
      builder.getFunctionType(inputTypes, rop.getOperandTypes()));
  /* }); */
  func->setAttr("init_level", builder.getI64IntegerAttr(max_required_level));
  SmallVector<int64_t, 4> scales_in;
  SmallVector<int64_t, 4> scales_out;
  for (auto &&arg : func.getArguments()) {
    scales_in.push_back(arg.getType()
                            .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                            .getScale());
  }
  func->setAttr("arg_scale", builder.getDenseI64ArrayAttr(scales_in));
  for (auto &&restype : func.getResultTypes()) {
    scales_out.push_back(
        restype.dyn_cast<hecate::earth::HEScaleTypeInterface>().getScale());
  }
  func->setAttr("res_scale", builder.getDenseI64ArrayAttr(scales_out));
}

void hecate::earth::inferTypeForward(hecate::earth::ForwardMgmtInterface sop) {
  Operation *oop = sop.getOperation();
  auto iop = dyn_cast<mlir::InferTypeOpInterface>(oop);
  SmallVector<Type, 4> retTypes;
  if (iop.inferReturnTypes(oop->getContext(), oop->getLoc(), oop->getOperands(),
                           oop->getAttrDictionary(), oop->getRegions(),
                           retTypes)
          .succeeded()) {
    oop->getResults().back().setType(retTypes.back());
  }
}
