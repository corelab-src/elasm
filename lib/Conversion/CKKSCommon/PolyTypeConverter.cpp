
#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace hecate;

PolyTypeConverter::PolyTypeConverter(int64_t base_level)
    : base_level(base_level) {
  addConversion([&](mlir::Type t) { return t; });
  addConversion([&](mlir::FunctionType t) { return convertFunctionType(t); });
  addConversion([&](mlir::RankedTensorType t) { return convertTensorType(t); });

  addConversion(
      [&](hecate::earth::CipherType t) { return convertCipherType(t); });
  addConversion(
      [&](hecate::earth::PlainType t) { return convertPlainType(t); });

  /* addConversion([&](hecate::earth::HEScaleTypeInterface t) { */
  /*   return convertScaleType(t); */
  /* }); */
}

mlir::Type PolyTypeConverter::convertFunctionType(mlir::FunctionType t) {
  mlir::SmallVector<mlir::Type, 4> inputTys;
  mlir::SmallVector<mlir::Type, 4> outputTys;
  for (auto &&t : t.getInputs()) {
    inputTys.push_back(convertType(t));
  }
  for (auto &&t : t.getResults()) {
    outputTys.push_back(convertType(t));
  }
  return mlir::FunctionType::get(t.getContext(), inputTys, outputTys);
}

mlir::Type PolyTypeConverter::convertTensorType(mlir::TensorType t) {
  return mlir::RankedTensorType::get(t.getShape(),
                                     convertType(t.getElementType()));
}

mlir::Type PolyTypeConverter::convertCipherType(hecate::earth::CipherType t) {
  return hecate::ckks::PolyType::get(t.getContext(), 2,
                                     base_level - t.getLevel());
}
mlir::Type PolyTypeConverter::convertPlainType(hecate::earth::PlainType t) {
  return hecate::ckks::PolyType::get(t.getContext(), 1,
                                     base_level - t.getLevel());
}
/* mlir::Type */
/* PolyTypeConverter::convertScaleType(hecate::earth::HEScaleTypeInterface t) {
 */
/*   return hecate::ckks::PolyType::get(t.getContext(), t.isCipher() ? 2 : 1, */
/*                                      base_level - t.getLevel()); */
/* } */
