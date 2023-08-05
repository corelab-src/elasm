#ifndef HECATE_CONVERSION_CKKSCOMMON_TYPECONVERTER_H
#define HECATE_CONVERSION_CKKSCOMMON_TYPECONVERTER_H

#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace hecate {
struct PolyTypeConverter : public mlir::TypeConverter {
  using TypeConverter::TypeConverter;

  PolyTypeConverter(int64_t base_level);
  mlir::Type convertFunctionType(mlir::FunctionType t);
  mlir::Type convertTensorType(mlir::TensorType t);
  mlir::Type convertCipherType(hecate::earth::CipherType t);
  mlir::Type convertPlainType(hecate::earth::PlainType t);

private:
  int64_t base_level;
};
} // namespace hecate

#endif
