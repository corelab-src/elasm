
#ifndef HECATE_CONVERSION_EARTHTOCKKS_EARTHTOCKKS_H
#define HECATE_CONVERSION_EARTHTOCKKS_EARTHTOCKKS_H

#include <memory>

#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"

namespace mlir {
namespace func {
class FuncOp;
}
class RewritePatternSet;
template <typename T> class OperationPass;
} // namespace mlir

namespace hecate {

#define GEN_PASS_DECL_EARTHTOCKKSCONVERSION
#include "mlir/Conversion/Passes.h.inc"

namespace earth {
std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createEarthToCKKSConversionPass();

void populateEarthToCKKSConversionPatterns(mlir::MLIRContext *ctxt,
                                           mlir::TypeConverter &converter,
                                           mlir::RewritePatternSet &patterns,
                                           int64_t init_level);

} // namespace earth

} // namespace hecate

#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H
