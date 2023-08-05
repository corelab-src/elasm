
#ifndef HECATE_CONVERSION_CKKSTOCKKS_UPSCALETOMULCP_H
#define HECATE_CONVERSION_CKKSTOCKKS_UPSCALETOMULCP_H

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

#define GEN_PASS_DECL_UPSACLETOMULCPCONVERSION
#include "mlir/Conversion/Passes.h.inc"

namespace ckks {

std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createUpscaleToMulcpConversionPass();

void populateUpscaleToMulcpConversionPatterns(
    mlir::MLIRContext *ctxt, mlir::RewritePatternSet &patterns);

} // namespace ckks

} // namespace hecate

#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H
