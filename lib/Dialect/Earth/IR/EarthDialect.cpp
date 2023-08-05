

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include <fstream>

using namespace mlir;

#define GET_TYPEDEF_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"

#include "hecate/Dialect/Earth/IR/EarthOpsDialect.cpp.inc"

#include "hecate/Dialect/Earth/IR/EarthCanonicalizerPattern.inc"

struct ScaleTypeTensorModel
    : public hecate::earth::HEScaleTypeInterface::ExternalModel<
          ScaleTypeTensorModel, mlir::RankedTensorType> {
  bool isCipher(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.isCipher();
    } else {
      return false;
    }
  }
  hecate::earth::HEScaleTypeInterface toCipher(Type t) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .toCipher());
  }
  hecate::earth::HEScaleTypeInterface toPlain(Type t) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .toPlain());
  }
  hecate::earth::HEScaleTypeInterface switchScale(Type t,
                                                  unsigned scale) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .switchScale(scale));
  }
  hecate::earth::HEScaleTypeInterface switchLevel(Type t,
                                                  unsigned level) const {
    auto tt = t.dyn_cast<RankedTensorType>();
    return RankedTensorType::get(
        tt.getShape(), tt.getElementType()
                           .dyn_cast<hecate::earth::HEScaleTypeInterface>()
                           .switchLevel(level));
  }
  unsigned getScale(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.getScale();
    } else {
      return 0;
    }
  }
  unsigned getLevel(Type t) const {
    if (auto scaleType = t.dyn_cast<mlir::RankedTensorType>()
                             .getElementType()
                             .dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      return scaleType.getLevel();
    } else {
      return 0;
    }
  }
};

void hecate::earth::EarthDialect::initialize() {
  // Registers all the Types into the EVADialect class
  addTypes<
#define GET_TYPEDEF_LIST
#include "hecate/Dialect/Earth/IR/EarthOpsTypes.cpp.inc"
      >();

  // Registers all the Operations into the EVADialect class
  addOperations<
#define GET_OP_LIST
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"
      >();
  mlir::RankedTensorType::attachInterface<ScaleTypeTensorModel>(*getContext());
}

std::map<std::string, llvm::SmallVector<int64_t>> latencyTable;

std::map<std::string, llvm::SmallVector<double>> noiseTable;

int64_t hecate::earth::EarthDialect::polynomialDegree;
int64_t hecate::earth::EarthDialect::rescalingFactor;
int64_t hecate::earth::EarthDialect::bootstrapLevelLowerBound;
int64_t hecate::earth::EarthDialect::bootstrapLevelUpperBound;
int64_t hecate::earth::EarthDialect::levelUpperBound;
int64_t hecate::earth::EarthDialect::levelLowerBound;

template <typename... Args>
void addOperationNamesTo(SmallVectorImpl<StringRef> &names) {
  (void)std::initializer_list<int>{
      0, (names.push_back(Args::getOperationName()), 0)...};
}

std::map<std::string, llvm::SmallVector<int64_t>>
hecate::earth::EarthDialect::getLatencyTable() {
  return latencyTable;
}
std::map<std::string, llvm::SmallVector<double>>
hecate::earth::EarthDialect::getNoiseTable() {
  return noiseTable;
}

void hecate::earth::EarthDialect::setCKKSParameters(llvm::StringRef filename) {
  // it should be read from file but currently fixed
  hecate::earth::EarthDialect::levelLowerBound = 1;  // 0 should be forbidden
  hecate::earth::EarthDialect::levelUpperBound = 13; // L =14
                                                     // inclusion
  hecate::earth::EarthDialect::polynomialDegree = 1LL << 15;

  SmallVector<StringRef, 4> names;
  addOperationNamesTo<
#define GET_OP_LIST
#include "hecate/Dialect/Earth/IR/EarthOps.cpp.inc"
      >(names);

  std::ifstream iff(filename.str());
  nlohmann::json config = nlohmann::json::parse(iff);
  hecate::earth::EarthDialect::rescalingFactor = config["rescalingFactor"];
  hecate::earth::EarthDialect::levelLowerBound = config["levelLowerBound"];
  hecate::earth::EarthDialect::levelUpperBound = config["levelUpperBound"];
  hecate::earth::EarthDialect::bootstrapLevelLowerBound =
      config["bootstrapLevelLowerBound"];
  hecate::earth::EarthDialect::bootstrapLevelUpperBound =
      config["bootstrapLevelUpperBound"];
  hecate::earth::EarthDialect::polynomialDegree = config["polynomialDegree"];

  for (auto &&name : names) {
    for (auto &&suffix : {"_single", "_double"}) {
      auto &latTab = latencyTable[name.str() + suffix];
      auto &noTab = noiseTable[name.str() + suffix];
      latTab.push_back(0);
      noTab.push_back(0);
      if (config["latencyTable"].contains(name.str() + suffix) &&
          config["latencyTable"][name.str() + suffix].size()) {
        for (auto &&data : config["latencyTable"][name.str() + suffix]) {
          latTab.push_back(data.get<int64_t>());
        }
      }
      latTab.resize(levelUpperBound + 1, latTab.back());

      if (config["noiseTable"].contains(name.str() + suffix) &&
          config["noiseTable"][name.str() + suffix].size()) {
        for (auto &&data : config["noiseTable"][name.str() + suffix]) {
          noTab.push_back(data.get<double>());
        }
      }
      noTab.resize(levelUpperBound + 1, noTab.back());

      latencyTable[name.str() + suffix] = latTab;
      noiseTable[name.str() + suffix] = noTab;
    }
  }
}

::mlir::LogicalResult hecate::earth::ConstantOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ConstantOpAdaptor(operands, attributes, regions);

  inferredReturnTypes.push_back(mlir::RankedTensorType::get(
      llvm::SmallVector<int64_t, 1>{1}, PlainType::get(context, 0, 0)));
  return ::mlir::success();
}

void hecate::earth::RescaleOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<RescaleUpscalePattern>(context);
}
void hecate::earth::RotateOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  /* patterns.add<RotateOffsetModuloPattern>(context); */
}

::mlir::LogicalResult hecate::earth::RescaleOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = RescaleOpAdaptor(operands, attributes, regions);
  auto lScale = earth::getScaleType(op.getValue());
  inferredReturnTypes.push_back(
      lScale.switchLevel(lScale.getLevel() + 1)
          .switchScale(lScale.getScale() -
                       hecate::earth::EarthDialect::rescalingFactor));

  return ::mlir::success();
}

void hecate::earth::ModswitchOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ModswitchModswitchPattern, ModswitchConstantPattern,
               ZeroModswitchPattern>(context);
}

::mlir::LogicalResult hecate::earth::ModswitchOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = ModswitchOpAdaptor(operands, attributes, regions);
  auto lScale = earth::getScaleType(op.getValue());
  inferredReturnTypes.push_back(
      lScale.switchLevel(lScale.getLevel() + op.getDownFactor()));
  return ::mlir::success();
}

void hecate::earth::UpscaleOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns
      .add<UpscaleUpscalePattern, UpscaleConstantPattern, ZeroUpscalePattern>(
          context);
}

::mlir::LogicalResult hecate::earth::UpscaleOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = UpscaleOpAdaptor(operands, attributes, regions);
  auto lScale = earth::getScaleType(op.getValue());
  inferredReturnTypes.push_back(
      lScale.switchScale(lScale.getScale() + op.getUpFactor()));

  return ::mlir::success();
}

void hecate::earth::AddOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<AddZeroPattern>(context);
}

::mlir::LogicalResult hecate::earth::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = AddOpAdaptor(operands, attributes, regions);
  auto lScale = earth::getScaleType(op.getLhs());
  auto lTensor = earth::getTensorType(op.getLhs());
  auto rScale = earth::getScaleType(op.getRhs());
  auto rTensor = earth::getTensorType(op.getRhs());
  if (lScale.getLevel() == rScale.getLevel() &&
      lScale.getScale() == rScale.getScale() &&
      lTensor.getShape()[0] == rTensor.getShape()[0]) {
    inferredReturnTypes.push_back(lScale.toCipher());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

void hecate::earth::MulOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<MulZeroPattern, MulOnePattern, NegMulPattern>(context);
}

::mlir::LogicalResult hecate::earth::MulOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {
  auto op = MulOpAdaptor(operands, attributes, regions);
  auto lScale = earth::getScaleType(op.getLhs());
  auto lTensor = earth::getTensorType(op.getLhs());
  auto rScale = earth::getScaleType(op.getRhs());
  auto rTensor = earth::getTensorType(op.getRhs());
  if (lScale.getLevel() == rScale.getLevel() &&
      lTensor.getShape()[0] == rTensor.getShape()[0]) {
    inferredReturnTypes.push_back(
        lScale.switchScale(lScale.getScale() + rScale.getScale()).toCipher());
    return ::mlir::success();
  } else {
    return ::mlir::failure();
  }
}

mlir::RankedTensorType hecate::earth::getTensorType(mlir::Value v) {
  return v.getType().dyn_cast<mlir::RankedTensorType>();
}
hecate::earth::HEScaleTypeInterface hecate::earth::getScaleType(mlir::Value v) {
  return v.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>();
}
