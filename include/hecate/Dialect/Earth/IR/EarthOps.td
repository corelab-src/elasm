
#ifndef HECATE_ARITH_OPS
#define HECATE_ARITH_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "hecate/Dialect/Earth/IR/HEParameterInterface.td"
include "hecate/Dialect/Earth/IR/ForwardManagementInterface.td"



def Earth_Dialect : Dialect {
  let name = "earth";
  let cppNamespace = "::hecate::earth";
  let extraClassDeclaration = [{

    static void setCKKSParameters(llvm::StringRef filename);
    static std::map<std::string, llvm::SmallVector<int64_t>> getLatencyTable();
    static std::map<std::string, llvm::SmallVector<double>> getNoiseTable();
    static int64_t polynomialDegree;
    static int64_t rescalingFactor;
    static int64_t bootstrapLevelLowerBound;
    static int64_t bootstrapLevelUpperBound;
    static int64_t levelUpperBound;
    static int64_t levelLowerBound;
  }];
  let useFoldAPI= kEmitFoldAdaptorFolder;
  let useDefaultTypePrinterParser = 1;
}

def CipherType :TypeDef<Earth_Dialect, "Cipher", [DeclareTypeInterfaceMethods<HEScaleTypeInterface>, DeclareTypeInterfaceMethods<MemRefElementTypeInterface>]> {
  let summary = "HE Cipher Type with encrypted, scale, and, level";
  let description = [{
    A type for CKKS parameter optimization.
      Type that implement this interface can be analyzed and managed by scale management scheme.
  }];
  let mnemonic = "ci";
  let parameters = (ins "unsigned":$scale, "unsigned":$level);
  let assemblyFormat = "`<` $scale `*` $level `>`";
  let extraClassDefinition = [{
    bool $cppClass::isCipher() const { return true;}
    ::hecate::earth::HEScaleTypeInterface $cppClass::toCipher() const {return *this;}
    ::hecate::earth::HEScaleTypeInterface $cppClass::toPlain() const {return ::hecate::earth::PlainType::get(this->getContext(), getScale(), getLevel());}
    ::hecate::earth::HEScaleTypeInterface $cppClass::switchScale(unsigned scale) const {return get(this->getContext(), scale, getLevel()); }
    ::hecate::earth::HEScaleTypeInterface $cppClass::switchLevel(unsigned level) const {return get(this->getContext(), getScale(), level);}
  }];
  let genVerifyDecl = 0;
}

def PlainType :TypeDef<Earth_Dialect, "Plain", [DeclareTypeInterfaceMethods<HEScaleTypeInterface>, DeclareTypeInterfaceMethods<MemRefElementTypeInterface>]> {
  let summary = "HE Plain Type with encrypted, scale, and, level";
  let description = [{
    A type for CKKS parameter optimization.
      Type that implement this interface can be analyzed and managed by scale management scheme.
  }];
  let mnemonic = "pl";
  let parameters = (ins "unsigned":$scale, "unsigned":$level);
  let assemblyFormat = "`<` $scale `*` $level `>`";
  let extraClassDefinition = [{
    bool $cppClass::isCipher() const { return false;}
    ::hecate::earth::HEScaleTypeInterface $cppClass::toPlain() const {return *this;}
    ::hecate::earth::HEScaleTypeInterface $cppClass::toCipher() const  {return ::hecate::earth::CipherType::get(this->getContext(), getScale(), getLevel());}
    ::hecate::earth::HEScaleTypeInterface $cppClass::switchScale(unsigned scale) const  {return get(this->getContext(), scale, getLevel()); }
    ::hecate::earth::HEScaleTypeInterface $cppClass::switchLevel(unsigned level) const {return get(this->getContext(), getScale(), level);}
  }];
  let genVerifyDecl = 0;
}


def HETensor : TensorOf<[CipherType,PlainType]>;
def CipherTensor : TensorOf<[CipherType]>;
def PlainTensor : TensorOf<[PlainType]>;

class Earth_Op<string mnemonic, list<Trait> traits = []> :
Op<Earth_Dialect, mnemonic, !listconcat(traits, [Pure, DeclareOpInterfaceMethods<HEScaleOpInterface>, DeclareOpInterfaceMethods<HEAutoDiffInterface>, DeclareOpInterfaceMethods<HEProfInterface>, DeclareOpInterfaceMethods<InferTypeOpInterface>, DeclareOpInterfaceMethods<ForwardMgmtInterface>])> {
  let results = (outs CipherTensor);
  string arithClassDefinition = [{
    // - ScaleOpInterface Non-default methods
    // - isSingle, isConsume is remaining methods

    ::hecate::earth::HEScaleTypeInterface $cppClass::getScaleType(){return getType().dyn_cast<::hecate::earth::HEScaleTypeInterface>();}
    ::hecate::earth::HEScaleTypeInterface $cppClass::getOperandScaleType(unsigned idx){return getOperation()->getOperand(idx).getType().dyn_cast<::hecate::earth::HEScaleTypeInterface>();}

    // - HEProfInterface Non-default methods
    unsigned $cppClass::getLatencyOf(unsigned level) {
      return ::hecate::earth::EarthDialect::getLatencyTable()[getOperationName().str()+ (isSingle()? "_single" : "_double")][level];
    }
    unsigned $cppClass::getNoiseScale() {
      return getScale();
    }

    double $cppClass::getNoiseOf(unsigned level) {
      return ::hecate::earth::EarthDialect::getNoiseTable()[getOperationName().str()+ (isSingle()? "_single" : "_double")][level];
    }
    unsigned $cppClass::getCipherLevel() {
      return getOperation()->getParentOfType<::mlir::func::FuncOp>()->getAttrOfType<mlir::IntegerAttr>("init_level").getInt() - HEScaleOpInterfaceTrait::getRescaleLevel();
    }
  }];
  let extraClassDefinition = arithClassDefinition;
}

/* ######### Zero-operand Operations ########## */

def ConstantOp : Earth_Op<"constant",[]> {

  let summary = "constant";

  let arguments = (ins AnyAttr:$value, F64Attr:$rms_var);
  // AnyAttr because of constant hoisting
  let results = (outs PlainTensor);

  code extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) ;
    bool isConst();
  }];

  code extraClassDefinition = arithClassDefinition # [{
    bool $cppClass::isConst(){
      return true;
    }
    bool $cppClass::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs){
      return !rhs.back().dyn_cast<HEScaleTypeInterface>().isCipher();
    }
    bool $cppClass::isConsume() {
      return false;
    }
    bool $cppClass::isSingle() {
      return true;
    }
    ::llvm::SmallVector<double, 2> $cppClass::differentiate(::llvm::ArrayRef<double> gradient,
        ::llvm::ArrayRef<double> estimation){
      return {gradient[0]};
    }
    ::llvm::SmallVector<double, 1> $cppClass::estimateValue(::llvm::ArrayRef<double> estimation){
      return {getRmsVar().convertToDouble()};
    }

  }];
  let builders = [OpBuilder<(ins "llvm::ArrayRef<double>": $arr),
      [{
        double rms = 0.0;
        for (auto && dval : arr){
          rms += dval * dval;
        }
        rms /= arr.size();
        rms = std::sqrt(rms);
        auto denseValues = mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get(llvm::SmallVector<int64_t, 1>{static_cast<long>(arr.size())},
              $_builder.getF64Type()),
            arr);
        build($_builder, $_state, denseValues, $_builder.getF64FloatAttr(rms));
      }]>];
}

/* ######### Unary Operations ########## */

class Unary_Earth_Op<string mnemonic, list<Trait> traits = []> :
Earth_Op<mnemonic, !listconcat(traits, [])> {
  string unaryClassDefinition = arithClassDefinition # [{
    bool $cppClass::isConsume() {
      return false;
    }
    bool $cppClass::isSingle() {
      return true;
    }
    ::llvm::SmallVector<double, 2> $cppClass::differentiate(::llvm::ArrayRef<double> gradient,
        ::llvm::ArrayRef<double> estimation){
      return {gradient[0]};
    }
    ::llvm::SmallVector<double, 1> $cppClass::estimateValue(::llvm::ArrayRef<double> estimation){
      return {estimation[0]};
    }
  }];
  code extraClassDefinition = unaryClassDefinition;
}

def RotateOp : Unary_Earth_Op<"rotate", [SameOperandsAndResultType]> {
  let arguments = (ins HETensor:$value, DenseI64ArrayAttr:$offset);
  code extraClassDeclaration = [{
    bool isNoisy();
    void processOperandsSNR(int64_t param);
    unsigned getNum();

  }];
  code extraClassDefinition = unaryClassDefinition # [{
    bool $cppClass::isNoisy() {
      return true;
    }
    unsigned $cppClass::getNum(){
      SmallVector<SmallVector<int, 4>, 4> nafs(getOffset().size());
      llvm::transform(getOffset(), nafs.begin(), [&](int64_t i) {
          return hecate::naf(i % (hecate::earth::EarthDialect::polynomialDegree / 2));
          });

      unsigned sum = 0 ;
      for (auto && i : nafs){
        for (auto && j : i){
          sum+= (j% (hecate::earth::EarthDialect::polynomialDegree / 2)) ? 1 : 0;
        }
      }

      return sum;
    }
    void $cppClass::processOperandsSNR(int64_t param){
      if (getOperandScale(0) < param ){
        OpBuilder builder(getOperation());
        getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getValue(), param - getOperandScale(0)));
      }
    }
  }];
  let hasCanonicalizer = 1;
}
def NegateOp : Unary_Earth_Op<"negate", [SameOperandsAndResultType]> {
  let arguments = (ins HETensor:$value);
  code extraClassDeclaration = [{
    void processOperandsSNR(int64_t param);
  }];
  code extraClassDefinition = unaryClassDefinition # [{
    void $cppClass::processOperandsSNR(int64_t param){
      if (getOperandScale(0) < param ){
        OpBuilder builder(getOperation());
        getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getValue(), param - getOperandScale(0)));
      }
    }
  }];
}

/* ######## Scale Management Operations ####### */

def RescaleOp : Unary_Earth_Op<"rescale"> {
  let arguments = (ins HETensor:$value);
  let hasCanonicalizer = 1;
}
def ModswitchOp : Unary_Earth_Op<"modswitch"> {
  let arguments = (ins HETensor:$value, I64Attr:$downFactor);
  let results = (outs HETensor);
  let hasCanonicalizer = 1;
}
def UpscaleOp : Unary_Earth_Op<"upscale"> {
  let arguments = (ins HETensor:$value, I64Attr:$upFactor);
  let results = (outs HETensor);
  let hasCanonicalizer = 1;
}

def ApplyScheduleOp : Unary_Earth_Op<"apply_schedule", [SameOperandsAndResultType]> {
  let arguments = (ins HETensor:$value, I64Attr:$scaleFactor, I64Attr:$levelFactor);
  let results = (outs HETensor);
  code extraClassDeclaration = [{
    void processOperandsEVA(int64_t param);
    void processOperandsSNR(int64_t param);
    void processResultsEVA(int64_t param);
  }];
  code extraClassDefinition = unaryClassDefinition # [{
    void $cppClass::processOperandsEVA(int64_t param){
      // This implementation highly depends on Upscale Bubbling and Canonicalization
      OpBuilder builder(getOperation());
      int scaleFactor = getScaleFactor();
      int levelFactor = getLevelFactor();
      if ( levelFactor > 0){
        getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getValue(), param+ hecate::earth::EarthDialect::rescalingFactor - getOperandScale(0) ));
        getOperation()->setOperand(0, builder.create<hecate::earth::RescaleOp>( getLoc(), getValue()));
        --levelFactor;
      }
      if (levelFactor > 0){
        getOperation()->setOperand(0, builder.create<hecate::earth::ModswitchOp>( getLoc(), getValue(), levelFactor));
      }
      if (scaleFactor > 0 ){
        getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getValue(), scaleFactor));
      }
    }
    void $cppClass::processOperandsSNR(int64_t param){
      if (getOperandScale(0) < param ){
        OpBuilder builder(getOperation());
        getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getValue(), param - getOperandScale(0)));
      }
      processOperandsPARS(param);
    }
    void $cppClass::processResultsEVA(int64_t param) {
      OpBuilder builder(getOperation());
      builder.setInsertionPointAfter(getOperation());
      IRRewriter rewriter(builder);
      rewriter.replaceAllUsesWith( getResult(), getValue());
    }
  }];
}



/* ######### Binary Operations ########## */

def AddOp: Earth_Op<"add", [Commutative]> {
  let arguments = (ins HETensor:$lhs, HETensor:$rhs);
  code extraClassDeclaration = [{
    void processOperandsEVA(int64_t param);
    void processOperandsPARS(int64_t param);
    void processOperandsSNR(int64_t param);
  }];
  code extraClassDefinition = arithClassDefinition # [{
    void $cppClass::processOperandsEVA(int64_t param){
      OpBuilder builder(getOperation());
      if (!(isOperandCipher(0) && isOperandCipher(1))){
        int lowerIdx = isOperandCipher(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperand(lowerIdx).setType(getOperandScaleType(higherIdx).toPlain());
        return;
      }
      if (getOperandScale(0) != getOperandScale(1)){
        int lowerIdx = getOperandScale(0) < getOperandScale(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperation()->setOperand(lowerIdx, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(lowerIdx), getOperandScale(higherIdx) - getOperandScale(lowerIdx)));
      }
      if (getOperandLevel(0) != getOperandLevel(1)){
        int lowerIdx = getOperandLevel(0) < getOperandLevel(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperation()->setOperand(lowerIdx, builder.create<hecate::earth::ModswitchOp>( getLoc(), getOperand(lowerIdx), getOperandLevel(higherIdx) - getOperandLevel(lowerIdx)));
      }
    }

    void $cppClass::processOperandsPARS(int64_t param){

      OpBuilder builder(getOperation());
      if (!(isOperandCipher(0) && isOperandCipher(1))){
        int lowerIdx = isOperandCipher(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperand(lowerIdx).setType(getOperandScaleType(higherIdx).toPlain());
        return;
      }
      if (getOperandLevel(0) != getOperandLevel(1)){
        int lowerIdx = getOperandScale(0) < getOperandScale(1) ? 0 : 1;
        getOperation()->setOperand(lowerIdx, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(lowerIdx), param+ hecate::earth::EarthDialect::rescalingFactor - getOperandScale(lowerIdx) ));
        getOperation()->setOperand(lowerIdx, builder.create<hecate::earth::RescaleOp>( getLoc(), getOperand(lowerIdx)));
      }
      processOperandsEVA(param);
    }

    void $cppClass::processOperandsSNR(int64_t param){
      if (!(isOperandCipher(0) && isOperandCipher(1))){
        OpBuilder builder(getOperation());
        int lowerIdx = isOperandCipher(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        if (getOperandScaleType(higherIdx).getScale() > param)
          getOperand(lowerIdx).setType(getOperandScaleType(higherIdx).toPlain());
        else{
          getOperand(lowerIdx).setType(getOperandScaleType(higherIdx).toPlain().switchScale(param));
          getOperation()->setOperand(higherIdx, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(higherIdx), param - getOperandScale(higherIdx)));
        }
        return;
      }
      if (getOperandScale(0) < param ){
        OpBuilder builder(getOperation());
        getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(0), param - getOperandScale(0)));
      }
      if (getOperandScale(1) < param ){
        OpBuilder builder(getOperation());
        getOperation()->setOperand(1, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(1), param - getOperandScale(1)));
      }
      processOperandsPARS(param);
    }

    bool $cppClass::isConsume() {
      return false;
    }
    bool $cppClass::isSingle() {
      return !(isOperandCipher(0) && isOperandCipher(1));
    }
    ::llvm::SmallVector<double, 2> $cppClass::differentiate(::llvm::ArrayRef<double> gradient,
        ::llvm::ArrayRef<double> estimation){
      return {gradient[0] , gradient[0]};
    }
    ::llvm::SmallVector<double, 1> $cppClass::estimateValue(::llvm::ArrayRef<double> estimation){
      return {1.0};
    }
  }];
  let hasCanonicalizer = 1;
}
def MulOp: Earth_Op<"mul", [Commutative]> {
  let arguments = (ins HETensor:$lhs, HETensor:$rhs);
  code extraClassDeclaration = [{
    void processOperandsEVA(int64_t param);
    void processOperandsPARS(int64_t param);
    void processOperandsSNR(int64_t param);
    void processResultsEVA(int64_t param);
  }];
  code extraClassDefinition = arithClassDefinition #[{
    void $cppClass::processOperandsEVA(int64_t param){
      OpBuilder builder(getOperation());
      if (!(isOperandCipher(0) && isOperandCipher(1))){
        int lowerIdx = isOperandCipher(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperand(lowerIdx).setType(getOperandScaleType(higherIdx).toPlain().switchScale(param));
        return;
      }
      if (getOperandLevel(0) != getOperandLevel(1)){
        int lowerIdx = getOperandLevel(0) < getOperandLevel(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperation()->setOperand(lowerIdx, builder.create<hecate::earth::ModswitchOp>( getLoc(), getOperand(lowerIdx), getOperandLevel(higherIdx) - getOperandLevel(lowerIdx)));
      }
    }
    void $cppClass::processOperandsPARS(int64_t param){
      OpBuilder builder(getOperation());
      if (!(isOperandCipher(0) && isOperandCipher(1))){
        int lowerIdx = isOperandCipher(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperand(lowerIdx).setType(getOperandScaleType(higherIdx).toPlain().switchScale(param));
      }

      if (getOperandLevel(0) != getOperandLevel(1)){
        int lowerIdx = getOperandScale(0) < getOperandScale(1) ? 0 : 1;
        getOperation()->setOperand(lowerIdx, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(lowerIdx), param+ hecate::earth::EarthDialect::rescalingFactor - getOperandScale(lowerIdx) ));
        getOperation()->setOperand(lowerIdx, builder.create<hecate::earth::RescaleOp>( getLoc(), getOperand(lowerIdx)));
      }

      processOperandsEVA(param);

      if (getOperandScale(0) + getOperandScale(1)  > 2* param + hecate::earth::EarthDialect::rescalingFactor){
        getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(0), param+ hecate::earth::EarthDialect::rescalingFactor - getOperandScale(0) ));
        getOperation()->setOperand(0, builder.create<hecate::earth::RescaleOp>( getLoc(), getOperand(0)));
        if (getOperandLevel(0)!= getOperandLevel(1)){
          getOperation()->setOperand(1, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(1), param+ hecate::earth::EarthDialect::rescalingFactor - getOperandScale(1) ));
          getOperation()->setOperand(1, builder.create<hecate::earth::RescaleOp>( getLoc(), getOperand(1)));
        }
      }
    }
    void $cppClass::processOperandsSNR(int64_t param){
      if (!(isOperandCipher(0) && isOperandCipher(1))){
        int lowerIdx = isOperandCipher(1) ? 0 : 1;
        int higherIdx = 1-lowerIdx;
        getOperand(lowerIdx).setType(getOperandScaleType(higherIdx).toPlain().switchScale(param));
        return;
      }
      /* if (getOperandScale(0)+getOperandScale(1) < param ){ */
      /*   OpBuilder builder(getOperation()); */
      /*   getOperation()->setOperand(0, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(0), param - getOperandScale(0))); */
      /* } */
      /* if (getOperandScale(1) < param ){ */
      /*   OpBuilder builder(getOperation()); */
      /*   getOperation()->setOperand(1, builder.create<hecate::earth::UpscaleOp>( getLoc(), getOperand(1), param - getOperandScale(1))); */
      /* } */
      processOperandsPARS(param);
    }

    void $cppClass::processResultsEVA(int64_t param){
      OpBuilder builder(getOperation());
      builder.setInsertionPointAfter(getOperation());
      IRRewriter rewriter(builder);
      auto target = getResult();
      while(target.getType().dyn_cast<HEScaleTypeInterface>().getScale() >= param + hecate::earth::EarthDialect::rescalingFactor){
        auto old = target;
        target = builder.create<RescaleOp>(getLoc(), target);
        rewriter.replaceAllUsesExcept(old, target, target.getDefiningOp());
      }
    }

    bool $cppClass::isConsume() {
      return true;
    }
    bool $cppClass::isSingle() {
      return !(isOperandCipher(0) && isOperandCipher(1));
    }
    ::llvm::SmallVector<double, 2> $cppClass::differentiate(::llvm::ArrayRef<double> gradient,
        ::llvm::ArrayRef<double> estimation){
      return {gradient[0] * estimation[1], gradient[0]*estimation[0]};
    }
    ::llvm::SmallVector<double, 1> $cppClass::estimateValue(::llvm::ArrayRef<double> estimation){
      return {estimation[0] * estimation[1]};
    }
  }];
  let hasCanonicalizer = 1;
}


#endif
