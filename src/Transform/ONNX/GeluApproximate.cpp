

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Pass/Passes.hpp"

using namespace mlir;


/// Include the patterns defined in the Declarative Rewrite framework.


namespace {


DenseElementsAttr createDenseElementsAttrFromFloatAttr(
    PatternRewriter &rewriter, Type elementType, FloatAttr attr) {
  SmallVector<int64_t, 1> dims(1, 1);
  SmallVector<float, 1> values(1, attr.getValue().convertToFloat());
  auto tensorType = mlir::RankedTensorType::get(dims, elementType);
  return mlir::DenseElementsAttr::get(tensorType, llvm::makeArrayRef(values));
}

#include "src/Transform/ONNX/GeluApproximate.inc"
struct GeluApproximatePass
    : public PassWrapper<GeluApproximatePass, FunctionPass> {
  void runOnFunction() final;
};
} // end anonymous namespace.

void GeluApproximatePass::runOnFunction() {
  auto function = getFunction();
  MLIRContext *context = &getContext();

  ConversionTarget target(getContext());
  target.addLegalDialect<ONNXOpsDialect>();
    target.addLegalDialect<ONNXOpsDialect>();
   target.addIllegalOp<ONNXPowOp>();
  OwningRewritePatternList patterns;
  populateWithGenerated(context, patterns);

  applyPatternsAndFoldGreedily(function, std::move(patterns));
} // end anonymous namespace

/*!
 * Create a DecomposeONNX pass.
 */
std::unique_ptr<mlir::Pass> mlir::createGeluApproximatePass() {
  return std::make_unique<GeluApproximatePass>();
}
