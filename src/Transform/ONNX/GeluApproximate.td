//===-  GeluApprox.td - Rewriting for decomposing ONNX Ops -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines language-specific pattern match rewritings for ONNX using
// Declarative Rewrite Rules (DRR) specified using TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_GELU
#define ONNX_GELU

#ifndef OP_BASE
include "src/Dialect/ONNX/ONNXOps.td"
#endif // OP_BASE

/// Note: The DRR definition used for defining patterns is shown below:
///
/// class Pattern<
///    dag sourcePattern, list<dag> resultPatterns,
///    list<dag> additionalConstraints = [],
///    dag benefitsAdded = (addBenefit 0)
/// >;
def three : NativeCodeCall<"$_builder.getF32FloatAttr(3.0)">;
def onehalf : NativeCodeCall<"$_builder.getF32FloatAttr(0.5)">;
def one : NativeCodeCall<"$_builder.getF32FloatAttr(1.0)">;
def sqrtpi : NativeCodeCall<"$_builder.getF32FloatAttr(0.044715)">;
def scalar : NativeCodeCall<"$_builder.getF32FloatAttr(0.7978845608)">;
def GetNullAttr : NativeCodeCall<"Attribute()">;
def createDenseElementsAttrFromFloatAttr : NativeCodeCall<
  "createDenseElementsAttrFromFloatAttr($_builder, $0.getType().cast<ShapedType>().getElementType(), $1)">;

// (ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (constant)))
/*
def Gelu : Pat<
 (ONNXMulOp (ONNXMulOp (ONNXAddOp (ONNXTanhOp (ONNXMulOp (ONNXAddOp (ONNXMulOp (ONNXPowOp $x, $three), $scalar), $x), $sqrt2dividedpi)), $one), $x ),$onehalf),
(ONNXDivOp $x, (ONNXAddOp $one, (ONNXExpOp (ONNXMulOp $x, (ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (constant))) )))),
[],
  (addBenefit 2)>;


def Gelu2 : Pat<
 (ONNXMulOp (ONNXMulOp  $x , (ONNXAddOp (ONNXTanhOp (ONNXMulOp (ONNXAddOp (ONNXMulOp (ONNXPowOp $x, $three), $scalar), $x), $sqrt2dividedpi)), $one)),$onehalf),
(ONNXDivOp $x, (ONNXAddOp $one, (ONNXExpOp (ONNXMulOp $x, (ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (constant))) )))),
[],
  (addBenefit 2)>;
*/

def Gelu : Pat<
 (ONNXGeluOp $x),
 (ONNXMulOp (ONNXMulOp (ONNXAddOp (ONNXTanhOp (ONNXMulOp (ONNXAddOp (ONNXMulOp (ONNXPowOp $x, (ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (three)))
), (ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (scalar)))
), $x), (ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (sqrtpi)))
)), (ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (one)))
), $x ),(ONNXConstantOp (GetNullAttr), (createDenseElementsAttrFromFloatAttr $x, (onehalf)))
),

[],
  (addBenefit 2)>;


#endif // ONNX_GELU
