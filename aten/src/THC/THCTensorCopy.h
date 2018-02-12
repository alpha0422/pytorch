#ifndef TH_CUDA_TENSOR_COPY_INC
#define TH_CUDA_TENSOR_COPY_INC

#include "THCTensor.h"
#include "THCGeneral.h"
#include "THCHalf.h"
#include "THCStream.h"

#include "generic/THCTensorCopy.h"
#include "THCGenerateAllTypes.h"

void
THC_expandGrad(THCState* state,
               THCudaHalfTensor* dst,
               THCudaHalfTensor* src,
               uint64_t N,
               uint64_t C,
               uint64_t H,
               uint64_t W,
               uint64_t U,
               uint64_t V,
               uint64_t Hq,
               uint64_t Wq);

void
THC_reverseWeight(THCState* state,
                  THCudaHalfTensor* dst,
                  THCudaHalfTensor* src,
                  uint64_t C,
                  uint64_t K,
                  uint64_t R,
                  uint64_t S);

#endif
