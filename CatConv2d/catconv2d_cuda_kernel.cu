#include <torch/extension.h>
#include <thrust/device_vector.h>

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cooperative_groups.h>
//#include <cuda/barrier>
#include <vector>
#include <vector_types.h>

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define PXL_GLOBAL_PTR   "l"
#else
#define PXL_GLOBAL_PTR   "r"
#endif

#define M 128
#define N 64
#define K 4


inline __device__ void __prefetch_global_(const void* const ptr)
{
  asm ("prefetch.global.L2 [%0];" : : PXL_GLOBAL_PTR(ptr));
  //asm volatile("prefetchu.L1 [%0];" : : PXL_GLOBAL_PTR(ptr));
  //asm volatile ("prefetch.global.L2 [%0];"::"l"(ptr) );
}


__global__ void  __launch_bounds__(128,3) catconv2d_cuda_3x3_128x16_forward_kernel(
    const float* __restrict__ input_0,
    const float* __restrict__ input_1,
    const float* __restrict__ input_2,
    const float* __restrict__ input_3,
    const float* __restrict__ input_4,
    const float* __restrict__ input_5,
    const float* __restrict__ input_6,
    const float* __restrict__ input_7,
    const float* __restrict__ input_8,
    const float* __restrict__ input_9,
    const float* __restrict__ input_10,
    const float* __restrict__ input_11,
    const float* __restrict__ input_12,
    const float* __restrict__ input_13,
    const float* __restrict__ input_14,
    const float* __restrict__ input_15,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int    input_sz_0,
    int    input_sz_1,
    int    input_sz_2,
    int    input_sz_3,
    int    input_sz_4,
    int    input_sz_5,
    int    input_sz_6,
    int    input_sz_7,
    int    input_sz_8,
    int    input_sz_9,
    int    input_sz_10,
    int    input_sz_11,
    int    input_sz_12,
    int    input_sz_13,
    int    input_sz_14,
    int    input_sz_15,

    bool   relu,
    int    in_sz,
    int    out_sz,
    int    h_sz,
    int    w_sz)
{
  //long long int time0;
  //asm volatile("mov.u64 %0, %%clock64;" : "=l"(time0));
  int warp_i = threadIdx.x;
  int warp_k = threadIdx.y;

  __shared__ int input_sz_[17];
  const __shared__ float*  __restrict__ input_list_[17];

  int tile_m = blockIdx.x * M;
  int tile_n = blockIdx.y * 16;

  int hw_sz = w_sz*h_sz;

  int w_sz_align = ((w_sz + 7) / 8) * 8;
  int h_sz_align = ((h_sz + 7) / 8) * 8;
  int hw_sz_align = h_sz_align*w_sz_align;
  
  int b       = blockIdx.x  / ((hw_sz_align + M - 1)/M);
  int hw_base = tile_m   - b*(((hw_sz_align + M - 1)/M)*M);


  input_sz_[0] = input_sz_0;
  input_sz_[1] = input_sz_1;
  input_sz_[2] = input_sz_2;
  input_sz_[3] = input_sz_3;
  input_sz_[4] = input_sz_4;
  input_sz_[5] = input_sz_5;
  input_sz_[6] = input_sz_6;
  input_sz_[7] = input_sz_7;
  input_sz_[8] = input_sz_8;
  input_sz_[9] = input_sz_9;
  input_sz_[10] = input_sz_10;
  input_sz_[11] = input_sz_11;
  input_sz_[12] = input_sz_12;
  input_sz_[13] = input_sz_13;
  input_sz_[14] = input_sz_14;
  input_sz_[15] = input_sz_15;
  input_sz_[16] = 256;
  input_list_[0] = input_0;
  input_list_[1] = input_1;
  input_list_[2] = input_2;
  input_list_[3] = input_3;
  input_list_[4] = input_4;
  input_list_[5] = input_5;
  input_list_[6] = input_6;
  input_list_[7] = input_7;
  input_list_[8] = input_8;
  input_list_[9] = input_9;
  input_list_[10] = input_10;
  input_list_[11] = input_11;
  input_list_[12] = input_12;
  input_list_[13] = input_13;
  input_list_[14] = input_14;
  input_list_[15] = input_15;
  input_list_[16] = input_15;


  extern __shared__ unsigned char smem_[];
  float* __restrict__ smem = (float *)smem_;

  float* __restrict__ sin  = smem;
  float* __restrict__ swt  = smem + 36*16 * K  ;


  
  int i  = (warp_i >>1) & 0xf;
  int r  = (warp_i & 0x1) << 4;
  int j  = (warp_i >> 5) + warp_k*2;
  
  int wi = (warp_i >> 2) + warp_k*16;
  int wj = (warp_i & 0x3);

  int sub_blk_i = (hw_base/64) + (r >> 4);
  int h_base = (sub_blk_i / (w_sz_align/8)) * 8;
  int w_base = (sub_blk_i % (w_sz_align/8)) * 8;


  int vi = 0;
  int m = j +4;

  int input_sz_cur = input_sz_0;
  const float* __restrict__ input_ptr_cur = input_0;


  bool w_padding = (w_base + i == 0) || (w_base + i - ((i/4)*2) - 1 >= w_sz);


  int sin_addr = (i&0x3)*36 + (i>>2)  + r + j*576;


  float res[64]= {0.0f};

  //////////////////////////////////////
  //             First DRAM Prefetch
  //////////////////////////////////////

  float tmp_vec_in[10];

  int addr  = b *input_sz_0*hw_sz + j*hw_sz + (h_base -1)*w_sz + w_base + i - ((i/4)*2) - 1;

#pragma unroll
  for(int n=0; n<10; n++)
  {
    bool h_padding = ((h_base + n ) == 0) || ((h_base + n -1) >= h_sz);

    tmp_vec_in[n] = (h_padding || w_padding)? 0.0f :  __ldg( input_0 + addr );
    addr += w_sz;
  }

  if ( m >= input_sz_0) {
    m -= input_sz_0;
    vi++;

    input_sz_cur  = input_sz_1;
    input_ptr_cur = input_1;
    //addr  = b *input_sz_1*hw_sz + m*hw_sz + (h_base -1)*w_sz + w_base + i - ((i/4)*2) - 1;
  }

  float tmp_wt_in[9];

  {
    int ich = wj + warp_k*4;
    int och = tile_n + (warp_i >> 2);
    int addr = (och*in_sz + ich)*9;
    bool wt_valid = (ich < in_sz) && (och < out_sz);

#pragma unroll
    for (int x=0; x<9; x++)
    {
      tmp_wt_in[x]  = wt_valid?  weights[ addr + x] : 0.0f;
    }
  }



  //////////////////////////////////////
  //             MAIN LOOP
  //////////////////////////////////////
  for (int k=0; k < in_sz; k+=4)
  {
    

    ////////////////////////////////////////////////////////////////////////
    float tmp_vec_in0[10];
    float hval[10];
    float vec_in_trans[16];

#pragma unroll
    for(int n=0; n<10; n++)
    {
      int pair_thread = (i & 0x2) == 0? 4 + (warp_i&0x1) : 2 + (warp_i&0x1);

      float f0 = (i & 0x3) == 3? -1.0f : 1.0f;
      float f1 = (i & 0x1) == 0? -1.0f : 1.0f;
      hval[n] = __shfl_sync(0xff, tmp_vec_in[n  ], pair_thread, 8);
  
      tmp_vec_in0[n] = f0*tmp_vec_in[n] + f1 * hval[n];
    }

#pragma unroll
    for(int n=0; n<16; n++) {
      int x = n/4 * 2 + (n&0x3);
      vec_in_trans[n] = (n&0x3) == 0? tmp_vec_in0[x] - tmp_vec_in0[x+2] :
                        (n&0x3) == 1? tmp_vec_in0[x] + tmp_vec_in0[x+1] :
                        (n&0x3) == 2? tmp_vec_in0[x] - tmp_vec_in0[x-1] :
                                      tmp_vec_in0[x-2] - tmp_vec_in0[x];


      sin[ sin_addr + ((n&0x3)*144 + (n>>2)*4)] = vec_in_trans[n];
    }

    int addr  = b *input_sz_cur*hw_sz + m*hw_sz + (h_base -1)*w_sz + w_base + i - ((i/4)*2) - 1;

#pragma unroll
    for(int n=0; n<10; n++)
    {
       bool h_padding = ((h_base == 0 && n == 0) || (h_base + n -1) >= h_sz) ;

       tmp_vec_in[n] = (h_padding || w_padding)? 0.0f : __ldg( input_ptr_cur + addr );

       addr += w_sz;
    }

    m+=4;
    if ( m >= input_sz_cur ) {
      m -= input_sz_cur;
      vi++;

      input_sz_cur  = input_sz_[vi];
      input_ptr_cur = input_list_[vi];
    }

    __syncthreads();

    float tmp_in[16];

#pragma unroll
    for( int p=0; p < 4; p++) {
      reinterpret_cast<float4* >(tmp_in)[p  ] = reinterpret_cast<float4* >(sin + p*576 + i*36 + j*4 + r)[0];
    }

    if((k & 0x4)==0)
    {
      
      float tmp_wt_trans[16];

      tmp_wt_trans[0] = tmp_wt_in[0];
      tmp_wt_trans[3] = tmp_wt_in[2];
      tmp_wt_trans[12] = tmp_wt_in[6];
      tmp_wt_trans[15] = tmp_wt_in[8];

      float tmp_012 = (tmp_wt_in[0] + tmp_wt_in[1] + tmp_wt_in[2])*0.5;
      float tmp_678 = (tmp_wt_in[6] + tmp_wt_in[7] + tmp_wt_in[8])*0.5;
      float tmp_036 = (tmp_wt_in[0] + tmp_wt_in[3] + tmp_wt_in[6])*0.5;
      float tmp_258 = (tmp_wt_in[2] + tmp_wt_in[5] + tmp_wt_in[8])*0.5;

      float tmp_345 = (tmp_wt_in[3] + tmp_wt_in[4] + tmp_wt_in[5])*0.5;

      tmp_wt_trans[1]  = tmp_012;
      tmp_wt_trans[2]  = tmp_012 - tmp_wt_in[1];
      tmp_wt_trans[13] = tmp_678;
      tmp_wt_trans[14] = tmp_678 - tmp_wt_in[7];

      tmp_wt_trans[4]  = tmp_036;
      tmp_wt_trans[8]  = tmp_036 - tmp_wt_in[3];
      tmp_wt_trans[7]  = tmp_258;
      tmp_wt_trans[11] = tmp_258 - tmp_wt_in[5];

      float tmp_345m4 = tmp_345 - tmp_wt_in[4];

      tmp_wt_trans[5] = (tmp_012 + tmp_678 + tmp_345)*0.5;
      tmp_wt_trans[6] = (tmp_wt_trans[2] + tmp_wt_trans[14] + tmp_345m4) * 0.5;
      tmp_wt_trans[9]  = tmp_wt_trans[5] - tmp_345;
      tmp_wt_trans[10] = tmp_wt_trans[6] - tmp_345m4;

      int order[16] = {0,3,12,15, 1,13,4,7, 2,14,8,11, 5,6,9,10};
#pragma unroll
      for (int xi=0; xi<16; xi++){
        int x = order[xi];
        swt[ (wi*17 + x)*4  + wj ] = tmp_wt_trans[x];    

      }

      int ich = k + 8 + wj + warp_k*4 ;
      int och = tile_n + (warp_i >> 2);
      int wtaddr = ((och * in_sz) + ich)*9 ;
      bool wt_valid = (ich < in_sz) && (och < out_sz);

#pragma unroll
      for (int x=0; x<9; x++)
      {
        tmp_wt_in[x] = wt_valid? weights[ wtaddr + x] : 0.0f;
      }
    
      __syncthreads();

    }
    //////////////////////////////////////////////////////////////////////

    float tmp_wt[4];

#pragma unroll
    for( int n=0; n <16 ; n++) {
      reinterpret_cast<float4* >(tmp_wt)[0] = reinterpret_cast<float4* >(swt + ((n + ((k>>2)&0x1)*16)*17 + i)*4 )[0];

#pragma unroll
      for( int x=0; x < 4; x++) {

#pragma unroll
        for( int p=0; p < 4; p++)
        {
          res[ p + n*4 ]   += tmp_in[p+x*4] * tmp_wt[x];
        }
	
      }
    }

      
      
    
    __syncthreads();
  } // End of input channl loop


  float res_trans[16];
#pragma unroll
  for(int n=0; n < 8; n++) //output chanel
#pragma unroll
    for(int x=0; x < 4; x++) //horizontal blocks (2x2)
    {
      sin[ n*32 + j*8 + x*2 + (r>>4) + i*258] = res[n*4 + x];
    }
  __syncthreads();

#pragma unroll
  for(int n=0; n<16; n++) {
    res[n]    = sin [ (warp_i + warp_k*64) + n*258];
    res[n+16] = sin [ (warp_i + warp_k*64 + 128) + n*258];
  }
  res_trans[ 0 ] = (res[0] + res[4] + res[8]) + (res[1] + res[5] + res[9]) + (res[2] + res[6] + res[10]);
  res_trans[ 1 ] = (res[1] + res[5] + res[9]) - (res[3] + res[7] + res[11]) - (res[2] + res[6] + res[10]);
  res_trans[ 2 ] = (res[4] + res[5] + res[6]) - (res[8] + res[9] + res[10]) - (res[12] + res[13] + res[14]);
  res_trans[ 3 ] = res[5] + res[10] + res[11] + res[14] + res[15] - (res[9] + res[13] + res[6] + res[7]);

  res_trans[ 4 ] = (res[16+0] + res[16+4] + res[16+8]) + (res[16+1] + res[16+5] + res[16+9]) + (res[16+2] + res[16+6] + res[16+10]);
  res_trans[ 5 ] = (res[16+1] + res[16+5] + res[16+9]) - (res[16+3] + res[16+7] + res[16+11]) - (res[16+2] + res[16+6] + res[16+10]);
  res_trans[ 6 ] = (res[16+4] + res[16+5] + res[16+6]) - (res[16+8] + res[16+9] + res[16+10]) - (res[16+12] + res[16+13] + res[16+14]);
  res_trans[ 7 ] = res[16+5] + res[16+10] + res[16+11] + res[16+14] + res[16+15] - (res[16+9] + res[16+13] + res[16+6] + res[16+7]);

  __syncthreads();

#pragma unroll
  for(int n=0; n<8; n++)
#pragma unroll
    for(int x=0; x < 4; x++) //horizontal blocks (2x2)
    {
      sin[ n*32 + j*8 + x*2 + (r>>4) + i*258] = res[n*4 + x + 32];
    }
  __syncthreads();

#pragma unroll
  for(int n=0; n<16; n++) {
    res[n]    = sin [ (warp_i + warp_k*64) + n*258];
    res[n+16] = sin [ (warp_i + warp_k*64 + 128) + n*258];
  }
  res_trans[ 8] = (res[0] + res[4] + res[8]) + (res[1] + res[5] + res[9]) + (res[2] + res[6] + res[10]);
  res_trans[ 9] = (res[1] + res[5] + res[9]) - (res[3] + res[7] + res[11]) - (res[2] + res[6] + res[10]);
  res_trans[10] = (res[4] + res[5] + res[6]) - (res[8] + res[9] + res[10]) - (res[12] + res[13] + res[14]);
  res_trans[11] = res[5] + res[10] + res[11] + res[14] + res[15] - (res[9] + res[13] + res[6] + res[7]);

  res_trans[12] = (res[16+0] + res[16+4] + res[16+8]) + (res[16+1] + res[16+5] + res[16+9]) + (res[16+2] + res[16+6] + res[16+10]);
  res_trans[13] = (res[16+1] + res[16+5] + res[16+9]) - (res[16+3] + res[16+7] + res[16+11]) - (res[16+2] + res[16+6] + res[16+10]);
  res_trans[14] = (res[16+4] + res[16+5] + res[16+6]) - (res[16+8] + res[16+9] + res[16+10]) - (res[16+12] + res[16+13] + res[16+14]);
  res_trans[15] = res[16+5] + res[16+10] + res[16+11] + res[16+14] + res[16+15] - (res[16+9] + res[16+13] + res[16+6] + res[16+7]);

 
#pragma unroll
  for(int n=0; n<16; n++)
    res_trans[n] += bias[ tile_n + (n>>2)*4 + j];

  if(relu) {
#pragma unroll
    for(int n=0; n<16; n++)
      res_trans[n] = res_trans[n]<0? 0 : res_trans[n];
  }

#pragma unroll
    for( int n=0; n<4; n++) {
      int och = tile_n + n*4 + j;
      int h = h_base + (i>>2 )*2;
      int w = w_base + (i&0x3)*2;
#pragma unroll
       for (int x=0; x < 4; x++){

        bool valid = (och < out_sz) && (h + (x>>1)) < h_sz && (w + (x&0x1) < w_sz);
        if(valid)  output[  b*out_sz*hw_sz + och*hw_sz + (h + (x>>1))*w_sz + w + (x&0x1) ] = res_trans[ n*4 + x ];
       }
    
  }    
  
}






//template <typename scalar_t>
__global__ void  __launch_bounds__(128,4)  catconv2d_cuda_1x1_128x64_forward_kernel(
    const float* __restrict__ input_0,
    const float* __restrict__ input_1,
    const float* __restrict__ input_2,
    const float* __restrict__ input_3,
    const float* __restrict__ input_4,
    const float* __restrict__ input_5,
    const float* __restrict__ input_6,
    const float* __restrict__ input_7,
    const float* __restrict__ input_8,
    const float* __restrict__ input_9,
    const float* __restrict__ input_10,
    const float* __restrict__ input_11,
    const float* __restrict__ input_12,
    const float* __restrict__ input_13,
    const float* __restrict__ input_14,
    const float* __restrict__ input_15,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int    input_sz_0,
    int    input_sz_1,
    int    input_sz_2,
    int    input_sz_3,
    int    input_sz_4,
    int    input_sz_5,
    int    input_sz_6,
    int    input_sz_7,
    int    input_sz_8,
    int    input_sz_9,
    int    input_sz_10,
    int    input_sz_11,
    int    input_sz_12,
    int    input_sz_13,
    int    input_sz_14,
    int    input_sz_15,
  
    bool   relu,
    int    in_sz,
    int    out_sz,
    int    h_sz,
    int    w_sz) 
{
  int warp_i = threadIdx.x;
  int warp_k = threadIdx.y;  

  __shared__ int input_sz_[16];
  __shared__ const float*  __restrict__ input_list_[16];
  
  int tile_m = blockIdx.x * M;
  int tile_n = blockIdx.y * N;


  int hw_sz = h_sz*w_sz;
  int b       = blockIdx.x  / ((hw_sz + M - 1)/M);
  int hw_base = tile_m   - b*(((hw_sz + M - 1)/M)*M);

  input_sz_[0] = input_sz_0;  
  input_sz_[1] = input_sz_1;  
  input_sz_[2] = input_sz_2;  
  input_sz_[3] = input_sz_3;  
  input_sz_[4] = input_sz_4;  
  input_sz_[5] = input_sz_5;  
  input_sz_[6] = input_sz_6;  
  input_sz_[7] = input_sz_7;  
  input_sz_[8] = input_sz_8;  
  input_sz_[9] = input_sz_9;  
  input_sz_[10] = input_sz_10; 
  input_sz_[11] = input_sz_11; 
  input_sz_[12] = input_sz_12; 
  input_sz_[13] = input_sz_13; 
  input_sz_[14] = input_sz_14; 
  input_sz_[15] = input_sz_15; 
  input_list_[0] = input_0;
  input_list_[1] = input_1;
  input_list_[2] = input_2;
  input_list_[3] = input_3;
  input_list_[4] = input_4;
  input_list_[5] = input_5;
  input_list_[6] = input_6;
  input_list_[7] = input_7;
  input_list_[8] = input_8;
  input_list_[9] = input_9;
  input_list_[10] = input_10;
  input_list_[11] = input_11;
  input_list_[12] = input_12;
  input_list_[13] = input_13;
  input_list_[14] = input_14;
  input_list_[15] = input_15;


  extern __shared__ unsigned char smem_[];
  float* __restrict__ smem = (float *)smem_;

  float* __restrict__ sin  = smem; 
  float* __restrict__ swt  = smem + M * K  ;
  float* __restrict__ sb   = smem + M * K  + N * K;

  
  int i  = warp_i & 0x3;
  int j  = warp_i >> 2;


  int vi = 0;
  int m = i+4;

  int input_sz_cur = input_sz_0;
  const float* __restrict__ input_ptr_cur = input_0;
  
  float tmp0_vec_in[4];
  float tmp0_wt_in[2];


#pragma unroll
  for(int n=0; n<4; n++) {
    int hw = hw_base + j + warp_k*64 + n*16;

    tmp0_vec_in[n] = hw < hw_sz? input_ptr_cur[ b *input_sz_cur*hw_sz + i*hw_sz + hw ] : 0;
  }

#pragma unroll
  for(int n=0; n<4; n++) {
    sin[ (j + n*16 + warp_k*64)*4 + i ]  = tmp0_vec_in[n];
  }


#pragma unroll
  for (int x=0; x<2; x++) {
    tmp0_wt_in[x] =
        x*16 + warp_k*32 + j + tile_n < out_sz?
        weights[ (tile_n + x*16 + warp_k*32 + j)*in_sz + i ] : 0;
  }

#pragma unroll
  for (int x=0; x<2; x++) {
    swt[ (x*16 + warp_k*32 + j)*4 + i  ] = tmp0_wt_in[x];
  }

  __syncthreads();

  
  if ( m >= input_sz_cur && vi<15) {
    m -= input_sz_cur;
    vi++;
    input_sz_cur  = input_sz_[vi];
    input_ptr_cur = input_list_[vi];
  }



  float res[64] = {0};

  

  //////////////////////////////////////
  //             MAIN LOOP
  //////////////////////////////////////
  for (int k=0; k < in_sz; k+=4) 
  {

    float tmp_wt[4];
    float tmp_in[16];

#pragma unroll
    for( int p=0; p < 4; p++) {
      reinterpret_cast<float4* >(tmp_in)[p] = reinterpret_cast<float4* >( sin + (p*32 + (warp_i & 0x1f))*4 )[0];
    }
#pragma unroll
    for( int n=0; n <16 ; n++) {
      reinterpret_cast<float4* >(tmp_wt)[0] = reinterpret_cast<float4* >( swt + (n*4 + warp_k*2 + (warp_i >> 5))*4 )[0];

#pragma unroll
      for( int p=0; p < 4; p++) {
	
#pragma unroll
        for( int x=0; x < 4; x++)
        {
          res[  p*16+n ]   += tmp_in[p*4+x] * tmp_wt[x];
	}

      }
    }
 
    //////////////////////////////////////
    //           DRAM fetch
    //////////////////////////////////////
    float tmp1_vec_in[4];
    float tmp1_wt_in[2];   
    

#pragma unroll
    for(int n=0; n<4; n++) 
    {
      int hw = hw_base + j + warp_k*64 + n*16;

      tmp1_vec_in[n] = hw <hw_sz? input_ptr_cur[ b *input_sz_cur*hw_sz + m*hw_sz + hw ] : 0;
    }

#pragma unroll
    for (int x=0; x<2; x++) 
    {
      tmp1_wt_in[x] =
           i + k + 4 < in_sz && (tile_n + x*16 + warp_k*32 + j) < out_sz?
           weights[ ((tile_n + x*16 + warp_k*32 + j) * in_sz) + i + k + 4 ] : 0;
    }


    __syncthreads();


#pragma unroll
    for(int n=0; n<4; n++) 
    {
      sin[ (j + n*16 + warp_k*64)*4 + i ]  = tmp1_vec_in[n];
    }

#pragma unroll
    for (int x=0; x<2; x++) 
    {
      swt[  (x*16 + warp_k*32 + j)*4 + i  ] = tmp1_wt_in[x];
    }

    m+=4;
    if ( m >= input_sz_cur && vi < 15) {
      m -= input_sz_cur;
      vi++;
      input_sz_cur  = input_sz_[vi];
      input_ptr_cur = input_list_[vi];

      __prefetch_global_( input_ptr_cur + b *input_sz_cur*hw_sz + m*hw_sz + hw_base+warp_k*64);
    }
    __syncthreads();


  } // End of input channl loop


#pragma unroll
    for(int n=0; n<64; n++) 
      res[n] += bias[ tile_n + (n&0xf)*4 + (warp_i >> 5) + warp_k*2];

  if(relu) {
#pragma unroll
    for(int n=0; n<64; n++)
      res[n] = res[n]<0? 0 : res[n];
  }

#pragma unroll
    for( int n=0; n<16; n++) {
      int och = tile_n + n*4 + (warp_i>>5) +warp_k*2;

#pragma unroll
      for (int x=0; x < 4; x++){
        int hw = hw_base + (warp_i&0x1f)  + x*32;

        bool valid = hw < hw_sz && och < out_sz;
        if(valid)  output[ b*out_sz*hw_sz + och*hw_sz + hw ] = res[ n + x*16 ];
      }
    }
  
}





template <typename scalar_t>
__global__ void lltm_cuda_backward_kernel(
          torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> d_output,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> X,
    const torch::PackedTensorAccessor<scalar_t,4,torch::RestrictPtrTraits,size_t> weights,
    const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> bias) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
}
//} //a namespace



std::vector<torch::Tensor> catconv2d_cuda_forward(
    std::vector<torch::Tensor> input_list,
    torch::Tensor weights,
    torch::Tensor bias,
    bool relu) {

  auto X = input_list;

  const auto batch_size = input_list[0].size(0);
  const auto h_size     = input_list[0].size(2);
  const auto w_size     = input_list[0].size(3);
  const auto ich        = weights.size(1);
  const auto outch      = weights.size(0);
  bool conv1x1 = (weights.size(2) == 1 && weights.size(3) == 1);
  bool conv3x3 = (weights.size(2) == 3 && weights.size(3) == 3);

  auto dtype = input_list[0].type();
  auto dev = input_list[0].device();
  auto output = torch::empty({batch_size, outch, h_size, w_size}, dtype);

  int list_sz = input_list.size();
  int cn = conv1x1? N : conv3x3? 16 : 0;
  int usage = conv1x1? M*K + N*K : 
              conv3x3?  36*16*K + 17*32*K : 0;

  int w_sz_align = (w_size + 7) / 8 * 8;
  int h_sz_align = (h_size + 7) / 8 * 8;

  int hw_sz = conv1x1? w_size* h_size :
              conv3x3? h_sz_align*w_sz_align : 0;

  
  const dim3 threads(64, 2);
  //const dim3 blocks(  (outch+cn-1)/cn, batch_size*((hw_sz + M-1)/M));
  const dim3 blocks(  batch_size*((hw_sz + M-1)/M), (outch+cn-1)/cn);

  int list_ptrs[16];
  int list_szs[16];

  for(int i=0; i<16; i++){
    if(list_sz > i){
      list_ptrs[i] = i;
      list_szs[i] = input_list[i].size(1);
    }
    else{
      list_ptrs[i] = list_sz-1;
      list_szs[i] = 0;
    }
  }




  if (conv1x1) {
    catconv2d_cuda_1x1_128x64_forward_kernel<<<blocks, threads, usage*sizeof(float)>>>(
    
        input_list[ list_ptrs[0] ].data<float>(),
        input_list[ list_ptrs[1] ].data<float>(),
	input_list[ list_ptrs[2] ].data<float>(),
	input_list[ list_ptrs[3] ].data<float>(),
	input_list[ list_ptrs[4] ].data<float>(),
	input_list[ list_ptrs[5] ].data<float>(),
	input_list[ list_ptrs[6] ].data<float>(),
	input_list[ list_ptrs[7] ].data<float>(),
	input_list[ list_ptrs[8] ].data<float>(),
	input_list[ list_ptrs[9] ].data<float>(),
	input_list[ list_ptrs[10] ].data<float>(),
	input_list[ list_ptrs[11] ].data<float>(),
	input_list[ list_ptrs[12] ].data<float>(),
	input_list[ list_ptrs[13] ].data<float>(),
	input_list[ list_ptrs[14] ].data<float>(),
	input_list[ list_ptrs[15] ].data<float>(),
	
	weights.data<float>(),
	bias.data<float>(),
	output.data<float>(),

	 list_szs[0] ,
	 list_szs[1] ,
	 list_szs[2] ,
	 list_szs[3] ,
	 list_szs[4] ,
	 list_szs[5] ,
	 list_szs[6] ,
	 list_szs[7] ,
	 list_szs[8] ,
	 list_szs[9] ,
	 list_szs[10] ,
	 list_szs[11] ,
	 list_szs[12] ,
	 list_szs[13] ,
	 list_szs[14] ,
	 list_szs[15] ,

        relu,
	ich,outch,h_size,w_size);
  }
  else if(conv3x3) {

    catconv2d_cuda_3x3_128x16_forward_kernel<<<blocks, threads, usage*sizeof(float)>>>(
   
        input_list[ list_ptrs[0] ].data<float>(),
        input_list[ list_ptrs[1] ].data<float>(),
        input_list[ list_ptrs[2] ].data<float>(),
        input_list[ list_ptrs[3] ].data<float>(),
        input_list[ list_ptrs[4] ].data<float>(),
        input_list[ list_ptrs[5] ].data<float>(),
        input_list[ list_ptrs[6] ].data<float>(),
        input_list[ list_ptrs[7] ].data<float>(),
        input_list[ list_ptrs[8] ].data<float>(),
        input_list[ list_ptrs[9] ].data<float>(),
        input_list[ list_ptrs[10] ].data<float>(),
        input_list[ list_ptrs[11] ].data<float>(),
        input_list[ list_ptrs[12] ].data<float>(),
        input_list[ list_ptrs[13] ].data<float>(),
        input_list[ list_ptrs[14] ].data<float>(),
        input_list[ list_ptrs[15] ].data<float>(),

        weights.data<float>(),
        bias.data<float>(),
        output.data<float>(),

         list_szs[0] ,
         list_szs[1] ,
         list_szs[2] ,
         list_szs[3] ,
         list_szs[4] ,
         list_szs[5] ,
         list_szs[6] ,
         list_szs[7] ,
         list_szs[8] ,
         list_szs[9] ,
         list_szs[10] ,
         list_szs[11] ,
         list_szs[12] ,
         list_szs[13] ,
         list_szs[14] ,
         list_szs[15] ,

        relu,
        ich,outch,h_size,w_size);  
  }

  return { output };
}






std::vector<torch::Tensor> catconv2d_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor X,
    torch::Tensor weights,
    torch::Tensor bias) {


  //auto d_X = d_gate_weights.mm(weights);
  //auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  //auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {grad_output };//{d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
