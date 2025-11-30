#pragma once
#include <stdint.h>

// typedef struct emit_3d{
//   uint cx, cy, cz;
//   uint lambda;
//   uint phase;
//   uint cutoff;
// } emit_3d;

#define cah_bi_name "CAH BiDi"

// typedef struct __attribute__((packed)){
//   uint8_t in;
//   uint8_t out;
// } cah_bi_cell;


typedef struct cah_bi{
  uint dim_x, dim_y;
  uint count;
  uint8_t* cells;
} cah_bi;


cah_bi* cah_bi_init( uint dim_x, uint dim_y );
void cah_bi_free( cah_bi* cah );
void cah_bi_step( cah_bi* cah );
void cah_bi_render( cah_bi* cah, Image* img );
