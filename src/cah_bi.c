#include "common.h"
#include <string.h>
#include <math.h>
#include <sys/mman.h>

#include "cah_bi.h"

#define LAMBDA      16
#define NTHREADS    16

#define IN 0
#define OUT 1

#define RULES { \
  0, 1, 2, 3, 4, 5, 6, 7, \
  8, 9, 10, 38, 12, 22, 14, 15, \
  16, 17, 18, 37, 20, 42, 13, 23, \
  24, 52, 44, 27, 28, 29, 30, 31, \
  32, 33, 34, 35, 36, 19, 11, 39, \
  40, 50, 21, 43, 26, 45, 46, 47, \
  48, 49, 41, 51, 25, 53, 54, 55, \
  56, 57, 58, 59, 60, 61, 62, 63 \
}
uint g_rules[64] = RULES;

uint8_t sd_map[6] = {3, 4, 5, 0, 1, 2};
uint gcount = 0;

#define CELL(x, y, z) cah->cells[((x) * cah->dim_y + (y)) * 2 + (z)]

static void emit(cah_bi* cah, int x, int y, int lamda);


cah_bi* cah_bi_init( uint dim_x, uint dim_y ) {
printf("start cah_bi_init x: %d, y: %d\n", dim_x, dim_y);

  cah_bi* cah = malloc( sizeof(cah_bi) );
  if ( !cah ) {printf("malloc fail cah_bi\n"); return 0;}

  // store the easy stuff
  cah->count = 0;
  cah->dim_x = dim_x;
  cah->dim_y = dim_y;

  // allocate the cells
  size_t size = sizeof(uint8_t) * dim_x * dim_y * 2;
printf("malloc cells size %ld\n", size);

  cah->cells = malloc( size );
  if ( !cah->cells ) {
    printf("malloc fail cah_bi cells\n");
    free(cah);
    return 0;
  }
  memset(cah->cells, 0, size);

printf("init cells\n");
  // fill the cells with randomness
// uint8_t (*cells)[dim_y][2] = (uint8_t (*)[dim_y][2])cah->cells;
  for ( uint x = 0; x < dim_x; x++) {
    for ( uint y = 0; y < dim_y; y++) {
      CELL(x,y,IN) = rand() % 64;
      CELL(x,y,OUT) = rand() % 64;
      // cells[x][y][IN] = rand() % 64;
      // cells[x][y][OUT] = rand() % 64;
  }}

  emit( cah, dim_x / 2, 50, 32 );

printf("end cah_bi_init\n");
  return cah;
}

void cah_bi_free( cah_bi* cah ) {
  if ( cah ) {
    if ( cah->cells ) { free(cah->cells); }
    free(cah);
  }
}


static void emit(cah_bi* cah, int cx, int cy, int radius) {
  int radius_sq = radius * radius;

  // Bounding box optimization
  int x_min = (cx - radius < 0) ? 0 : cx - radius;
  int x_max = (cx + radius >= cah->dim_x) ? cah->dim_x - 1 : cx + radius;
  int y_min = (cy - radius < 0) ? 0 : cy - radius;
  int y_max = (cy + radius >= cah->dim_y) ? cah->dim_y - 1 : cy + radius;

  for (int y = y_min; y <= y_max; y++) {
      for (int x = x_min; x <= x_max; x++) {
          // Calculate squared distance from center
          int dx = x - cx;
          int dy = y - cy;
          int dist_sq = dx * dx + dy * dy;
          
          // If inside circle, set to on
          if (dist_sq <= radius_sq) {
              int idx = y * cah->dim_x + x;
              CELL(x,y,IN) = 63;
              CELL(x,y,OUT) = 63;
          }
      }
  }
}

inline static void move_bit( uint8_t* src, uint8_t src_bit, uint8_t* dst, uint8_t dst_bit ) {
  // only do the move if there is something to move!
  if ( (*src & (1 << src_bit)) ) {
    // only do the move in the input slot is empty
    if (!(*dst & (1 << dst_bit))) {
      // set the dst bit and clear the src bit
      *dst |= (1 << dst_bit);
      *src &= ~(1 << src_bit);
    }
  }
}

//---------------------------------------------------------
// rotate 6 bits worth of n in a "random" direction
inline static uint8_t rand_rot6( uint8_t n ) {
  if ( 1 && gcount++ ) {
    return rot6l(n);
  } else {
    return rot6r(n);
  }
}

inline static uint8_t* pcell_wrap( cah_bi* cah, int x, int y, uint io ) {
  if (x < 0) { x = cah->dim_x - 1; }
  if (x >= cah->dim_x) { x = 0; }
  if (y < 0) { x = cah->dim_y - 1; }
  if (y >= cah->dim_y) { y = 0; }
  return &CELL(x,y,io);
}

inline static uint8_t run_rule( uint8_t n ) {
  switch (n) {
    case 9:
    case 18:
    case 36:
    case 27:
    case 45:
    case 54:
      return rand_rot6(n);
    // case 0:
    //   return do_empty();
    // case 63:
    //   return do_full();
  }
  // return n;
  return g_rules[n];
}

void cah_bi_step( cah_bi* cah ){
  // usleep(100000);
// printf("start cah_bi_step\n");

  int x,y;
  uint dx = cah->dim_x;
  uint dy = cah->dim_y;

// uint8_t (*cells)[dim_y][2] = (uint8_t (*)[dim_y][2])cah->cells;

  uint steps = dx * dy * 100;
  for ( uint n = 0; n < steps; n++ ){
    // pick a random cell to operate on 
    x = rand() % dx;  // 0 to dim_x-1
    y = rand() % dy;  // 0 to dim_y-1

    uint8_t* in = &CELL(x,y,IN);
    uint8_t* out = &CELL(x,y,OUT);

    // // step the cells out
    move_bit( out, 0, pcell_wrap(cah,x-1,y,IN), 3 );
    move_bit( out, 1, pcell_wrap(cah,x-1,y+1,IN), 4 );
    move_bit( out, 2, pcell_wrap(cah,x,y+1,IN), 5 );
    move_bit( out, 3, pcell_wrap(cah,x+1,y+1,IN), 0 );
    move_bit( out, 4, pcell_wrap(cah,x+1,y,IN), 1 );
    move_bit( out, 5, pcell_wrap(cah,x,y-1,IN), 2 );

    // // transform the incoming bits
    CELL(x,y,IN) = run_rule( CELL(x,y,IN) );

    // // move in to out
    move_bit( in, 0, out, 0 );
    move_bit( in, 1, out, 1 );
    move_bit( in, 2, out, 2 );
    move_bit( in, 3, out, 3 );
    move_bit( in, 4, out, 4 );
    move_bit( in, 5, out, 5 );
  }

// printf("end cah_bi_step\n");
}

void cah_bi_render( cah_bi* cah, Image* img ){
// printf("start cah_bi_render\n");
  int v, r, g;
  uint dim_x = cah->dim_x;
  uint dim_y = cah->dim_y;

  ImageClearBackground( img, BLACK );

  for ( uint x = 0; x < dim_x; x++) {
    for ( uint y = 0; y < dim_y; y++) {
      v = bits(CELL(x,y,IN)) + bits(CELL(x,y,OUT));
      v -= 6;
      if ( v >= 0 ) { r = 0; g = v * 42; }
      else { r = (-1 * v) * 42; g = 0; }
      Color color = {r, g, 0, 255};
      ImageDrawPixel(img, x, y, color);
  }}

// printf("end cah_bi_render\n");
}


/*
void render_3d(Image* img, sim_3d* sim ) {
  ImageClearBackground( img, BLACK );

  int amplitude;
  uint r, g, b;
  uint dim = sim->dim;
  uint plane = dim / 2;
  float (*present)[dim][dim] = (float (*)[dim][dim])sim->a;

  float max, min, sum;
  max = present[1][1][plane];
  min = max;
  sum = 0;

  // render out
  for ( uint y = 0; y < dim; y++) {
    for ( uint x = 0; x < dim; x++) {
      // if (present[x][y][plane] > max) {max = present[x][y][plane];}
      // if (present[x][y][plane] < min) {min = present[x][y][plane];}
      // sum += present[x][y][plane];

      amplitude = present[x][y][plane]; // color distribution.
      b = fabs(0.5 * amplitude);   // produces complementary magenta and emerald green.

      if ( amplitude > 0) {
        g = amplitude; // green color for positive amplitude.
        if (g > 255) r = g - 255; else r = 0;
      } else {
        r = -amplitude; // red color for negative amplitude.
        if (r > 255) g = r - 255; else g = 0;
      }
      if (r > 255) r = 255;
      if (g > 255) g = 255;
      if (b > 255) b = 255;

      Color color = {r, g, b, 255};
      ImageDrawPixel(img, x, y, color);
    }
  }

  char buff[100];
  sprintf( buff, "%d", sim->count );
  ImageDrawText(img, buff, 10, 10, 20, WHITE);

  // sprintf( buff, "min: %f", min );
  // ImageDrawText(img, buff, 10, 30, 20, WHITE);
  // sprintf( buff, "max: %f", max );
  // ImageDrawText(img, buff, 10, 50, 20, WHITE);
  // sprintf( buff, "avg: %f", sum / (dim * dim) );
  // ImageDrawText(img, buff, 10, 70, 20, WHITE);

}
*/

























/*

// init the sim
sim_3d* init_cah_bi( uint dim_x, uint dim_y ){
  sim_3d* sim = malloc( sizeof(sim_3d) );
  if ( !sim ) {printf("malloc fail sim_3d\n"); return 0;}
  memset(sim, 0, sizeof(sim_3d));

  // store the easy stuff
  sim->dim = dim;

  // allocate the large sim buffers
  size_t size = sizeof(float) * dim * dim * dim;
  sim->a = malloc( size );
  if (!sim->a) {printf("malloc fail a\n"); return 0;}
  sim->b = malloc( size );
  if (!sim->b) {printf("malloc fail b\n"); return 0;}

  // initialize the large buffers
  memset(sim->a, 0, size);
  memset(sim->b, 0, size);

  // lock the buffers in ram
  mlock(sim->a, size);
  mlock(sim->b, size);

  // set up the emitters
  sim->emit_count = 2;
  size = sizeof(emit_3d) * sim->emit_count ;
  sim->emit = malloc( size );
  memset( sim->emit, 0, size );
  uint c = dim / 2;
  sim->emit[0].cx = c - LAMBDA * 4;
  sim->emit[0].cy = c;
  sim->emit[0].cz = c;// - LAMBDA * 2;
  sim->emit[0].lambda = LAMBDA;
  sim->emit[0].phase = 0;
  sim->emit[0].cutoff = 1000;
  sim->emit[1].cx = c + LAMBDA * 4;
  sim->emit[1].cy = c;
  sim->emit[1].cz = c;// + LAMBDA * 2;
  sim->emit[1].lambda = LAMBDA;
  sim->emit[1].phase = 0;
  sim->emit[1].cutoff = 1000;

  // return the struct
  return sim;
}


// close the sim
void free_sim_3d( sim_3d* sim ) {
  if( !sim ) {return;}
  free( sim->a );
  free( sim->b );
  memset(sim, 0, sizeof(sim_3d));
  free( sim );
}

void do_emit_3d_one( sim_3d* sim, emit_3d* emit ) {
  if( emit->cutoff && (emit->cutoff < sim->count) ) {return;}

  uint dim = sim->dim;
  uint cx = emit->cx;
  uint cy = emit->cy;
  uint cz = emit->cz;
  uint lambda = emit->lambda;
  float (*present)[dim][dim] = (float (*)[dim][dim])sim->a;
  float (*trend)[dim][dim] = (float (*)[dim][dim])sim->b;

  int x, y, z; // three-dimensional Cartesian coordinates.
  float distance, curve, radian, impulse = 1000000 / (lambda * lambda * lambda);
  float xSquared, ySquared, zSquared; // float advisable for use with sqrt operator below.
  int halfLambda = lambda / 2; // full lambda cubic area below.
  float phase = 2 * M_PI * sim->count / lambda; // amplitude modulation.

  for (x = -halfLambda; x < halfLambda; x++) {
      xSquared = x * x;
      for (y = -halfLambda; y < halfLambda; y++) {
          ySquared = y * y;
          for (z = -halfLambda; z < halfLambda; z++) {
              zSquared = z * z; // scanning the cube in-depth.
              distance = sqrt(xSquared + ySquared + zSquared); // distance to the center (Pythagoras).
              radian = 2 * M_PI * distance / lambda;
              if (distance <= halfLambda) { // spherical source (radius = lambda / 2).
                  if (radian < .001) radian = .001; // avoid division by zero.
                  curve = sin(radian) / radian; // sinus cardinalis (Jocelyn Marcotte's electron curve).
                  present[x+cx][y+cy][z+cz] += cos(phase) * curve * impulse;
                  trend[x+cx][y+cy][z+cz] -= cos(phase) * curve * impulse;
              }
  }}}
}


void do_emit_3d( sim_3d* sim ) {
  if (!sim || !sim->emit_count || !sim->emit ) {return;}
  for ( uint i = 0; i < sim->emit_count; i++ ) {
    do_emit_3d_one( sim, &sim->emit[i] );
  }
}

void render_3d(Image* img, sim_3d* sim ) {
  ImageClearBackground( img, BLACK );

  int amplitude;
  uint r, g, b;
  uint dim = sim->dim;
  uint plane = dim / 2;
  float (*present)[dim][dim] = (float (*)[dim][dim])sim->a;

  float max, min, sum;
  max = present[1][1][plane];
  min = max;
  sum = 0;

  // render out
  for ( uint y = 0; y < dim; y++) {
    for ( uint x = 0; x < dim; x++) {
      // if (present[x][y][plane] > max) {max = present[x][y][plane];}
      // if (present[x][y][plane] < min) {min = present[x][y][plane];}
      // sum += present[x][y][plane];

      amplitude = present[x][y][plane]; // color distribution.
      b = fabs(0.5 * amplitude);   // produces complementary magenta and emerald green.

      if ( amplitude > 0) {
        g = amplitude; // green color for positive amplitude.
        if (g > 255) r = g - 255; else r = 0;
      } else {
        r = -amplitude; // red color for negative amplitude.
        if (r > 255) g = r - 255; else g = 0;
      }
      if (r > 255) r = 255;
      if (g > 255) g = 255;
      if (b > 255) b = 255;

      Color color = {r, g, b, 255};
      ImageDrawPixel(img, x, y, color);
    }
  }

  char buff[100];
  sprintf( buff, "%d", sim->count );
  ImageDrawText(img, buff, 10, 10, 20, WHITE);

  // sprintf( buff, "min: %f", min );
  // ImageDrawText(img, buff, 10, 30, 20, WHITE);
  // sprintf( buff, "max: %f", max );
  // ImageDrawText(img, buff, 10, 50, 20, WHITE);
  // sprintf( buff, "avg: %f", sum / (dim * dim) );
  // ImageDrawText(img, buff, 10, 70, 20, WHITE);

}




// Typical use: x is signal level, y is gain adjustment
// High signal (x) -> limit gain (y) more aggressively
float dynamic_limiter(float gain_y, float signal_x) {
  // float threshold = 10000.0;
  // float abs_signal = fabs(signal_x);
  
  // if (abs_signal < threshold) {
      // Below threshold: linear
      return gain_y;
  // } else {
  //     // Above threshold: compress gain based on how much signal exceeds
  //     float excess = abs_signal - threshold;
  //     float compression_ratio = 1.0 / (1.0 + excess * 2);
  //     return gain_y * compression_ratio;
  // }
}


// Typical use: x is signal level, y is gain adjustment
// High signal (x) -> limit gain (y) more aggressively
float dynamic_gain(float initial_gain, float x) {
  return initial_gain;
  
    // Maps x to gain between 1.0 and max_gain
    double max_gain = initial_gain * 2;
    double steepness = 0.2;
    
    // tanh goes from -1 to +1, we shift to 1 to max_gain
    double normalized = tanh(fabs(x) * steepness);
    double gain = 1.0 + normalized * (max_gain - 1.0);
    
    return initial_gain * gain;
}


typedef struct d3_get_context{
  void* p;
  uint dim;
  float current;
  float gain;
  float* out_acc;
  float* gain_acc;
} d3_get_context;

void d3_get(d3_get_context g, uint x, uint y, uint z ) {
  uint dim = g.dim;
  float (*present)[dim][dim] = (float (*)[dim][dim])g.p;

  float value = present[x][y][z];
  // float gain = g.gain;
  float gain = dynamic_gain( g.gain, g.current );
  // float gain = dynamic_limiter( g.gain, g.current );
  // float gain = dynamic_limiter( g.gain, value );

  // printf("orig: %f, pow: %f\n", g.gain, gain);

  *g.out_acc += value * gain;
  *g.gain_acc += gain;
  return;
}


void step_slice_3d( uint nslice, uint slices, sim_3d* sim ) {
  uint dim = sim->dim;
  uint slice = dim / slices + (dim % slices) / slices;
  uint start = nslice * slice;
  uint stop = start + slice;

  // account for remainders and offsets
  if ( start == 0 ) { start = 1; }
  if ( nslice == slices - 1) { stop = dim - 1; }

  float (*past)[dim][dim] = (float (*)[dim][dim])sim->a;
  float (*present)[dim][dim] = (float (*)[dim][dim])sim->b;
  float (*trend)[dim][dim] = (float (*)[dim][dim])sim->a;

  float orthogonal, diagonal, vertices; // influence from neighboring cells.
  float orthogonal_gain, diagonal_gain, vertices_gain; // influence from neighboring cells.
  float new, old;
  float gain_acc = 0.0;
  float out_acc = 0.0;

  d3_get_context g;
  g.p = sim->b;
  g.dim = dim;
  g.out_acc = &out_acc;
  g.gain_acc = &gain_acc;


  // run the sim
  for ( uint x = 1; x < dim-1; x++ ) {
    for ( uint y = 1; y < dim-1; y++ ) {
      for ( uint z = start; z < stop; z++ ) {
        g.current = present[x][y][z];

        // 6 orthogonal cells, geometric gain = 3 / 13
        g.gain = 3.0 / 13.0;
        orthogonal = 0.0; orthogonal_gain = 0.0;
        g.out_acc = &orthogonal;
        g.gain_acc = &orthogonal_gain;
        d3_get(g, x-1, y, z );
        d3_get(g, x, y-1, z );
        d3_get(g, x, y+1, z );
        d3_get(g, x, y, z-1 );
        d3_get(g, x, y, z+1 );
        d3_get(g, x+1, y, z );
        // orthogonal_gain /= 6.0; // average
        // orthogonal = present[x-1][y  ][z  ] + present[x  ][y-1][z  ] + present[x  ][y+1][z  ]
        //            + present[x  ][y  ][z-1] + present[x  ][y  ][z+1] + present[x+1][y  ][z  ];       // 6 orthogonal cells, gain = 3 / 13


        // 12 diagonal cells, gain = 3 / 26
        g.gain = 3.0 / 26.0;
        diagonal = 0.0; diagonal_gain = 0.0;
        g.out_acc = &diagonal;
        g.gain_acc = &diagonal_gain;
        d3_get(g, x-1, y-1, z );
        d3_get(g, x, y-1, z-1 );
        d3_get(g, x+1, y-1, z );

        d3_get(g, x-1, y, z-1 );
        d3_get(g, x, y-1, z+1 );
        d3_get(g, x+1, y, z-1 );

        d3_get(g, x-1, y, z+1 );
        d3_get(g, x, y+1, z-1 );
        d3_get(g, x+1, y, z+1 );

        d3_get(g, x-1, y+1, z );
        d3_get(g, x, y+1, z+1 );
        d3_get(g, x+1, y+1, z );
        // diagonal_gain /= 12.0; // average
        // diagonal   = present[x-1][y-1][z  ] + present[x  ][y-1][z-1] + present[x+1][y-1][z  ]
                   // + present[x-1][y  ][z-1] + present[x  ][y-1][z+1] + present[x+1][y  ][z-1]
                   // + present[x-1][y  ][z+1] + present[x  ][y+1][z-1] + present[x+1][y  ][z+1]
                   // + present[x-1][y+1][z  ] + present[x  ][y+1][z+1] + present[x+1][y+1][z  ];        // 12 diagonal cells, gain = 3 / 26


        // 8, gain 1 / 13
        g.gain = 1.0 / 13.0;
        vertices = 0.0; vertices_gain = 0.0;
        g.out_acc = &vertices;
        g.gain_acc = &vertices_gain;
        d3_get(g, x-1, y-1, z-1 );
        d3_get(g, x-1, y-1, z+1 );
        d3_get(g, x-1, y+1, z-1 );
        d3_get(g, x-1, y+1, z+1 );

        d3_get(g, x+1, y-1, z-1 );
        d3_get(g, x+1, y-1, z+1 );
        d3_get(g, x+1, y+1, z-1 );
        d3_get(g, x+1, y+1, z+1 );
        // vertices_gain /= 8.0; // average
        // vertices   = present[x-1][y-1][z-1] + present[x-1][y-1][z+1] + present[x-1][y+1][z-1] + present[x-1][y+1][z+1]
                   // + present[x+1][y-1][z-1] + present[x+1][y-1][z+1] + present[x+1][y+1][z-1] + present[x+1][y+1][z+1]; // 8, gain 1 / 13

        float present_gain = orthogonal_gain + diagonal_gain + vertices_gain - 2;
        // printf("present_gain: %f\n", present_gain);

        new = orthogonal + diagonal + vertices - present_gain * present[x][y][z];
        // new = .23076923 * orthogonal + .1153846 * diagonal + .076923 * vertices - 1.384615 * present[x][y][z];

        old = past[x][y][z];
        trend[x][y][z] = new - old;
        // trend[x][y][z] = .23076923 * orthogonal + .1153846 * diagonal + .076923 * vertices - 1.384615 * present[x][y][z] - past[x][y][z];
      }
    }
  }
}



// void step_slice_3d( uint nslice, uint slices, sim_3d* sim ) {
//   uint dim = sim->dim;
//   uint slice = dim / slices + (dim % slices) / slices;
//   uint start = nslice * slice;
//   uint stop = start + slice;

//   // account for remainders and offsets
//   if ( start == 0 ) { start = 1; }
//   if ( nslice == slices - 1) { stop = dim - 1; }

//   float (*past)[dim][dim] = (float (*)[dim][dim])sim->a;
//   float (*present)[dim][dim] = (float (*)[dim][dim])sim->b;
//   float (*trend)[dim][dim] = (float (*)[dim][dim])sim->a;

//   float orthogonal, diagonal, vertices; // influence from neighboring cells.
//   float new, old;

//   // run the sim
//   for ( uint x = 1; x < dim-1; x++ ) {
//     for ( uint y = 1; y < dim-1; y++ ) {
//       for ( uint z = start; z < stop; z++ ) {
//         orthogonal = present[x-1][y  ][z  ] + present[x  ][y-1][z  ] + present[x  ][y+1][z  ]
//                    + present[x  ][y  ][z-1] + present[x  ][y  ][z+1] + present[x+1][y  ][z  ];       // 6 orthogonal cells, gain = 3 / 13
//         diagonal   = present[x-1][y-1][z  ] + present[x  ][y-1][z-1] + present[x+1][y-1][z  ]
//                    + present[x-1][y  ][z-1] + present[x  ][y-1][z+1] + present[x+1][y  ][z-1]
//                    + present[x-1][y  ][z+1] + present[x  ][y+1][z-1] + present[x+1][y  ][z+1]
//                    + present[x-1][y+1][z  ] + present[x  ][y+1][z+1] + present[x+1][y+1][z  ];        // 12 diagonal cells, gain = 3 / 26
//         vertices   = present[x-1][y-1][z-1] + present[x-1][y-1][z+1] + present[x-1][y+1][z-1] + present[x-1][y+1][z+1]
//                    + present[x+1][y-1][z-1] + present[x+1][y-1][z+1] + present[x+1][y+1][z-1] + present[x+1][y+1][z+1]; // 8, gain 1 / 13

//         new = .23076923 * orthogonal + .1153846 * diagonal + .076923 * vertices - 1.384615 * present[x][y][z];

//         old = past[x][y][z];
//         trend[x][y][z] = new - old;
//         // trend[x][y][z] = .23076923 * orthogonal + .1153846 * diagonal + .076923 * vertices - 1.384615 * present[x][y][z] - past[x][y][z];
//       }
//     }
//   }
// }

typedef struct slice_3d{
  uint nslice, slices;
  sim_3d* sim;
} slice_3d;

void* worker_slice_3d(void* arg) {
  slice_3d* slice = (slice_3d*)arg;
  step_slice_3d( slice->nslice, slice->slices, slice->sim );
  return NULL;
}

void do_step_3d( sim_3d* sim ) {
  slice_3d slices[NTHREADS];
  pthread_t threads[NTHREADS];

  for ( uint i = 0; i < NTHREADS; i++ ) {
    slices[i].nslice = i;
    slices[i].slices = NTHREADS;
    slices[i].sim = sim;
    if (pthread_create(&threads[i], NULL, worker_slice_3d, &slices[i]) != 0) {
      fprintf(stderr, "Error creating thread %d\n", i);
      return;
    }
  } 
  
  // Wait for all threads to complete
  for (int i = 0; i < NTHREADS; i++) {
    if (pthread_join(threads[i], NULL) != 0) {
      fprintf(stderr, "Error joining thread %d\n", i);
      return;
    }
  }
}

void step_3d( sim_3d* sim, Image* img ) {
  do_emit_3d( sim );
  sim->dim < 200 ? step_slice_3d( 0, 1, sim ) : do_step_3d(sim);

  // swap a & b
  void* t = sim->a;
  sim->a = sim->b;
  sim->b = t;

  sim->count++;
  render_3d( img, sim );
}
*/