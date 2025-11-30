#include "common.h"
#include "cah_bi.h"
#include <time.h>

#define DIM_X  400
#define DIM_Y  400

// #define STEP_THREADS 1
// #define DRAW_THREADS 1

// #define LAMBDA      16
// #define INIT_VALUE  0.00001

typedef struct {
    Image image;
    bool ready;
    pthread_mutex_t mutex;
    // float time;
} SharedImage;

void *main_sim_thread(void *arg) {
    SharedImage *shared = (SharedImage *)arg;

    // allocate the sim
    void* sim = cah_bi_init( DIM_X, DIM_Y );
    // void* sim = init_sim_3d( DIM );
    // void* sim = init_d2( DIM );

    while (true) {
      // sleep(1);
      // start = clock();
      cah_bi_step( sim );
      cah_bi_render( sim, &shared->image );
      // end = clock();

      // Signal main thread
      pthread_mutex_lock(&shared->mutex);
      shared->ready = true;
      // shared->time = ((float)(end - start)) / CLOCKS_PER_SEC;
      pthread_mutex_unlock(&shared->mutex);
    }

    cah_bi_free( sim );
    // free_d2( sim );
    return NULL;
}

// clock_t start = clock();
// my_function();
// clock_t end = clock();

// double cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;


int main(void) {
  char buff[200];
  struct timespec start, end;
  double elapsed = 0.0;

  // Suppress info messages - only show warnings and errors
  SetTraceLogLevel(LOG_WARNING);

  sprintf(buff, "%s %dx%d", cah_bi_name, DIM_X, DIM_Y );
  InitWindow(DIM_X, DIM_Y + 40, buff );
  SetTargetFPS(60);

  SharedImage shared = {
      .image = GenImageColor(DIM_X, DIM_Y, BLACK),
      .ready = false
      // .time = 0.0
  };
  pthread_mutex_init(&shared.mutex, NULL);
  
  // Start main_sim_thread
  pthread_t thread;
  pthread_create(&thread, NULL, main_sim_thread, &shared);
  
  Texture2D texture = LoadTextureFromImage(shared.image);


  clock_gettime(CLOCK_MONOTONIC, &start);
  while (!WindowShouldClose()) {
      // Check if thread has new data
      pthread_mutex_lock(&shared.mutex);
      if (shared.ready) {
          UpdateTexture(texture, shared.image.data);  // Main thread only
          shared.ready = false;
          clock_gettime(CLOCK_MONOTONIC, &end);
          elapsed = (end.tv_sec - start.tv_sec) + 
                 (end.tv_nsec - start.tv_nsec) / 1000000000.0;
          start = end;
      }
      pthread_mutex_unlock(&shared.mutex);

      // prepare the step time string
      sprintf(buff, "step: %.2fs", elapsed );
      
      BeginDrawing();
          ClearBackground(DARKGRAY);
          DrawTexture(texture, 0, 40, WHITE);
          DrawText(buff, 10, 10, 20, GREEN);
      EndDrawing();
  }

  UnloadImage(shared.image);
  UnloadTexture(texture);
  CloseWindow();
  return 0;
}