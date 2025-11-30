#pragma once
#include "raylib.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define rot6l(n) (((n << 1) & 63) | ((n & 32) >> 5))
#define rot6r(n) (n >> 1) | ((n & 1) << 5)
#define spin(n) rot6r(rot6r(rot6r(n)))


// fast way to count on bits - uses cpu instruction
#define bits(x) __builtin_popcount(x)
#define bitsll(x) __builtin_popcountll(x)  // unsigned long long
