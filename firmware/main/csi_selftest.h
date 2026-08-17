// Boot-time parity check: does this board reproduce the PC's predictions?
//
// Run once at startup before trusting any live detection. Returns true only if
// every embedded test vector predicts exactly what scikit-learn predicted for
// the same raw frames on the laptop.
#pragma once

#include <stdbool.h>

bool csi_selftest_run(void);
