#include "common.h"

float get_time_mark()
// double get_time_mark()
{
    timeval t;
    gettimeofday(&t, nullptr);
    return t.tv_sec + t.tv_usec * 0.000001f;
}